"""Stage 3, judge-robust GRPO on the stage 2 MOSS baseline (GPU, dwnd-01; judges on the RTX 3070).

    # once, only for arm d: genuine same-person EN->FIL ECAPA similarity on held-out bilingual speakers
    python -m training.fil_moss.stage3 ceiling --bilingual /abs/bilingual.jsonl --out /abs/run/ceiling.json
    # one arm per run, each under the guard
    python -m training.fil_moss.guard --max-seconds 21600 -- python -m training.fil_moss.stage3 run --arm c \
        --stage2-report /abs/run/stage2/report.json --codes /abs/run/codes --prompts /abs/prompts.txt \
        --out /abs/run/stage3-c [--bilingual /abs/bilingual.jsonl --ceiling /abs/run/ceiling.json]
    # after the arms: pre-registered adoption decisions
    python -m training.fil_moss.stage3 compare --reports /abs/run/stage3-*/report.json

Starts from stage 2's selected checkpoint (it refuses if stage 2 kept the pretrained model) and keeps stage 2's
conditioning. Per step: one unpaired Filipino prompt, ``rollouts_per_prompt`` sampled rollouts in one batch on
the BF16 mirror; each scored by every judge, ECAPA and pacing; rewards are ranks within the group
(``grpo.group_rewards``); advantages are group-normalised; the update is the clipped surrogate plus k3 KL to the
frozen stage 2 checkpoint (``objective.grpo_token_loss``), in micro-batches. Rollouts that hit the frame cap lose
their artificial EOS label. The judge gap is logged as an alarm, never rewarded. Tripwires (silence fraction,
codec-token entropy) abort the run when the reward rises while they degrade; two aborts kill the arm.

Speaker term: with an unconditioned stage 2 (the MOSS baseline) it is similarity to the centroid of the stage 2
checkpoint's own voice, measured at start ("anchor"); with a reference-conditioned stage 2 it is similarity to
the conditioning clip, and arm d may add the CLVI term.

Pipelining (``stage3.pipeline``): step t's rollouts are judged on the 3070 while the GB10 applies step t-1's
update and generates step t+1. Rollouts are then one update old when trained on; their behaviour
log-probabilities are recorded at generation time and the clipped ratio corrects for the lag.

``bilingual.jsonl`` rows: {"speaker_id", "consent_signed": bool, "role": "train"|"heldout", "en": [wav paths],
"fil": [wav paths]}. Only consented speakers are used, for rewards and for evaluation.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import get_scheduler

from . import evaluation as E
from . import gates as G
from . import grpo as P
from . import judges as J
from . import runtime as R
from .objective import grpo_token_loss
from .scoring import cer, judge_gap, silence_fraction, token_entropy
from .stage1 import load_codes, sha

HERE = Path(__file__).resolve().parent


def _wav(path, sr_to):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim != 1:
        wav = wav.mean(axis=1)
    return J.resample(wav, sr, sr_to)


def load_bilingual(path):
    rows = [r for r in R.read_jsonl(path) if r.get("consent_signed") is True]
    return [r for r in rows if r["role"] == "train"], [r for r in rows if r["role"] == "heldout"]


def ceiling(a):
    recipe = json.loads(Path(a.recipe).read_text())
    _, heldout = load_bilingual(a.bilingual)
    if not heldout:
        raise SystemExit("no consented held-out bilingual speakers")
    speaker = J.Speaker(recipe["judges"]["speaker"])
    sims = []
    for r in heldout:
        en = speaker.centroid([_wav(p, J.JUDGE_SR) for p in r["en"]])
        fil = speaker.centroid([_wav(p, J.JUDGE_SR) for p in r["fil"]])
        sims.append({"speaker_id": r["speaker_id"], "sim": speaker.cosine(en, fil)})
    out = {"speakers": len(sims), "ecapa_ceiling": float(np.mean([s["sim"] for s in sims])), "per_speaker": sims}
    R.write_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_speaker"}), flush=True)


def behaviour_logprobs(policy, ds, rollouts, micro):
    """Per-rollout token log-probabilities under the current policy (no grad), recorded at generation time."""
    out = [None] * len(rollouts)
    idx = [i for i, r in enumerate(rollouts) if len(r["codes"]) > 0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for part in R.chunks(idx, micro):
            inputs = R.batch_many(ds, [rollouts[i]["record"] for i in part])
            inputs = R.drop_eos_label(inputs, [j for j, i in enumerate(part) if rollouts[i]["truncated"]])
            logp, _, _, sample = R.token_logprobs(policy, inputs)
            for j, i in enumerate(part):
                out[i] = logp[sample == j].detach()
    return out


def update(policy, reference, ds, rollouts, adv, old, cfg, micro):
    """One clipped-surrogate + KL update over a group, in micro-batches. Returns (mean KL, rollouts used)."""
    usable = [i for i, r in enumerate(rollouts) if len(r["codes"]) > 0]
    kls = []
    for part in R.chunks(usable, micro):
        inputs = R.batch_many(ds, [rollouts[i]["record"] for i in part])
        inputs = R.drop_eos_label(inputs, [j for j, i in enumerate(part) if rollouts[i]["truncated"]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logp, valid, _, sample = R.token_logprobs(policy, inputs)
            with torch.no_grad():
                logp_ref, _, _, _ = R.token_logprobs(reference, inputs)
        total = 0.0
        for j, i in enumerate(part):
            m = sample == j
            lp = logp[m]
            behaviour = lp.detach() if old is None else old[i]
            loss, kl = grpo_token_loss(lp, behaviour, logp_ref[m], float(adv[i]), valid[m], cfg["clip_epsilon"],
                                       cfg["kl_beta"])
            total = total + loss
            kls.append(float(kl))
        (total / len(usable)).backward()
    return (float(np.mean(kls)) if kls else None), len(usable)


def run(a):
    recipe = json.loads(Path(a.recipe).read_text())
    cfg, s2cfg, gen, seed = recipe["stage3"], recipe["stage2"], recipe["generation"], recipe["baseline"]["seed"]
    micro, pipeline = cfg["logprob_microbatch"], cfg["pipeline"]
    arm = cfg["arms"][a.arm]
    s2 = json.loads(Path(a.stage2_report).read_text())
    entry = G.stage3_entry(s2, a.codes)
    if entry["verdict"] != G.PASS:
        raise SystemExit("stage 3 refused: %s" % "; ".join(entry["reasons"]))
    cond = s2["conditioning"]
    mode = P.speaker_mode(arm, cond)

    bi_train, bi_heldout = load_bilingual(a.bilingual) if a.bilingual else ([], [])
    ceiling_value = json.loads(Path(a.ceiling).read_text())["ecapa_ceiling"] if a.ceiling else None
    clvi = G.clvi_mode(len(bi_train), ceiling_value, cfg)
    if mode == "conditioning+clvi" and clvi["verdict"] != "reward":
        raise SystemExit("arm %s needs the CLVI reward, which is %s: %s"
                         % (a.arm, clvi["verdict"], "; ".join(clvi["reasons"])))

    out = R.new_output_dir(a.out)
    ledger = Path(a.ledger or (out.parent / ("stage3-arm-%s-ledger.jsonl" % a.arm)))
    aborts = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    if aborts >= cfg["kill"]["max_tripwire_aborts"]:
        raise SystemExit("arm %s already aborted %d times; killed" % (a.arm, aborts))

    R.setup()
    R.seed_all(seed)
    rng = random.Random(seed)
    proc = R.processor()
    train_rows = load_codes(a.codes, "train")
    val_rows = load_codes(a.codes, "validation")
    if cond == "reference":
        val_rows = [r for r in val_rows if r.get("ref_audio_codes") is not None]
    test_texts = [r["conversations"][0]["text"] for r in R.read_jsonl(Path(a.codes) / "test_codes.jsonl")]
    prompts, dropped = P.filter_prompts(Path(a.prompts).read_text().splitlines(),
                                        [r["conversations"][0]["text"] for r in val_rows] + test_texts)
    if not prompts:
        raise SystemExit("no usable prompts (%s)" % dropped)
    rng.shuffle(prompts)
    voices = [r for r in train_rows if r.get("ref_audio_codes") is not None] if cond == "reference" else []
    rate = P.speaking_rate([{"text": r["conversations"][0]["text"], "seconds": r["seconds"]} for r in train_rows])
    ds = R.dataset([], proc)
    val_ds = R.dataset(val_rows, proc)
    subset = E.generation_subset(val_rows, cfg["eval_generation_clips"], seed, cond)
    if not subset or (cond == "reference" and not voices):
        raise SystemExit("need validation rows (and, for reference conditioning, training voices)")

    start_ckpt = s2["best_checkpoint"]
    policy = R.load_model(start_ckpt)
    params = R.freeze(policy, s2cfg["frozen"])
    reference = R.load_model(start_ckpt, dtype=torch.bfloat16, train=False)
    mirror = R.mirror(start_ckpt)
    optimizer = R.adamw8bit(params, cfg["learning_rate"], s2cfg)
    scheduler = get_scheduler("constant", optimizer)
    codec = R.load_codec()
    eng = R.engine(mirror, codec)
    pool = J.make_pool(recipe)
    if arm["intelligibility"].startswith("single:") and arm["intelligibility"].split(":")[1] not in pool.names:
        raise SystemExit("arm %s needs judge %s, which failed: %s" % (a.arm, arm["intelligibility"], pool.failed))
    speaker = J.Speaker(recipe["judges"]["speaker"])

    anchor = None
    if mode == "anchor":  # centroid of the starting checkpoint's own (unconditioned) voice
        embs = []
        for part, audio in E._batches(eng, E.generation_subset(train_rows, cfg["anchor_clips"], seed + 1, "none"),
                                      "none", gen, cfg["gen_batch"], seed + 1):
            embs += [speaker.embed(J.resample(a24, R.SAMPLE_RATE)) for a24, _ in audio if len(a24) > R.SAMPLE_RATE // 4]
        anchor = torch.nn.functional.normalize(torch.stack(embs).mean(0), dim=0)

    def encode_en(path):
        return R.encode_wav(codec, _wav(path, R.SAMPLE_RATE))

    clvi_pool = []
    if mode == "conditioning+clvi":
        for r in bi_train:
            clvi_pool.append({"en_codes": [encode_en(p) for p in r["en"]],
                              "en_emb": [speaker.embed(_wav(p, J.JUDGE_SR)) for p in r["en"]],
                              "fil_centroid": speaker.centroid([_wav(p, J.JUDGE_SR) for p in r["fil"]])})
    xling = []
    if cond == "reference":
        for i, r in enumerate(bi_heldout):
            text = subset[i % len(subset)]["conversations"][0]["text"]
            xling.append({"text": text, "ref_codes": encode_en(r["en"][0]),
                          "target_centroid": speaker.centroid([_wav(p, J.JUDGE_SR) for p in r["fil"]]),
                          "cap": R.max_frames(P.expected_seconds(text, rate), gen)})

    report = {"arm": a.arm, "arm_config": arm, "speaker_mode": mode, "conditioning": cond,
              "stage2_report": a.stage2_report, "start_checkpoint": start_ckpt, "recipe_sha256": sha(a.recipe),
              "judges": pool.names, "judges_failed": pool.failed, "judge_events": pool.events, "pipeline": pipeline,
              "logprob_microbatch": micro, "clvi": clvi, "ecapa_ceiling": ceiling_value, "prompts": len(prompts),
              "prompts_dropped": dropped, "speaking_rate_chars_per_second": rate, "prior_aborts": aborts,
              "evals": [], "steps": [], "status": "running"}
    cache, voice_emb = {}, {}

    def xling_sim():
        if not xling:
            return None
        sims = []
        for start in range(0, len(xling), cfg["gen_batch"]):
            part = xling[start:start + cfg["gen_batch"]]
            R.seed_all(seed)
            outs = R.generate(eng, [p["text"] for p in part], [p["ref_codes"] for p in part], [p["cap"] for p in part], gen)
            for p, (codes, _) in zip(part, outs):
                w = J.resample(R.decode(codec, codes), R.SAMPLE_RATE)
                sims.append(speaker.cosine(speaker.embed(w), p["target_centroid"]) if len(w) else 0.0)
        return float(np.mean(sims))

    def evaluate(step):
        started = time.monotonic()
        m = E.evaluate(policy, mirror, eng, val_ds, val_rows, subset, pool, speaker,
                       dict(s2cfg, gen_batch=cfg["gen_batch"], samples_per_eval=cfg["samples_per_eval"],
                            conditioning=cond), gen, seed, cache, out / "samples" / ("step-%05d" % step))
        m.update(step=step, xling_sim=xling_sim(), eval_seconds=time.monotonic() - started)
        report["evals"].append(m)
        print(json.dumps({"eval": m}), flush=True)
        return m

    def produce(step):
        """Generate one group on the mirror, decode, send to the judges; record behaviour log-probs."""
        t0 = time.monotonic()
        text = prompts[(step - 1) % len(prompts)]
        use_clvi = bool(clvi_pool) and rng.random() < arm.get("clvi_batch_fraction", 0.0)
        clvi_target = None
        if use_clvi:
            who = rng.choice(clvi_pool)
            j = rng.randrange(len(who["en_codes"]))
            ref_codes, target, clvi_target = who["en_codes"][j], who["en_emb"][j], who["fil_centroid"]
        elif mode == "anchor":
            ref_codes, target = None, anchor
        else:
            voice = rng.choice(voices)
            ref_codes = voice["conversations"][0]["audio_codes"]
            if voice["id"] not in voice_emb:
                voice_emb[voice["id"]] = speaker.embed(_wav(voice["conversations"][0]["wav"], J.JUDGE_SR))
            target = voice_emb[voice["id"]]
        expected = P.expected_seconds(text, rate)
        k = cfg["rollouts_per_prompt"]
        outs = R.generate(eng, [text] * k, [ref_codes] * k, R.max_frames(expected, gen), gen, do_sample=True)
        rollouts = [{"codes": c, "truncated": tr, "record": R.record(text, c, ref_codes),
                     "wav16": J.resample(R.decode(codec, c), R.SAMPLE_RATE)} for c, tr in outs]
        handle = pool.submit([r["wav16"] for r in rollouts])
        old = behaviour_logprobs(policy, ds, rollouts, micro) if pipeline else None
        return {"step": step, "text": text, "expected": expected, "use_clvi": use_clvi, "target": target,
                "clvi_target": clvi_target, "rollouts": rollouts, "handle": handle, "old": old,
                "produce_seconds": time.monotonic() - t0}

    def consume(g):
        """Collect judges, compute rewards, update, sync the mirror; returns the step's log row."""
        t0 = time.monotonic()
        hyps = pool.collect(g["handle"])
        wait = time.monotonic() - t0
        scored, sil, ent = [], [], []
        for r, h in zip(g["rollouts"], hyps):
            emb = speaker.embed(r["wav16"]) if len(r["wav16"]) >= J.JUDGE_SR // 4 else None
            scored.append({
                "cers": {k: cer(g["text"], x) for k, x in h.items()},
                "voice_sim": speaker.cosine(emb, g["target"]) if emb is not None else 0.0,
                "clvi_sim": ((speaker.cosine(emb, g["clvi_target"]) if emb is not None else 0.0)
                             if g["use_clvi"] else None),
                "duration_ratio": max(J.trimmed_seconds(r["wav16"], J.JUDGE_SR), 1e-3) / g["expected"]})
            sil.append(silence_fraction(r["wav16"].tolist(), J.JUDGE_SR))
            ent.append(token_entropy(np.asarray(r["codes"])[:, 0].tolist()) if len(r["codes"]) else 0.0)
        rewards, adv, _ = P.group_rewards(arm, mode, cfg, scored)
        kl, used = None, 0
        if any(x != 0.0 for x in adv):
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            kl, used = update(policy, reference, ds, g["rollouts"], adv, g["old"], cfg, micro)
            torch.nn.utils.clip_grad_norm_(params, s2cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            R.sync(mirror, policy)
        return {"step": g["step"], "reward": float(np.mean(rewards)),
                "worst_cer": float(np.mean([max(s["cers"].values()) for s in scored])),
                "judge_gap": float(np.mean([judge_gap(s["cers"]) for s in scored])), "kl": kl,
                "silence_fraction": float(np.mean(sil)), "token_entropy": float(np.mean(ent)),
                "truncated": sum(r["truncated"] for r in g["rollouts"]), "updated_rollouts": used,
                "clvi": g["use_clvi"], "produce_seconds": g["produce_seconds"], "judge_wait_seconds": wait,
                "consume_seconds": time.monotonic() - t0}

    baseline = evaluate(0)
    best, history, pending = None, [], None
    start = time.monotonic()
    status, stop_reason = "complete", None
    for step in range(1, cfg["max_steps"] + 2):
        current = produce(step) if step <= cfg["max_steps"] else None
        if not pipeline and current is not None:
            pending, current = current, None
        if pending is None:
            pending = current
            continue
        row = consume(pending)
        pending = current
        report["steps"].append(row)
        history.append(row)
        print(json.dumps(row), flush=True)
        done = row["step"]
        w = P.windows(history, cfg["tripwires"]["window_steps"])
        if w:
            trip = G.tripwire(w[0], w[1], cfg)
            if trip["verdict"] == G.KILL:
                status, stop_reason = "tripwire_abort", trip
                with ledger.open("a") as f:
                    f.write(json.dumps({"run": str(out), "step": done, "reasons": trip["reasons"]}) + "\n")
                break
        if done % cfg["eval_every_steps"] == 0 or done == cfg["max_steps"]:
            m = evaluate(done)
            verdict = G.stage3_evaluation_verdict(m, done, baseline, cfg)
            m["verdict"] = verdict
            # selection: lowest worst-judge CER among evaluations that passed the stop rules and beat step 0
            if (verdict["verdict"] == G.CONTINUE and m["worst_cer"] < baseline["worst_cer"]
                    and (best is None or m["worst_cer"] < best["worst_cer"])):
                R.save(policy, proc, out / "best_checkpoint")
                best = m
            report["judge_events"] = pool.events
            R.write_json(out / "progress.json", report)
            if verdict["verdict"] == G.KILL:
                status, stop_reason = "killed", verdict
                break
    R.save(policy, proc, out / "last_checkpoint")
    final = report["evals"][-1]
    report.update(status=status, stop_reason=stop_reason, train_seconds=time.monotonic() - start,
                  cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, best=best,
                  best_checkpoint=str(out / "best_checkpoint") if best else None,
                  last_checkpoint=str(out / "last_checkpoint"), judge_events=pool.events,
                  final={"worst_cer": final["worst_cer"], "voice": G.voice_measure(final),
                         "judge_gap": final["judge_gap"], "xling_sim": final.get("xling_sim"),
                         "start_worst_cer": baseline["worst_cer"], "start_voice": G.voice_measure(baseline),
                         "start_xling_sim": baseline.get("xling_sim"), "steps": len(report["steps"]),
                         "aborts": aborts + (1 if status == "tripwire_abort" else 0)},
                  quality="native listening pending")
    R.write_json(out / "report.json", report)
    print("STAGE3_ARM", a.arm, status, flush=True)
    return 0 if status == "complete" else 3


def compare(a):
    recipe = json.loads(Path(a.recipe).read_text())
    final = {}
    for path in a.reports:
        rep = json.loads(Path(path).read_text())
        if rep["arm"] in final:
            raise SystemExit("two reports for arm %s" % rep["arm"])
        final[rep["arm"]] = rep["final"]
    print(json.dumps({"final": final, "decisions": G.arm_decisions(final, recipe["stage3"])}, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--recipe", default=str(HERE / "recipe.json"))
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("ceiling")
    c.add_argument("--bilingual", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=("a", "b", "c", "d"))
    r.add_argument("--stage2-report", required=True)
    r.add_argument("--codes", required=True)
    r.add_argument("--prompts", required=True, help="unpaired Filipino text, one prompt per line")
    r.add_argument("--out", required=True)
    r.add_argument("--bilingual")
    r.add_argument("--ceiling")
    r.add_argument("--ledger")
    m = sub.add_parser("compare")
    m.add_argument("--reports", nargs="+", required=True)
    a = p.parse_args(argv)
    return {"ceiling": ceiling, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
