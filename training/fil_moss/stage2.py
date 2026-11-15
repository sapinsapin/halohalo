"""Stage 2, judge-robust twin-anchored GRPO on the stage 1 checkpoint (GPU, dwnd-01).

    # once: genuine same-person EN->FIL ECAPA similarity on held-out bilingual speakers (gates arm d)
    python -m training.fil_moss.stage2 ceiling --bilingual /abs/bilingual.jsonl --out /abs/run/ceiling.json
    # one arm per run, each under the guard
    python -m training.fil_moss.guard --max-seconds 14400 -- python -m training.fil_moss.stage2 run --arm c \
        --stage1-report /abs/run/stage1/report.json --codes /abs/run/codes --prompts /abs/prompts.txt \
        --out /abs/run/stage2-c [--bilingual /abs/bilingual.jsonl --ceiling /abs/run/ceiling.json]
    # after all arms: pre-registered adoption decisions
    python -m training.fil_moss.stage2 compare --reports /abs/run/stage2-{a,b,c,d}/report.json

Per step: one unpaired Filipino prompt, ``rollouts_per_prompt`` sampled rollouts from the policy with one
training speaker's clip as the voice, each scored by every judge, ECAPA and pacing; rewards are ranks within the
group (``grpo.group_rewards``); advantages are group-normalised; the update is the clipped surrogate plus k3 KL
to the frozen stage 1 model (``objective.grpo_token_loss``). Truncated rollouts lose their artificial EOS label.
The judge gap is logged as an alarm. Tripwires (silence fraction, codec-token entropy) abort the run when the
reward rises while they degrade; aborts are appended to the arm's ledger, and two aborts kill the arm.

``bilingual.jsonl`` rows: {"speaker_id", "consent_signed": bool, "role": "train"|"heldout",
"en": [wav paths], "fil": [wav paths]}. Only consented speakers are used, for rewards and for evaluation.
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

HERE = Path(__file__).resolve().parent


def _wav(path, sr_to):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim != 1:
        wav = wav.mean(axis=1)
    return J.resample(wav, sr, sr_to)


def load_bilingual(path):
    rows = [r for r in R.read_jsonl(path) if r.get("consent_signed") is True]
    return [r for r in rows if r["role"] == "train"], [r for r in rows if r["role"] == "heldout"]


# ------------------------------------------------------------------------------------------------- ceiling

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


# ----------------------------------------------------------------------------------------------------- run

def run(a):
    recipe = json.loads(Path(a.recipe).read_text())
    cfg, gen, seed = recipe["stage2"], recipe["generation"], recipe["baseline"]["seed"]
    arm = cfg["arms"][a.arm]
    s1 = json.loads(Path(a.stage1_report).read_text())
    entry = G.stage2_entry(s1)
    if entry["verdict"] != G.PASS:
        raise SystemExit("stage 2 refused: %s" % "; ".join(entry["reasons"]))
    if s1.get("codes") != a.codes:
        raise SystemExit("stage 1 trained on different codes (%s)" % s1.get("codes"))

    bi_train, bi_heldout = load_bilingual(a.bilingual) if a.bilingual else ([], [])
    ceiling_value = json.loads(Path(a.ceiling).read_text())["ecapa_ceiling"] if a.ceiling else None
    clvi = G.clvi_mode(len(bi_train), ceiling_value, cfg)
    if arm["speaker"] == "conditioning+clvi" and clvi["verdict"] != "reward":
        raise SystemExit("arm %s needs the CLVI reward, which is %s: %s"
                         % (a.arm, clvi["verdict"], "; ".join(clvi["reasons"])))

    out = R.new_output_dir(a.out)
    ledger = Path(a.ledger or (out.parent / ("stage2-arm-%s-ledger.jsonl" % a.arm)))
    aborts = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    if aborts >= cfg["kill"]["max_tripwire_aborts"]:
        raise SystemExit("arm %s already aborted %d times; killed" % (a.arm, aborts))

    R.setup()
    R.seed_all(seed)
    rng = random.Random(seed)
    proc = R.processor()
    train_rows = R.read_jsonl(Path(a.codes) / "train_codes.jsonl")
    val_rows = R.read_jsonl(Path(a.codes) / "validation_codes.jsonl")
    test_texts = [r["conversations"][0]["text"] for r in R.read_jsonl(Path(a.codes) / "test_codes.jsonl")]
    heldout_texts = [r["conversations"][0]["text"] for r in val_rows] + test_texts
    prompts, dropped = P.filter_prompts(Path(a.prompts).read_text().splitlines(), heldout_texts)
    if not prompts:
        raise SystemExit("no usable prompts (%s)" % dropped)
    rng.shuffle(prompts)
    voices = [r for r in train_rows if r.get("ref_audio_codes") is not None]
    rate = P.speaking_rate([{"text": r["conversations"][0]["text"], "seconds": r["seconds"]} for r in train_rows])
    ds = R.dataset(voices[:1], proc)
    subset = E.generation_subset(val_rows, cfg["eval_generation_clips"], seed)
    if not subset or not voices:
        raise SystemExit("need validation and training rows with a reference voice")

    policy = R.load_model(s1["best_checkpoint"])
    params = R.freeze(policy, recipe["stage1"]["train"]["frozen"])
    reference = R.load_model(s1["best_checkpoint"], dtype=torch.bfloat16, train=False)
    optimizer = R.adamw8bit(params, cfg["learning_rate"], recipe["stage1"]["train"])
    scheduler = get_scheduler("constant", optimizer)
    codec = R.load_codec()
    eng = R.engine(policy, codec)
    judges, failed = J.load(recipe["judges"]["fil"])
    if arm["intelligibility"].startswith("single:") and arm["intelligibility"].split(":")[1] not in judges:
        raise SystemExit("arm %s needs judge %s, which failed: %s" % (a.arm, arm["intelligibility"], failed))
    speaker = J.Speaker(recipe["judges"]["speaker"])

    def encode_en(path):
        return R.encode_wav(codec, _wav(path, R.SAMPLE_RATE))

    clvi_pool = []
    if clvi["verdict"] == "reward":
        for r in bi_train:
            clvi_pool.append({"speaker_id": r["speaker_id"], "en_codes": [encode_en(p) for p in r["en"]],
                              "en_emb": [speaker.embed(_wav(p, J.JUDGE_SR)) for p in r["en"]],
                              "fil_centroid": speaker.centroid([_wav(p, J.JUDGE_SR) for p in r["fil"]])})
    xling = []
    for i, r in enumerate(bi_heldout):
        text = subset[i % len(subset)]["conversations"][0]["text"]
        xling.append({"text": text, "ref_codes": encode_en(r["en"][0]),
                      "target_centroid": speaker.centroid([_wav(p, J.JUDGE_SR) for p in r["fil"]]),
                      "frame_cap": R.max_frames(P.expected_seconds(text, rate), gen)})

    report = {"arm": a.arm, "arm_config": arm, "stage1_report": a.stage1_report, "start_checkpoint":
              s1["best_checkpoint"], "judges": sorted(judges), "judges_failed": failed, "clvi": clvi,
              "ecapa_ceiling": ceiling_value, "prompts": len(prompts), "prompts_dropped": dropped,
              "speaking_rate_chars_per_second": rate, "prior_aborts": aborts, "evals": [], "steps": [],
              "status": "running"}

    def evaluate(step):
        m = E.generate_and_score(policy, eng, subset, judges, speaker, gen, [seed], xling=xling or None)
        m["step"] = step
        report["evals"].append(m)
        print(json.dumps({"eval": m}), flush=True)
        return m

    baseline = evaluate(0)
    best = None
    history = []
    start = time.monotonic()
    status, stop_reason = "complete", None
    for step in range(1, cfg["max_steps"] + 1):
        text = prompts[(step - 1) % len(prompts)]
        use_clvi = bool(clvi_pool) and rng.random() < arm.get("clvi_batch_fraction", 0.0)
        if use_clvi:
            who = rng.choice(clvi_pool)
            j = rng.randrange(len(who["en_codes"]))
            ref_codes, cond_emb, clvi_target = who["en_codes"][j], who["en_emb"][j], who["fil_centroid"]
        else:
            voice = rng.choice(voices)
            ref_codes = voice["conversations"][0]["audio_codes"]
            cond_emb = speaker.embed(_wav(voice["conversations"][0]["wav"], J.JUDGE_SR))
            clvi_target = None
        expected = P.expected_seconds(text, rate)
        cap = R.max_frames(expected, gen)
        policy.eval()
        outs = R.generate(eng, [text] * cfg["rollouts_per_prompt"], [ref_codes] * cfg["rollouts_per_prompt"],
                          cap, gen, do_sample=True)
        rollouts, sil, ent = [], [], []
        for codes, truncated in outs:
            wav16 = J.resample(R.decode(codec, codes), R.SAMPLE_RATE)
            hyps = J.transcribe_all(judges, wav16)
            emb = speaker.embed(wav16) if len(wav16) >= J.JUDGE_SR // 4 else None
            rollouts.append({
                "codes": codes, "truncated": truncated,
                "cers": {k: cer(text, h) for k, h in hyps.items()},
                "cond_sim": speaker.cosine(emb, cond_emb) if emb is not None else 0.0,
                "clvi_sim": (speaker.cosine(emb, clvi_target) if emb is not None else 0.0) if use_clvi else None,
                "duration_ratio": max(J.trimmed_seconds(wav16, J.JUDGE_SR), 1e-3) / expected})
            sil.append(silence_fraction(wav16.tolist(), J.JUDGE_SR))
            ent.append(token_entropy(np.asarray(codes)[:, 0].tolist()) if len(codes) else 0.0)
        rewards, adv, _ = P.group_rewards(arm, cfg, rollouts)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        kls, updated = [], 0
        if any(x != 0.0 for x in adv):
            usable = [(r, x) for r, x in zip(rollouts, adv) if len(r["codes"]) > 0]
            for r, x in usable:
                inputs = R.batch(ds, R.record(text, r["codes"], ref_codes))
                if r["truncated"]:
                    inputs = R.drop_eos_label(inputs)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logp, valid, _ = R.token_logprobs(policy, inputs)
                    with torch.no_grad():
                        logp_ref, _, _ = R.token_logprobs(reference, inputs)
                loss, kl = grpo_token_loss(logp, logp.detach(), logp_ref, float(x), valid, cfg["clip_epsilon"],
                                           cfg["kl_beta"])
                (loss / len(usable)).backward()
                kls.append(float(kl))
                updated += 1
            torch.nn.utils.clip_grad_norm_(params, recipe["stage1"]["train"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
        gaps = [judge_gap(r["cers"]) for r in rollouts]
        row = {"step": step, "reward": float(np.mean(rewards)),
               "worst_cer": float(np.mean([max(r["cers"].values()) for r in rollouts])),
               "judge_gap": float(np.mean(gaps)), "kl": float(np.mean(kls)) if kls else None,
               "silence_fraction": float(np.mean(sil)), "token_entropy": float(np.mean(ent)),
               "truncated": sum(r["truncated"] for r in rollouts), "updated_rollouts": updated, "clvi": use_clvi}
        report["steps"].append(row)
        history.append(row)
        print(json.dumps(row), flush=True)
        w = P.windows(history, cfg["tripwires"]["window_steps"])
        if w:
            trip = G.tripwire(w[0], w[1], cfg)
            if trip["verdict"] == G.KILL:
                status, stop_reason = "tripwire_abort", trip
                with ledger.open("a") as f:
                    f.write(json.dumps({"run": str(out), "step": step, "reasons": trip["reasons"]}) + "\n")
                break
        if step % cfg["eval_every_steps"] == 0 or step == cfg["max_steps"]:
            m = evaluate(step)
            verdict = G.stage2_evaluation_verdict(m, step, baseline, cfg)
            m["verdict"] = verdict
            if best is None or G.better_checkpoint(m, best):
                R.save(policy, proc, out / "best_checkpoint")
                best = m
            R.write_json(out / "progress.json", report)
            if verdict["verdict"] == G.KILL:
                status, stop_reason = "killed", verdict
                break
    R.save(policy, proc, out / "last_checkpoint")
    final = report["evals"][-1]
    report.update(status=status, stop_reason=stop_reason, train_seconds=time.monotonic() - start,
                  cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                  best=best, best_checkpoint=str(out / "best_checkpoint") if best else None,
                  last_checkpoint=str(out / "last_checkpoint"),
                  final={"worst_cer": final["worst_cer"], "spk_sim": final["spk_sim"],
                         "judge_gap": final["judge_gap"], "xling_sim": final.get("xling_sim"),
                         "steps": len(report["steps"]),
                         "aborts": aborts + (1 if status == "tripwire_abort" else 0)})
    R.write_json(out / "report.json", report)
    print("STAGE2_ARM", a.arm, status, flush=True)
    return 0 if status == "complete" else 3


# ------------------------------------------------------------------------------------------------- compare

def compare(a):
    recipe = json.loads(Path(a.recipe).read_text())
    final = {}
    for path in a.reports:
        rep = json.loads(Path(path).read_text())
        if rep["arm"] in final:
            raise SystemExit("two reports for arm %s" % rep["arm"])
        final[rep["arm"]] = rep["final"]
    decisions = G.arm_decisions(final, recipe["stage2"])
    print(json.dumps({"final": final, "decisions": decisions}, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--recipe", default=str(HERE / "recipe.json"))
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("ceiling")
    c.add_argument("--bilingual", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=("a", "b", "c", "d"))
    r.add_argument("--stage1-report", required=True)
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
