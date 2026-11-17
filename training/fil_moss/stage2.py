"""Stage 2, bounded baseline SFT of MOSS-TTS-Realtime on Filipino plus held-out evaluation (GPU, dwnd-01;
judges optionally on the RTX 3070).

    python -m training.fil_moss.guard --max-seconds 14400 -- python -m training.fil_moss.stage2 train \
        --codes /abs/run/codes --stage1-report /abs/run/stage1/report.json --out /abs/run/stage2
    python -m training.fil_moss.guard --max-seconds 1800 -- python -m training.fil_moss.stage2 latency \
        --checkpoint /abs/run/stage2/best_checkpoint --codes /abs/run/codes --out /abs/run/stage2-latency.json

Execution-sequence steps 5 and 6 of README.md, in the configuration the Hiligaynon pilot measured on the GB10:
start from the pinned pretrained model (not the Hiligaynon checkpoint); FP32 weights and gradients, BF16
autocast, bitsandbytes AdamW8bit, LR 1e-5, betas (0.9, 0.95), weight decay 0.1, 20 warmup steps, linear decay,
clip 1.0, ``language_model.embed_tokens`` frozen; recorded seed, order and schedule. The update budget is
``epochs`` passes over the frozen training split (not a fixed step count). No reference voice by default
(``conditioning: none``, as in the pilot); ``reference`` conditions on another clip of the same speaker and is
the separate multi-speaker experiment the pilot did not run.

Evaluation at step 0 (the pretrained model) and every ``eval_every_updates``: per-domain teacher-forced loss and
their equally weighted mean, which selects ``best_checkpoint`` (kept apart from ``last_checkpoint``), plus
generation metrics from every judge (``evaluation.py``) and WAVs for native listening. Test rows are never read.
The report compares the selected checkpoint with step 0 per domain and per judge; it does not call the voice good.

``latency`` measures streaming first-audio latency and real-time factor separately, through upstream's own
streaming session and frame decoder with the defaults of upstream ``app.py`` (offline generation time is not a
streaming benchmark).
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import get_scheduler

from . import evaluation as E
from . import gates as G
from . import judges as J
from . import runtime as R
from .stage1 import load_codes, sha

HERE = Path(__file__).resolve().parent


def train(a):
    recipe = json.loads(Path(a.recipe).read_text())
    t, gen, seed = recipe["stage2"], recipe["generation"], recipe["baseline"]["seed"]
    s1 = json.loads(Path(a.stage1_report).read_text())
    entry = G.stage2_entry(s1, a.codes)
    if entry["verdict"] != G.PASS:
        raise SystemExit("stage 2 refused: %s" % "; ".join(entry["reasons"]))
    out = R.new_output_dir(a.out)
    R.setup()
    R.seed_all(seed)
    proc = R.processor()
    cond = t["conditioning"]
    train_rows = [E.condition(r, cond) for r in load_codes(a.codes, "train")
                  if cond != "reference" or r.get("ref_audio_codes") is not None]
    val_rows = load_codes(a.codes, "validation")
    if {r["id"] for r in train_rows} & {r["id"] for r in val_rows}:
        raise SystemExit("train/validation id overlap")
    if cond == "reference":
        val_rows = [r for r in val_rows if r.get("ref_audio_codes") is not None]
    train_ds, val_ds = R.dataset(train_rows, proc), R.dataset(val_rows, proc)
    subset = E.generation_subset(val_rows, t["eval_generation_clips"], seed, cond)

    model = R.load_model(R.MODEL_DIR)
    params = R.freeze(model, t["frozen"])
    optimizer = R.adamw8bit(params, t["learning_rate"], t)
    bs = t["batch_size"]
    per_epoch = math.ceil(len(train_rows) / bs)
    total = per_epoch * t["epochs"]
    scheduler = get_scheduler(t["schedule"], optimizer, num_warmup_steps=t["warmup_steps"], num_training_steps=total)
    codec = R.load_codec()
    mirror = R.mirror(R.MODEL_DIR)
    eng = R.engine(mirror, codec)
    pool = J.make_pool(recipe)
    speaker = J.Speaker(recipe["judges"]["speaker"])
    cache = {}
    g = torch.Generator().manual_seed(seed)
    order = [i for _ in range(t["epochs"]) for i in torch.randperm(len(train_rows), generator=g).tolist()]
    batches = R.chunks(order, bs)
    report = {"recipe_sha256": sha(a.recipe), "codes": a.codes, "stage1_smoke": s1["smoke"], "conditioning": cond,
              "train_rows": len(train_rows), "validation_rows": len(val_rows), "batch_size": bs, "updates": total,
              "seed": seed, "training_order_sha256": hashlib.sha256(
                  json.dumps([train_rows[i]["id"] for i in order]).encode()).hexdigest(),
              "judges": pool.names, "judges_failed": pool.failed, "judge_events": pool.events,
              "evals": [], "steps": [], "best": None, "best_checkpoint": None, "status": "running"}

    def evaluate(update):
        started = time.monotonic()
        m = E.evaluate(model, mirror, eng, val_ds, val_rows, subset, pool, speaker, t, gen, seed, cache,
                       out / "samples" / ("update-%06d" % update))
        m.update(update=update, epoch_fraction=update / per_epoch, eval_seconds=time.monotonic() - started)
        report["evals"].append(m)
        print(json.dumps({"eval": m}), flush=True)
        return m

    zero = evaluate(0)
    report["zero_shot"] = zero
    start = time.monotonic()
    for update, idx in enumerate(batches, 1):
        t0 = time.monotonic()
        loss, norm = R.train_step(model, params, optimizer, scheduler,
                                  R.batch_many(train_ds, [train_rows[i] for i in idx]), t["grad_clip"])
        report["steps"].append({"update": update, "loss": loss, "grad_norm": norm,
                                "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - t0})
        if update % t["eval_every_updates"] == 0 or update == len(batches):
            m = evaluate(update)
            if G.better_checkpoint(m, report["best"]):
                R.save(model, proc, out / "best_checkpoint")
                report["best"], report["best_checkpoint"] = m, str(out / "best_checkpoint")
            report["judge_events"] = pool.events
            R.write_json(out / "progress.json", report)
    report["train_seconds"] = time.monotonic() - start
    report["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    R.save(model, proc, out / "last_checkpoint")
    report["last_checkpoint"] = str(out / "last_checkpoint")
    pretrained_best = report["best"] is None or not G.better_checkpoint(report["best"], zero)
    if pretrained_best:
        report["best"], report["best_checkpoint"] = zero, str(R.MODEL_DIR)
    report["summary"] = G.baseline_summary(zero, report["best"], pretrained_best)
    report["status"] = "complete"
    R.write_json(out / "report.json", report)
    print("STAGE2_BASELINE complete", json.dumps(report["summary"]), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------- latency

def latency(a):
    """First-audio latency and real-time factor through upstream's streaming path (``app.py`` run_stream)."""
    recipe = json.loads(Path(a.recipe).read_text())
    lc, gen, seed = recipe["latency"], recipe["generation"], recipe["baseline"]["seed"]
    R.setup()
    from transformers import AutoTokenizer
    from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor
    from mossttsrealtime.streaming_mossttsrealtime import (AudioStreamDecoder, MossTTSRealtimeInference,
                                                           MossTTSRealtimeStreamingSession)
    model = R.load_model(a.checkpoint, dtype=torch.bfloat16, train=False)
    tok = AutoTokenizer.from_pretrained(R.MODEL_DIR, trust_remote_code=True)
    proc = MossTTSRealtimeProcessor(tokenizer=tok)
    codec = R.load_codec()
    rows = E.generation_subset(load_codes(a.codes, "validation"), lc["sentences"], seed, "none")
    results = []
    for i, r in enumerate([rows[0]] + rows):  # first run warms kernels and is discarded
        R.seed_all(seed)
        inf = MossTTSRealtimeInference(model, tok, max_length=lc["max_length"])
        inf.reset_generation_state(keep_cache=False)
        session = MossTTSRealtimeStreamingSession(
            inf, proc, codec=codec, codec_sample_rate=R.SAMPLE_RATE,
            codec_encode_kwargs={"chunk_duration": lc["chunk_duration"]}, prefill_text_len=proc.delay_tokens_len,
            temperature=gen["temperature"], top_p=gen["top_p"], top_k=gen["top_k"], do_sample=True,
            repetition_penalty=gen["repetition_penalty"], repetition_window=gen["repetition_window"])
        session.clear_voice_prompt()
        system = proc.make_ensemble(None)
        user = tok("<|im_end|>\n<|im_start|>user\n" + lc["user_text"] + "<|im_end|>\n<|im_start|>assistant\n")["input_ids"]
        turn = np.full((len(user), proc.channels + 1), proc.audio_channel_pad, dtype=np.int64)
        turn[:, 0] = user
        session.reset_turn(input_ids=np.concatenate([system, turn], axis=0), include_system_prompt=True,
                           reset_cache=True)
        decoder = AudioStreamDecoder(codec, chunk_frames=lc["decode_chunk_frames"],
                                     overlap_frames=lc["decode_overlap_frames"], device=torch.device("cuda"))
        text_tokens = tok.encode(r["conversations"][0]["text"], add_special_tokens=False)
        first, samples = None, 0
        torch.cuda.synchronize()
        t0 = time.monotonic()

        def emit(frames):
            nonlocal first, samples
            for frame in frames:
                tokens = frame[0] if frame.dim() == 3 else frame
                stop = bool((tokens[:, 0] == 1026).any())
                tokens = tokens[(tokens[:, 0] != 1026) & ((tokens >= 0) & (tokens < 1024)).all(dim=1)]
                if tokens.numel():
                    decoder.push_tokens(tokens.detach())
                    for wav in decoder.audio_chunks():
                        if wav.numel():
                            if first is None:
                                torch.cuda.synchronize()
                                first = time.monotonic() - t0
                            samples += wav.numel()
                if stop:
                    return True
            return False
        with codec.streaming(batch_size=1):
            for k in range(0, len(text_tokens), lc["text_chunk_tokens"]):
                emit(session.push_text_tokens(text_tokens[k:k + lc["text_chunk_tokens"]]))
            emit(session.end_text())
            while True:
                more = session.drain(max_steps=1)
                if not more or emit(more) or session.inferencer.is_finished:
                    break
            final = decoder.flush()
            if final is not None:
                samples += final.numel()
        torch.cuda.synchronize()
        wall = time.monotonic() - t0
        if i:
            secs = samples / R.SAMPLE_RATE
            results.append({"id": r["id"], "first_audio_seconds": first, "wall_seconds": wall,
                            "audio_seconds": secs, "rtf": wall / secs if secs else None})
    firsts = [x["first_audio_seconds"] for x in results if x["first_audio_seconds"] is not None]
    rtfs = [x["rtf"] for x in results if x["rtf"]]
    out = {"checkpoint": a.checkpoint, "settings": lc, "runs": results,
           "first_audio_p50": float(np.median(firsts)) if firsts else None,
           "first_audio_p90": float(np.percentile(firsts, 90)) if firsts else None,
           "rtf_p50": float(np.median(rtfs)) if rtfs else None,
           "note": "batch 1, one stream, GB10 shared with resident services; full text available at start"}
    R.write_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--recipe", default=str(HERE / "recipe.json"))
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--codes", required=True)
    tr.add_argument("--stage1-report", required=True)
    tr.add_argument("--out", required=True)
    la = sub.add_parser("latency")
    la.add_argument("--checkpoint", required=True)
    la.add_argument("--codes", required=True)
    la.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return train(a) if a.cmd == "train" else latency(a)


if __name__ == "__main__":
    raise SystemExit(main())
