"""Measure the GB10 (and remote 3070 judge) components stages 1 and 2 spend time on, then estimate wall-clock.

    python -m training.fil_moss.guard --max-seconds 1800 -- python -m training.fil_moss.bench \
        --codes ~/work/hil-moss-pilot/data/pld_mix/pld_validation_codes.jsonl --out /abs/bench.json \
        [--train-hours 1 --validation-hours 0.1]

Uses already-encoded clips for timing only (the pilot's Hiligaynon PLD validation codes work: real lengths,
real text); writes numbers, never audio or transcripts. Trains a throwaway in-memory copy of the pretrained
model for a few steps and discards it. Reads: train step time at batch 1/4/8 with peak memory, no-grad forward
times, mirror sync, batched generation per frame at batch 1/8, codec decode, each judge locally, ECAPA, and the
remote judges if the server answers.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from . import judges as J
from . import runtime as R
from .timing import estimate

HERE = Path(__file__).resolve().parent


def timed(fn, repeat, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        fn()
        torch.cuda.synchronize()
        times.append(time.monotonic() - t0)
    return statistics.median(times)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--codes", required=True, help="a *_codes.jsonl with audio_codes (timing only)")
    p.add_argument("--out", required=True)
    p.add_argument("--recipe", default=str(HERE / "recipe.json"))
    p.add_argument("--train-hours", type=float, default=1.0)
    p.add_argument("--validation-hours", type=float, default=0.1)
    p.add_argument("--judges-only", action="store_true",
                   help="reuse the GB10 timings already saved in --out and measure only the judges (local and remote)")
    a = p.parse_args(argv)
    recipe = json.loads(Path(a.recipe).read_text())
    t1, gen = recipe["stage2"], recipe["generation"]
    R.setup()
    if a.judges_only:
        return judges_only(a, recipe)
    R.seed_all(0)
    rows = R.read_jsonl(Path(a.codes).expanduser())[:16]
    for r in rows:  # pilot rows carry no reference; condition on another row's codes like the Filipino export does
        r.setdefault("ref_audio_codes", None)
    for i, r in enumerate(rows):
        if r["ref_audio_codes"] is None:
            r["ref_audio_codes"] = rows[(i + 1) % len(rows)]["conversations"][0]["audio_codes"]
    clip_seconds = statistics.mean(len(r["conversations"][0]["audio_codes"]) / gen["frames_per_second"] for r in rows)
    t = {"frames_per_second": gen["frames_per_second"], "bench_mean_clip_seconds": clip_seconds}
    proc = R.processor()
    ds = R.dataset(rows, proc)

    t0 = time.monotonic()
    model = R.load_model(R.MODEL_DIR)
    t["load_seconds"] = time.monotonic() - t0
    params = R.freeze(model, t1["frozen"])
    opt = R.adamw8bit(params, 1e-7, t1)  # tiny LR: the copy is discarded anyway
    t["train_step_seconds"], t["train_peak_gib"] = {}, {}
    for b in (1, 4, 8):
        torch.cuda.reset_peak_memory_stats()
        inputs = R.batch_many(ds, rows[:b])
        t["train_step_seconds"][str(b)] = timed(lambda: R.train_step(model, params, opt, None, inputs, 1.0), 4)
        t["train_peak_gib"][str(b)] = torch.cuda.max_memory_allocated() / 2 ** 30
        print(json.dumps({"train_batch": b, "seconds": t["train_step_seconds"][str(b)],
                          "peak_gib": t["train_peak_gib"][str(b)]}), flush=True)
    R.write_json(a.out, {"partial": True, "timings": t})

    def fwd(m, b):
        inputs = R.batch_many(ds, rows[:b])

        def go():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                R.token_logprobs(m, inputs)
        return go
    t["forward_seconds"] = {str(b): timed(fwd(model, b), 3) for b in (1, 4, 8)}

    ref = R.load_model(R.MODEL_DIR, dtype=torch.bfloat16, train=False)
    t["ref_forward_seconds"] = {str(b): timed(fwd(ref, b), 3) for b in (1, 4, 8)}
    mirror = R.mirror(R.MODEL_DIR)
    t["sync_seconds"] = timed(lambda: R.sync(mirror, model), 3)
    t0 = time.monotonic()
    R.save(model, proc, Path(a.out).with_suffix(".tmp-ckpt"))
    t["save_seconds"] = time.monotonic() - t0
    t0 = time.monotonic()
    R.load_model(Path(a.out).with_suffix(".tmp-ckpt"), train=False).cpu()
    t["save_reload_seconds"] = t["save_seconds"] + time.monotonic() - t0
    import shutil
    shutil.rmtree(Path(a.out).with_suffix(".tmp-ckpt"))
    del opt
    torch.cuda.empty_cache()

    codec = R.load_codec()
    eng = R.engine(mirror, codec)
    text = rows[0]["conversations"][0]["text"]
    refc = rows[0]["ref_audio_codes"]
    t["gen_seconds_per_frame"] = {}
    for b in (1, 8):
        res = {}

        def go(cap):
            R.seed_all(0)
            outs = R.generate(eng, [text] * b, [refc] * b, cap, gen)
            res[cap] = max(len(c) for c, _ in outs)
        short, long_ = timed(lambda: go(10), 2), timed(lambda: go(80), 2)
        t["gen_seconds_per_frame"][str(b)] = (long_ - short) / max(1, res[80] - res[10])
        t["gen_prefill_seconds"] = max(0.0, short - 10 * t["gen_seconds_per_frame"][str(b)])
        print(json.dumps({"gen_batch": b, "seconds_per_frame": t["gen_seconds_per_frame"][str(b)]}), flush=True)
    t["peak_all_resident_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    R.write_json(a.out, {"partial": True, "timings": t})
    print(json.dumps({k: t[k] for k in ("forward_seconds", "ref_forward_seconds", "sync_seconds", "save_seconds",
                                        "save_reload_seconds", "peak_all_resident_gib")}), flush=True)

    sample = R.generate(eng, [text], [refc], 75, gen)[0][0]
    audio24 = R.decode(codec, sample)
    secs = max(len(audio24) / R.SAMPLE_RATE, 1e-3)
    t["decode_seconds_per_audio_second"] = timed(lambda: R.decode(codec, sample), 3) / secs
    wav16 = J.resample(audio24, R.SAMPLE_RATE)
    t["trim_seconds"] = timed(lambda: J.trimmed_seconds(wav16, J.JUDGE_SR), 3, warmup=1)
    try:
        speaker = J.Speaker(recipe["judges"]["speaker"])
        t["ecapa_seconds"] = timed(lambda: speaker.embed(wav16), 5)
    except Exception as exc:
        t["ecapa_seconds"], t["ecapa_error"] = 0.05, str(exc)
    R.write_json(a.out, {"partial": True, "timings": t})
    return measure_judges(a, recipe, t, wav16)


def judges_only(a, recipe):
    """Resume from saved timings; time the judges on upstream's own demo clip (public, ships with the code)."""
    import soundfile as sf
    t = json.loads(Path(a.out).read_text())["timings"]
    wav, sr = sf.read(R.ROOT / "upstream/moss_tts_realtime/audio/user1.wav", dtype="float32")
    wav16 = J.resample(wav if wav.ndim == 1 else wav.mean(axis=1), sr)
    t["judge_clip_seconds"] = len(wav16) / J.JUDGE_SR
    return measure_judges(a, recipe, t, wav16)


def measure_judges(a, recipe, t, wav16):
    local, failed = J.load(recipe["judges"]["fil"])
    t["judge_local_seconds"] = {n: timed(lambda f=f: f(wav16), 3) for n, f in local.items()}
    if "whisper" in local:
        t["judge_local_seconds_whisper_batch8_per_utt"] = timed(lambda: local["whisper"].batch([wav16] * 8), 2) / 8
    t["judge_local_failed"] = failed
    print(json.dumps({"judge_local_seconds": t["judge_local_seconds"], "failed": failed}), flush=True)
    del local
    torch.cuda.empty_cache()
    pool = J.make_pool(recipe)
    t["judge_events"] = pool.events
    if pool.remote_names:  # remote judges alone, one warm-up, then a group of 8 as stage 3 sends it
        batch = [wav16] * 8
        pool._remote(batch[:1], pool.remote_names)
        t0 = time.monotonic()
        pool._remote(batch, pool.remote_names)
        t["judge_remote_seconds_per_utt"] = (time.monotonic() - t0) / len(batch)
    else:
        t["judge_remote_seconds_per_utt"] = None
    t["host_mem_available_gib"] = R_mem()
    data = {"train_hours": a.train_hours, "validation_hours": a.validation_hours, "mean_clip_seconds": t["bench_mean_clip_seconds"]}
    remote = bool(t["judge_remote_seconds_per_utt"])
    result = {"timings": t, "estimate": {"batch_%d" % b: estimate(t, data, recipe, remote=remote, batch_size=b)
                                         for b in (1, 4, 8)}}
    R.write_json(a.out, result)
    print(json.dumps(result, indent=1), flush=True)


def R_mem():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return None


if __name__ == "__main__":
    main()
