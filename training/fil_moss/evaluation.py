"""Validation evaluation shared by stage 1 and stage 2 (GPU).

Teacher-forced loss on every validation row, then generation on a fixed subset scored by every loaded judge:
worst-judge CER (pooled per judge, then the max over judges), mean ECAPA similarity between the generated audio
and the target row's own real recording (same speaker, a different clip from the conditioning reference, so
copying the reference's channel or prosody does not score), and mean trimmed-duration ratio against that
recording. Test rows are never passed here.
"""
import hashlib

import numpy as np
import soundfile as sf
import torch

from . import judges as J
from . import runtime as R
from .scoring import corpus_cer, judge_gap, mean


def generation_subset(rows, n, seed):
    """Deterministic subset of rows that have a reference voice."""
    usable = [r for r in rows if r.get("ref_audio_codes") is not None]
    usable.sort(key=lambda r: hashlib.sha256(("%s|%s" % (seed, r["id"])).encode()).hexdigest())
    return usable[:n]


def teacher_forced_loss(model, ds, rows):
    model.eval()
    losses = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for r in rows:
            loss = model(**R.batch(ds, r)).loss
            if not torch.isfinite(loss):
                raise AssertionError("non-finite validation loss on %s" % r["id"])
            losses.append(float(loss))
    model.train()
    return mean(losses)


def generate_and_score(model, eng, rows, judges, speaker, gen_cfg, seeds, xling=None):
    """Generate every row in ``rows`` (each with its reference) for each seed and score it.

    ``xling``: optional list of {"text", "ref_codes", "target_centroid"} cross-lingual probes (EN reference,
    FIL text, centroid of the speaker's real FIL clips); reported as ``xling_sim``.
    """
    model.eval()
    pairs = {name: [] for name in judges}
    sims, ratios, truncated, n = [], [], 0, 0
    target_cache = {}
    for seed in seeds:
        for r in rows:
            R.seed_all(seed)
            cap = R.max_frames(r["seconds"], gen_cfg)
            (codes, trunc), = R.generate(eng, [r["conversations"][0]["text"]], [r["ref_audio_codes"]], cap, gen_cfg)
            truncated += trunc
            n += 1
            wav16 = J.resample(R.decode(eng.codec, codes), R.SAMPLE_RATE)
            for name, hyp in J.transcribe_all(judges, wav16).items():
                pairs[name].append((r["conversations"][0]["text"], hyp))
            if r["id"] not in target_cache:
                real, sr = sf.read(r["conversations"][0]["wav"], dtype="float32")
                real16 = J.resample(real, sr)
                target_cache[r["id"]] = (speaker.embed(real16), J.trimmed_seconds(real16, J.JUDGE_SR))
            emb, ref_secs = target_cache[r["id"]]
            sims.append(speaker.cosine(speaker.embed(wav16), emb) if len(wav16) >= J.JUDGE_SR // 4 else 0.0)
            ratios.append(J.trimmed_seconds(wav16, J.JUDGE_SR) / ref_secs if ref_secs > 0 else float("inf"))
    cers = {name: corpus_cer(p) for name, p in pairs.items()}
    out = {"cer": cers, "worst_cer": max(cers.values()), "judge_gap": judge_gap(cers), "judges": sorted(cers),
           "spk_sim": mean(sims), "duration_ratio": float(np.median(ratios)), "duration_ratio_mean": mean(ratios),
           "truncated_fraction": truncated / n, "utterances": n}
    if xling:
        xs = []
        for probe in xling:
            R.seed_all(seeds[0])
            (codes, _), = R.generate(eng, [probe["text"]], [probe["ref_codes"]], probe["frame_cap"], gen_cfg)
            wav16 = J.resample(R.decode(eng.codec, codes), R.SAMPLE_RATE)
            xs.append(speaker.cosine(speaker.embed(wav16), probe["target_centroid"]) if len(wav16) else 0.0)
        out["xling_sim"] = mean(xs)
    model.train()
    return out


def english_regression(model, eng, rows, judges, gen_cfg, seed):
    """Worst-judge CER on the English regression prompts ({"text", "ref_audio_codes", "seconds"} rows)."""
    model.eval()
    pairs = {name: [] for name in judges}
    for r in rows:
        R.seed_all(seed)
        (codes, _), = R.generate(eng, [r["text"]], [r["ref_audio_codes"]], R.max_frames(r["seconds"], gen_cfg),
                                 gen_cfg)
        wav16 = J.resample(R.decode(eng.codec, codes), R.SAMPLE_RATE)
        for name, hyp in J.transcribe_all(judges, wav16).items():
            pairs[name].append((r["text"], hyp))
    model.train()
    cers = {name: corpus_cer(p) for name, p in pairs.items()}
    return {"english_cer": cers, "english_worst_cer": max(cers.values())}
