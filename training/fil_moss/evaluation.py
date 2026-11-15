"""Held-out evaluation for the Filipino MOSS baseline (GPU). Validation rows only; test rows never reach here.

Two parts, as in the Hiligaynon pilot and the execution sequence in README.md:

* teacher-forced loss per validation domain and their equally weighted mean (the checkpoint-selection number);
* generation on a fixed subset, scored by every judge: pooled CER per judge, worst judge, judge gap, per-domain
  and code-switched vs monolingual worst-judge CER (code-switch handling), trimmed-duration ratio against the
  row's real recording, frame-cap hits, and the speaker measure that fits the conditioning:
  ``none`` (the baseline, as in the pilot): voice consistency, the mean pairwise ECAPA similarity across
  generated utterances; ``reference``: similarity to the target row's own real recording (a different clip from
  the conditioning reference, so copying the reference's channel does not score).

A few generations per evaluation are written as 24 kHz WAVs for native listening; they stay on the GB10 with the
run. Generation runs in batches on the BF16 mirror while the remote judges transcribe the previous batch.
"""
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import judges as J
from . import runtime as R
from .gates import balanced_validation_loss
from .scoring import corpus_cer, grouped_cer, judge_gap, mean, mean_pairwise, median, silence_fraction


def condition(row, conditioning):
    """The record the model sees: the reference voice is kept only for ``reference`` conditioning."""
    if conditioning == "reference":
        if row.get("ref_audio_codes") is None:
            raise ValueError("row %s has no reference" % row["id"])
        return row
    out = dict(row)
    out["ref_audio_codes"] = None
    out["ref_wav"] = None
    return out


def generation_subset(rows, n, seed, conditioning):
    """Deterministic subset; reference conditioning only uses rows that have a reference."""
    usable = [r for r in rows if conditioning != "reference" or r.get("ref_audio_codes") is not None]
    usable.sort(key=lambda r: hashlib.sha256(("%s|%s" % (seed, r["id"])).encode()).hexdigest())
    return usable[:n]


def domain_losses(model, ds, rows, conditioning, batch_size):
    """{domain: mean per-utterance teacher-forced loss} plus their equally weighted mean."""
    was_training = model.training
    model.eval()
    by_domain = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for part in R.chunks(rows, batch_size):
            logp, valid, _, sample = R.token_logprobs(model, R.batch_many(ds, [condition(r, conditioning) for r in part]))
            for j, r in enumerate(part):
                m = sample == j
                loss = -logp[m][valid[m]].mean()
                if not torch.isfinite(loss):
                    raise AssertionError("non-finite validation loss on %s" % r["id"])
                by_domain.setdefault(r.get("domain", "all"), []).append(float(loss))
    model.train(was_training)
    losses = {d: mean(v) for d, v in by_domain.items()}
    return {"domain_losses": losses, "balanced_validation_loss": balanced_validation_loss(losses),
            "validation_rows": len(rows)}


def _batches(eng, rows, conditioning, gen_cfg, gen_batch, seed):
    for start in range(0, len(rows), gen_batch):
        part = rows[start:start + gen_batch]
        R.seed_all(seed + start)
        refs = [r["ref_audio_codes"] if conditioning == "reference" else None for r in part]
        outs = R.generate(eng, [r["conversations"][0]["text"] for r in part], refs,
                          [R.max_frames(r["seconds"], gen_cfg) for r in part], gen_cfg)
        yield part, [(R.decode(eng.codec, codes), trunc) for codes, trunc in outs]


def generate_and_score(eng, rows, pool, speaker, gen_cfg, conditioning, seed, gen_batch=8, target_cache=None,
                       samples_dir=None, samples=0):
    cache = {} if target_cache is None else target_cache
    for r in rows:
        if r["id"] not in cache:
            real, sr = sf.read(r["conversations"][0]["wav"], dtype="float32")
            real16 = J.resample(real, sr)
            cache[r["id"]] = (speaker.embed(real16), J.trimmed_seconds(real16, J.JUDGE_SR))
    scored, embs, sims, ratios, sil, truncated = [], [], [], [], [], 0
    written = 0

    def finish(part, audio, handle):
        nonlocal truncated, written
        for r, (a24, trunc), hyps in zip(part, audio, pool.collect(handle)):
            wav16 = J.resample(a24, R.SAMPLE_RATE)
            for judge, hyp in hyps.items():
                scored.append({"judge": judge, "ref": r["conversations"][0]["text"], "hyp": hyp,
                               "domain": r.get("domain", "all"), "code_switched": bool(r.get("code_switched"))})
            real_emb, real_secs = cache[r["id"]]
            emb = speaker.embed(wav16) if len(wav16) >= J.JUDGE_SR // 4 else None
            if emb is not None:
                embs.append(emb)
                sims.append(speaker.cosine(emb, real_emb))
            ratios.append(J.trimmed_seconds(wav16, J.JUDGE_SR) / real_secs if real_secs > 0 else float("inf"))
            sil.append(silence_fraction(wav16.tolist(), J.JUDGE_SR))
            truncated += trunc
            if samples_dir is not None and written < samples:
                Path(samples_dir).mkdir(parents=True, exist_ok=True)
                sf.write(Path(samples_dir) / ("%02d_%s.wav" % (written, r["id"])), a24, R.SAMPLE_RATE)
                written += 1

    pending = None
    for part, audio in _batches(eng, rows, conditioning, gen_cfg, gen_batch, seed):
        handle = pool.submit([J.resample(a, R.SAMPLE_RATE) for a, _ in audio])
        if pending:
            finish(*pending)
        pending = (part, audio, handle)
    if pending:
        finish(*pending)

    judges = sorted({s["judge"] for s in scored})
    cers = {j: corpus_cer([(s["ref"], s["hyp"]) for s in scored if s["judge"] == j]) for j in judges}

    def worst_by(key):
        per = {j: grouped_cer([s for s in scored if s["judge"] == j], key) for j in judges}
        groups = {g for d in per.values() for g in d}
        return {g: max(per[j][g] for j in judges if g in per[j]) for g in groups}
    by_switch = worst_by(lambda s: "code_switched" if s["code_switched"] else "monolingual")
    out = {"cer": cers, "worst_cer": max(cers.values()), "judge_gap": judge_gap(cers), "judges": judges,
           "worst_cer_by_domain": worst_by(lambda s: s["domain"]),
           "code_switched_worst_cer": by_switch.get("code_switched"),
           "monolingual_worst_cer": by_switch.get("monolingual"),
           "duration_ratio": median(ratios), "truncated_fraction": truncated / len(rows),
           "silence_fraction_mean": mean(sil), "utterances": len(rows), "conditioning": conditioning,
           "similarity_to_real_speaker": mean(sims) if sims else None,
           "voice_consistency": mean_pairwise(embs[:64], speaker.cosine)}
    return out


def evaluate(model, mirror, eng, val_ds, val_rows, subset, pool, speaker, cfg, gen_cfg, seed, cache, samples_dir):
    """One full evaluation: sync the mirror, per-domain losses, then generation metrics."""
    R.sync(mirror, model)
    m = domain_losses(model, val_ds, val_rows, cfg["conditioning"], cfg["eval_batch_size"])
    m.update(generate_and_score(eng, subset, pool, speaker, gen_cfg, cfg["conditioning"], seed, cfg["gen_batch"],
                                cache, samples_dir, cfg["samples_per_eval"]))
    return m


def finite(audio):
    return bool(np.isfinite(audio).all())
