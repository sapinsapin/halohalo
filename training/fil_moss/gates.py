"""Stage 1, 2 and 3 decisions as pure functions of recorded metrics. Standard library only.

The stages follow the MOSS notes in this repository: ``training/fil_moss/README.md`` (execution sequence) and the
Hiligaynon GB10 pilot (``hiligaynon_tts/moss/README.md``):

* stage 1, engineering smoke on the frozen Filipino data: loss and batch parity, one forward/backward/AdamW8bit
  step with FP32 gradients and BF16 compute verified, save/reload, and a matched-seed comparison sample;
* stage 2, bounded baseline SFT and held-out evaluation: checkpoint selection by the equally weighted mean of
  per-domain validation losses (the pilot's rule), best and last kept separately, no test data in selection;
* stage 3, GRPO on the stage 2 baseline: entry, CLVI descoping, tripwires, stop rules and arm adoption, each
  relative to the arm's own step-0 evaluation.

Training completion is never a quality verdict: native listening stays pending in every stage 2 summary.
Each function returns a dict with a ``verdict`` and its reasons; the scripts store it in their reports.
"""
import math

PASS, FAIL, INCOMPLETE, CONTINUE = "pass", "fail", "incomplete", "continue"


def _result(verdict, reasons, **extra):
    out = {"verdict": verdict, "reasons": list(reasons)}
    out.update(extra)
    return out


# --------------------------------------------------------------------------------------------- stage 1 checks

def parity_check(wrapper_loss, reference_loss, rtol):
    """Per-token log-probabilities gathered from ``local_logits`` must reproduce upstream's loss on the same batch.
    Batched evaluation and per-domain losses are computed from them."""
    if not (math.isfinite(wrapper_loss) and math.isfinite(reference_loss)):
        return _result(FAIL, ["non-finite loss"], wrapper=wrapper_loss, reference=reference_loss)
    rel = abs(wrapper_loss - reference_loss) / max(abs(reference_loss), 1e-12)
    ok = rel <= rtol
    return _result(PASS if ok else FAIL, [] if ok else ["relative difference %.3g > %.3g" % (rel, rtol)],
                   wrapper=wrapper_loss, reference=reference_loss, relative_difference=rel)


def batch_parity_check(single, batched, atol):
    """Per-utterance mean log-probabilities one at a time vs in one padded batch. Batched training and
    evaluation must credit each utterance the same (BF16 kernels differ by shape, hence a tolerance)."""
    if len(single) != len(batched) or not single:
        return _result(FAIL, ["length mismatch"])
    worst = max(abs(x - y) for x, y in zip(single, batched))
    ok = math.isfinite(worst) and worst <= atol
    return _result(PASS if ok else FAIL, [] if ok else ["max difference %.4g > %.4g" % (worst, atol)],
                   max_abs_difference=worst)


def one_step_check(m):
    """One optimizer step: finite loss and gradient norm, FP32 gradients, BF16 linear outputs under autocast
    (observed, not inferred from flags, as in the pilot), and the probed weights actually moved."""
    reasons = []
    if not (math.isfinite(m["loss"]) and math.isfinite(m["grad_norm"])):
        reasons.append("non-finite loss or gradient norm")
    if not m["grads_fp32"]:
        reasons.append("gradients are not FP32")
    if m["linear_output_dtype"] != "torch.bfloat16":
        reasons.append("linear output dtype %s, expected torch.bfloat16" % m["linear_output_dtype"])
    if m["probe_changed_elements"] <= 0:
        reasons.append("optimizer step did not change the probed parameters")
    return _result(FAIL if reasons else PASS, reasons)


def reload_check(weights_equal, loss_before, loss_after):
    reasons = []
    if not weights_equal:
        reasons.append("reloaded weights differ")
    if not abs(loss_after - loss_before) <= 1e-6 * max(1.0, abs(loss_before)):
        reasons.append("reloaded loss %.6f differs from %.6f" % (loss_after, loss_before))
    return _result(FAIL if reasons else PASS, reasons, loss_before=loss_before, loss_after=loss_after)


def sample_check(samples, max_silence_fraction=0.95):
    """Comparison samples: every one finite and not silent. Hitting the frame cap is reported, not failed
    (the pilot's Bible previews hit it); listening decides."""
    reasons = []
    if not samples:
        reasons.append("no samples generated")
    for s in samples:
        if not s["finite"]:
            reasons.append("%s: non-finite audio" % s["name"])
        elif s["silence_fraction"] > max_silence_fraction:
            reasons.append("%s: silent (%.2f)" % (s["name"], s["silence_fraction"]))
    return _result(FAIL if reasons else PASS, reasons, truncated=[s["name"] for s in samples if s["truncated"]])


def smoke_verdict(checks):
    """Stage 1 passes only if every named check passed."""
    failed = ["%s: %s" % (k, "; ".join(v["reasons"]) or v["verdict"]) for k, v in checks.items() if v["verdict"] != PASS]
    return _result(FAIL if failed else PASS, failed, checks=sorted(checks))


# ----------------------------------------------------------------------------------------------- stage 2

def balanced_validation_loss(domain_losses):
    """Equally weighted mean of per-domain validation losses: a large domain cannot hide a worsening small one
    (the pilot's Bible loss rose 21% while the PLD-dominated mean would have looked fine)."""
    if not domain_losses:
        raise ValueError("no validation domains")
    return sum(domain_losses.values()) / len(domain_losses)


def better_checkpoint(candidate, best):
    """Lower balanced validation loss wins. ``best`` may be None."""
    return best is None or candidate["balanced_validation_loss"] < best["balanced_validation_loss"]


def stage2_entry(stage1_report, codes):
    reasons = []
    if not stage1_report:
        return _result(FAIL, ["no stage 1 report"])
    if stage1_report.get("smoke", {}).get("verdict") != PASS:
        reasons.append("stage 1 smoke did not pass")
    if stage1_report.get("codes") != codes:
        reasons.append("stage 1 ran on different codes (%s)" % stage1_report.get("codes"))
    return _result(FAIL if reasons else PASS, reasons)


def baseline_summary(zero, best, best_is_pretrained):
    """What the baseline changed relative to the step-0 (pretrained) evaluation, per domain and per judge.
    Lower is better for every delta. Quality is not judged here."""
    def delta(a, b):
        return None if a is None or b is None else b - a
    out = {
        "balanced_validation_loss": delta(zero["balanced_validation_loss"], best["balanced_validation_loss"]),
        "domain_losses": {d: delta(zero["domain_losses"].get(d), v) for d, v in best["domain_losses"].items()},
        "cer": {j: delta(zero["cer"].get(j), v) for j, v in best["cer"].items()},
        "worst_cer": delta(zero["worst_cer"], best["worst_cer"]),
        "code_switched_worst_cer": delta(zero.get("code_switched_worst_cer"), best.get("code_switched_worst_cer")),
    }
    worse = [d for d, v in out["domain_losses"].items() if v is not None and v > 0]
    notes = []
    if best_is_pretrained:
        notes.append("no finetuned checkpoint beat the pretrained model on balanced validation loss")
    if worse:
        notes.append("validation loss rose in: %s" % ", ".join(sorted(worse)))
    return _result("complete", notes, deltas_vs_zero_shot=out, quality="native listening pending",
                   domains_worse=sorted(worse))


# ----------------------------------------------------------------------------------------- stage 3 (GRPO)
# Every limit below is relative to stage 3's own step-0 evaluation of the stage 2 checkpoint; none is borrowed
# from another model's numbers.

KILL = "kill"


def voice_measure(m):
    """The speaker number an evaluation reports for its conditioning: voice consistency for unconditioned runs,
    similarity to the target speaker's real recording for reference-conditioned runs."""
    return m["similarity_to_real_speaker"] if m.get("conditioning") == "reference" else m["voice_consistency"]


def stage3_entry(stage2_report, codes):
    """GRPO starts only from a completed stage 2 baseline whose selected checkpoint is a finetuned one."""
    reasons = []
    if not stage2_report:
        return _result(FAIL, ["no stage 2 report"])
    if stage2_report.get("status") != "complete":
        reasons.append("stage 2 status %s" % stage2_report.get("status"))
    if stage2_report.get("stage1_smoke", {}).get("verdict") != PASS:
        reasons.append("stage 1 smoke did not pass")
    if stage2_report.get("codes") != codes:
        reasons.append("stage 2 ran on different codes (%s)" % stage2_report.get("codes"))
    best = stage2_report.get("best_checkpoint") or ""
    if not best.endswith("best_checkpoint"):
        reasons.append("stage 2 kept the pretrained model (%s); there is no baseline to improve" % best)
    return _result(FAIL if reasons else PASS, reasons, checkpoint=best,
                   conditioning=stage2_report.get("conditioning"))


def clvi_mode(consented_bilingual_speakers, ecapa_ceiling, cfg):
    """Whether the cross-lingual voice-identity term may be a reward. Too few consented bilingual speakers ->
    evaluation metric only; ECAPA's genuine same-person EN->FIL similarity below the floor -> ECAPA is too
    language-biased to reward, arm d is dropped until renormalised per language pair."""
    c = cfg["clvi"]
    if consented_bilingual_speakers < c["min_consented_bilingual_speakers"]:
        return _result("eval_only", ["%d consented bilingual speakers < %d"
                                     % (consented_bilingual_speakers, c["min_consented_bilingual_speakers"])])
    if ecapa_ceiling is None:
        return _result("eval_only", ["ECAPA ceiling on genuine EN-FIL pairs not measured"])
    if ecapa_ceiling < c["ecapa_ceiling_min"]:
        return _result("dropped", ["ECAPA same-person cross-language similarity %.3f < %.2f"
                                   % (ecapa_ceiling, c["ecapa_ceiling_min"])])
    return _result("reward", [])


def tripwire(window_first, window_last, cfg):
    """Abort when the reward rises while silence rises or codec-token entropy falls (reward hacking signs)."""
    t = cfg["tripwires"]
    reasons = []
    if window_last["reward"] > window_first["reward"]:
        if window_last["silence_fraction"] - window_first["silence_fraction"] > t["silence_fraction_rise_max"]:
            reasons.append("silence fraction rose %.3f -> %.3f while reward rose"
                           % (window_first["silence_fraction"], window_last["silence_fraction"]))
        e0 = window_first["token_entropy"]
        if e0 > 0 and (e0 - window_last["token_entropy"]) / e0 > t["token_entropy_drop_max"]:
            reasons.append("codec token entropy fell %.3f -> %.3f while reward rose" % (e0, window_last["token_entropy"]))
    return _result(KILL if reasons else CONTINUE, reasons)


def stage3_evaluation_verdict(metrics, step, baseline, cfg):
    """Stop rules for one arm, relative to its step-0 evaluation: the voice must not drift, the policy must not
    leave the data distribution (balanced validation loss), and the worst-judge CER must improve by
    ``no_gain_step``."""
    k = cfg["kill"]
    v0, v = voice_measure(baseline), voice_measure(metrics)
    if v0 is not None and v is not None and v0 - v > k["voice_drop_max"]:
        return _result(KILL, ["voice measure fell %.3f -> %.3f (> %.2f)" % (v0, v, k["voice_drop_max"])])
    b0, b = baseline["balanced_validation_loss"], metrics["balanced_validation_loss"]
    if (b - b0) / b0 > k["validation_loss_rise_max"]:
        return _result(KILL, ["balanced validation loss rose %.4f -> %.4f (> %.0f%%)"
                              % (b0, b, 100 * k["validation_loss_rise_max"])])
    if step >= k["no_gain_step"] and metrics["worst_cer"] >= baseline["worst_cer"]:
        return _result(KILL, ["no worst-judge CER gain by step %d (%.4f vs %.4f at step 0)"
                              % (k["no_gain_step"], metrics["worst_cer"], baseline["worst_cer"])])
    return _result(CONTINUE, [])


def arm_decisions(final, cfg):
    """Pre-registered adoption rules across arms, from each arm's final validation metrics and its own step 0.

    ``final``: {"a"|"b"|"c"|"d": {"worst_cer", "voice", "judge_gap", "xling_sim" (optional), "aborts", "steps",
    "start_worst_cer", "start_voice"}}.
    """
    a = cfg["adoption"]
    out = {}
    for arm, m in final.items():
        if m.get("aborts", 0) >= cfg["kill"]["max_tripwire_aborts"]:
            out[arm] = _result(KILL, ["tripwire aborted %d times" % m["aborts"]])
        elif m["worst_cer"] >= m["start_worst_cer"]:
            out[arm] = _result(FAIL, ["worst-judge CER did not improve on its starting checkpoint"])
        else:
            out[arm] = _result(PASS, [], relative_gain=(m["start_worst_cer"] - m["worst_cer"]) / m["start_worst_cer"])
    b, c, d = final.get("b"), final.get("c"), final.get("d")
    if b and c:
        gain = (b["worst_cer"] - c["worst_cer"]) / b["worst_cer"] if b["worst_cer"] else 0.0
        voice_equal = abs(c["voice"] - b["voice"]) <= a["equal_voice_tolerance"]
        why = []
        if gain < a["c_over_b_min_relative_gain"]:
            why.append("worst-judge CER gain %.3f < %.2f" % (gain, a["c_over_b_min_relative_gain"]))
        if not voice_equal:
            why.append("voice measure differs by more than %.2f" % a["equal_voice_tolerance"])
        if c["judge_gap"] >= b["judge_gap"]:
            why.append("judge gap did not shrink")
        steps = min(b.get("steps", 0), c.get("steps", 0))
        verdict = PASS if not why else (KILL if steps >= cfg["kill"]["no_separation_step"] and gain <= 0 else FAIL)
        out["c_over_b"] = _result(verdict, why, relative_gain=gain)
    if c and d:
        why = []
        xs = d.get("xling_sim")
        if xs is None:
            why.append("cross-lingual similarity not measured")
        else:
            if xs < d.get("start_xling_sim", 0) + a["d_xling_sim_min_gain_over_start"]:
                why.append("cross-lingual similarity %.3f gained less than %.2f over its start"
                           % (xs, a["d_xling_sim_min_gain_over_start"]))
            if c.get("xling_sim") is not None and xs - c["xling_sim"] < a["d_xling_sim_min_gain_over_c"]:
                why.append("cross-lingual similarity gain over c %.3f < %.2f"
                           % (xs - c["xling_sim"], a["d_xling_sim_min_gain_over_c"]))
        if c["voice"] - d["voice"] > a["d_same_language_voice_max_drop"]:
            why.append("same-language voice measure dropped %.3f" % (c["voice"] - d["voice"]))
        if d["worst_cer"] > c["worst_cer"]:
            why.append("worst-judge CER regressed vs c")
        out["d_over_c"] = _result(FAIL if why else PASS, why)
    return out
