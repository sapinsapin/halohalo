"""Stage 1 and stage 2 decisions as pure functions of recorded metrics and ``recipe.json``. Standard library only.

The GPU scripts measure; these functions decide. Each returns a dict with a ``verdict`` and the reasons, and
the scripts write that dict into their report, so a reader can see why a run stopped or passed without
rerunning it. Held-out test metrics are never an input here: selection and stopping use validation only.
"""
import math

PASS, FAIL, CONTINUE, KILL, INCOMPLETE = "pass", "fail", "continue", "kill", "incomplete"


def _result(verdict, reasons, **extra):
    out = {"verdict": verdict, "reasons": list(reasons)}
    out.update(extra)
    return out


# ----------------------------------------------------------------------------------------- stage 1, S1 checks

def parity_check(wrapper_loss, reference_loss, rtol):
    """Our token-level loss, recomputed from ``local_logits``, must equal upstream ``model(...).loss`` on the
    same frozen batch. This is what proves the log-probabilities stage 2 optimises are the trained ones."""
    if not (math.isfinite(wrapper_loss) and math.isfinite(reference_loss)):
        return _result(FAIL, ["non-finite loss"], wrapper=wrapper_loss, reference=reference_loss)
    rel = abs(wrapper_loss - reference_loss) / max(abs(reference_loss), 1e-12)
    ok = rel <= rtol
    return _result(PASS if ok else FAIL, [] if ok else ["relative difference %.3g > %.3g" % (rel, rtol)],
                   wrapper=wrapper_loss, reference=reference_loss, relative_difference=rel)


def overfit_check(initial_loss, final_loss, duration_ratios, cfg):
    """32-clip overfit: loss must collapse and the overfit model must stop at the reference pace.
    ``duration_ratios`` are generated/reference seconds for the overfit clips' own texts."""
    reasons = []
    if not math.isfinite(final_loss):
        reasons.append("non-finite final loss")
    elif final_loss > cfg["overfit_max_final_loss"]:
        reasons.append("final train loss %.4f > %.4f" % (final_loss, cfg["overfit_max_final_loss"]))
    if not duration_ratios:
        reasons.append("no duration measurements")
    tol = cfg["overfit_duration_tolerance"]
    bad = [r for r in duration_ratios if abs(r - 1.0) > tol]
    if bad:
        reasons.append("%d of %d clips outside duration ratio 1 +/- %.2f" % (len(bad), len(duration_ratios), tol))
    return _result(FAIL if reasons else PASS, reasons, initial_loss=initial_loss, final_loss=final_loss,
                   duration_ratio_min=min(duration_ratios) if duration_ratios else None,
                   duration_ratio_max=max(duration_ratios) if duration_ratios else None)


def duration_alarm(mean_ratio, cfg):
    """Per-evaluation pacing alarm. Above the kill ratio the run stops (the Qwen3-TTS failure mode: speech that
    runs to the generation cap). Outside the success band it is a warning only."""
    lo, hi = cfg["success"]["duration_ratio_range"]
    if mean_ratio > cfg["kill"]["duration_ratio_max"]:
        return _result(KILL, ["mean duration ratio %.3f > kill %.2f" % (mean_ratio, cfg["kill"]["duration_ratio_max"])])
    if not lo <= mean_ratio <= hi:
        return _result(CONTINUE, ["duration ratio %.3f outside [%.2f, %.2f]" % (mean_ratio, lo, hi)], warning=True)
    return _result(CONTINUE, [], warning=False)


def verification_verdict(parity, overfit, reload_ok):
    """S1 passes only if all three checks pass. Nothing in the main SFT runs before this."""
    reasons = []
    if parity["verdict"] != PASS:
        reasons.append("loss parity: " + "; ".join(parity["reasons"]))
    if overfit["verdict"] != PASS:
        reasons.append("overfit: " + "; ".join(overfit["reasons"]))
    if not reload_ok:
        reasons.append("checkpoint save/reload changed weights or outputs")
    return _result(FAIL if reasons else PASS, reasons)


# --------------------------------------------------------------------------------------- stage 1, main SFT

def sft_evaluation_verdict(metrics, zero_shot, epoch_fraction, cfg):
    """Decide after one validation evaluation of the main SFT.

    ``metrics`` / ``zero_shot``: {"worst_cer", "cer": {judge: x}, "spk_sim", "duration_ratio",
    "english_worst_cer" (optional)}. ``zero_shot`` is the same evaluation at step 0 (pretrained model).
    ``epoch_fraction`` is updates done / updates per epoch.
    """
    alarm = duration_alarm(metrics["duration_ratio"], cfg)
    if alarm["verdict"] == KILL:
        return alarm
    epochs = cfg["epochs"]
    if epoch_fraction >= epochs and metrics["worst_cer"] > zero_shot["worst_cer"]:
        return _result(KILL, ["worst-judge CER %.4f after %d epochs is worse than zero-shot %.4f"
                              % (metrics["worst_cer"], epochs, zero_shot["worst_cer"])])
    return _result(CONTINUE, alarm["reasons"])


def better_checkpoint(candidate, best):
    """Checkpoint selection on validation only: lower worst-judge CER, ties broken by higher speaker
    similarity. ``best`` may be None."""
    if best is None:
        return True
    if candidate["worst_cer"] != best["worst_cer"]:
        return candidate["worst_cer"] < best["worst_cer"]
    return candidate["spk_sim"] > best["spk_sim"]


def sft_success(metrics, zero_shot, cfg):
    """Stage 1 success criteria on the selected checkpoint (validation). All must hold; a criterion whose
    measurement is missing makes the result incomplete, never a pass."""
    s = cfg["success"]
    reasons, missing = [], []
    mms = metrics.get("cer", {}).get(s["primary_judge"])
    if mms is None:
        missing.append("%s CER" % s["primary_judge"])
    elif mms > s["primary_judge_cer_max"]:
        reasons.append("%s CER %.4f > %.4f" % (s["primary_judge"], mms, s["primary_judge_cer_max"]))
    if metrics.get("spk_sim") is None:
        missing.append("speaker similarity")
    elif metrics["spk_sim"] < s["spk_sim_min"]:
        reasons.append("speaker similarity %.3f < %.2f" % (metrics["spk_sim"], s["spk_sim_min"]))
    lo, hi = s["duration_ratio_range"]
    if not lo <= metrics["duration_ratio"] <= hi:
        reasons.append("duration ratio %.3f outside [%.2f, %.2f]" % (metrics["duration_ratio"], lo, hi))
    eng, eng0 = metrics.get("english_worst_cer"), zero_shot.get("english_worst_cer")
    if eng is None or eng0 is None:
        missing.append("English regression set")
    elif eng > eng0 * (1 + s["english_cer_max_relative_regression"]):
        reasons.append("English worst-judge CER %.4f regressed more than %.0f%% from %.4f"
                       % (eng, 100 * s["english_cer_max_relative_regression"], eng0))
    if reasons:
        return _result(FAIL, reasons, missing=missing)
    if missing:
        return _result(INCOMPLETE, ["not measured: " + ", ".join(missing)], missing=missing)
    return _result(PASS, [])


# ------------------------------------------------------------------------------------------- stage 2 entry

def stage2_entry(stage1_report):
    """Stage 2 runs only from a stage 1 checkpoint that passed. The Qwen plan's kill clause also allows RL from
    the zero-shot base, but its dependency section cancels stage 2 when stage 1 fails; this follows the stricter
    rule until someone decides otherwise in recipe.json."""
    reasons = []
    if not stage1_report:
        return _result(FAIL, ["no stage 1 report"])
    if stage1_report.get("verification", {}).get("verdict") != PASS:
        reasons.append("stage 1 verification did not pass")
    if stage1_report.get("success", {}).get("verdict") != PASS:
        reasons.append("stage 1 success criteria: %s" % stage1_report.get("success", {}).get("verdict"))
    if not stage1_report.get("best_checkpoint"):
        reasons.append("no selected checkpoint")
    return _result(FAIL if reasons else PASS, reasons, checkpoint=stage1_report.get("best_checkpoint"))


def clvi_mode(consented_bilingual_speakers, ecapa_ceiling, cfg):
    """Whether the cross-lingual voice-identity term may be a reward.

    Descope rule: fewer consented bilingual speakers than the minimum -> evaluation metric only.
    ECAPA ceiling: genuine same-person EN->FIL similarity below the floor means ECAPA is too language-biased
    to reward; arm d is dropped until it is renormalised per language pair.
    """
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


# ----------------------------------------------------------------------------------------- stage 2 running

def tripwire(window_first, window_last, cfg):
    """Abort when reward rises while a tripwire degrades between the first and last window of a run.
    Windows: {"reward", "silence_fraction", "token_entropy"} means over consecutive steps."""
    t = cfg["tripwires"]
    reasons = []
    if window_last["reward"] > window_first["reward"]:
        if window_last["silence_fraction"] - window_first["silence_fraction"] > t["silence_fraction_rise_max"]:
            reasons.append("silence fraction rose %.3f -> %.3f while reward rose"
                           % (window_first["silence_fraction"], window_last["silence_fraction"]))
        e0 = window_first["token_entropy"]
        if e0 > 0 and (e0 - window_last["token_entropy"]) / e0 > t["token_entropy_drop_max"]:
            reasons.append("codec token entropy fell %.3f -> %.3f while reward rose"
                           % (e0, window_last["token_entropy"]))
    return _result(KILL if reasons else CONTINUE, reasons)


def stage2_evaluation_verdict(metrics, step, baseline, cfg):
    """Per-evaluation stop rules for one arm. ``baseline`` is the same evaluation at step 0."""
    k = cfg["kill"]
    if metrics["spk_sim"] < k["spk_sim_min"]:
        return _result(KILL, ["speaker similarity %.3f < %.2f" % (metrics["spk_sim"], k["spk_sim_min"])])
    if step >= k["no_gain_step"] and metrics["worst_cer"] >= baseline["worst_cer"]:
        return _result(KILL, ["no worst-judge CER gain by step %d (%.4f vs %.4f at step 0)"
                              % (k["no_gain_step"], metrics["worst_cer"], baseline["worst_cer"])])
    return _result(CONTINUE, [])


def arm_decisions(final, cfg):
    """Pre-registered adoption rules across arms, from each arm's final validation metrics.

    ``final``: {"a"|"b"|"c"|"d": {"worst_cer", "spk_sim", "judge_gap", "xling_sim" (d, optional),
    "aborts": int, "steps": int}}.
    """
    a = cfg["adoption"]
    out = {}
    for arm, m in final.items():
        if m.get("aborts", 0) >= cfg["kill"]["max_tripwire_aborts"]:
            out[arm] = _result(KILL, ["tripwire aborted %d times" % m["aborts"]])
    b, c, d = final.get("b"), final.get("c"), final.get("d")
    if b and c:
        gain = (b["worst_cer"] - c["worst_cer"]) / b["worst_cer"] if b["worst_cer"] else 0.0
        sim_equal = abs(c["spk_sim"] - b["spk_sim"]) <= a["equal_sim_tolerance"]
        if gain >= a["c_over_b_min_relative_gain"] and sim_equal and c["judge_gap"] < b["judge_gap"]:
            out["c_over_b"] = _result(PASS, [], relative_gain=gain)
        else:
            why = []
            if gain < a["c_over_b_min_relative_gain"]:
                why.append("worst-judge CER gain %.3f < %.2f" % (gain, a["c_over_b_min_relative_gain"]))
            if not sim_equal:
                why.append("speaker similarity differs by more than %.2f" % a["equal_sim_tolerance"])
            if c["judge_gap"] >= b["judge_gap"]:
                why.append("judge gap did not shrink")
            steps = min(b.get("steps", 0), c.get("steps", 0))
            verdict = KILL if steps >= cfg["kill"]["no_separation_step"] and gain <= 0 else FAIL
            out["c_over_b"] = _result(verdict, why, relative_gain=gain)
    if c and d:
        why = []
        xs = d.get("xling_sim")
        if xs is None:
            why.append("cross-lingual similarity not measured")
        else:
            if xs < a["d_xling_sim_min"]:
                why.append("cross-lingual similarity %.3f < %.2f" % (xs, a["d_xling_sim_min"]))
            if c.get("xling_sim") is not None and xs - c["xling_sim"] < a["d_xling_sim_min_gain_over_c"]:
                why.append("cross-lingual similarity gain over c %.3f < %.2f"
                           % (xs - c["xling_sim"], a["d_xling_sim_min_gain_over_c"]))
        if c["spk_sim"] - d["spk_sim"] > a["d_same_language_sim_max_drop"]:
            why.append("same-language similarity dropped %.3f" % (c["spk_sim"] - d["spk_sim"]))
        if d["worst_cer"] > c["worst_cer"]:
            why.append("worst-judge CER regressed")
        out["d"] = _result(FAIL if why else PASS, why)
    return out
