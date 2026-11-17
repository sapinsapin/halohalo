"""Stage 3 (GRPO) bookkeeping for MOSS: prompt filtering, per-arm rewards, tripwire windows. Standard library only.

Pre-registered arms (recipe ``stage3.arms``; "voice" is the speaker term below):
  a  intelligibility from PLD-Whisper alone (the in-family judge; the single-ASR-reward baseline)
  b  mean intelligibility over the judge families
  c  worst-judge intelligibility
  d  worst-judge intelligibility plus the cross-lingual voice-identity (CLVI) term on a fraction of groups:
     condition on a bilingual speaker's English clip, reward similarity to their real Filipino recordings

The speaker term depends on how stage 2 trained (``stage2.conditioning``):
  none       (the MOSS baseline)  "anchor": similarity to the centroid of the stage 2 checkpoint's own voice,
             measured once at stage 3 start, so RL cannot drift the voice while chasing intelligibility
  reference  "conditioning": similarity to the clip the rollout was conditioned on; arm d is allowed only here
"""
from . import scoring as S


def filter_prompts(lines, heldout_texts, min_chars=10, max_chars=200):
    """Unpaired RL prompts. Drops empty lines, lines with digits (no verbaliser has been validated, so numeric
    text is excluded rather than normalised), lines outside the length bounds, duplicates, and any line whose
    normalised text equals a validation or test transcript. Returns (kept, {reason: count})."""
    held = {S.normalise_text(t) for t in heldout_texts}
    kept, seen, dropped = [], set(), {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for line in lines:
        text = " ".join(line.split())
        norm = S.normalise_text(text)
        if not norm:
            drop("empty")
        elif any(c.isdigit() for c in text):
            drop("digits")
        elif not min_chars <= len(norm) <= max_chars:
            drop("length")
        elif norm in held:
            drop("heldout_transcript")
        elif norm in seen:
            drop("duplicate")
        else:
            seen.add(norm)
            kept.append(text)
    return kept, dropped


def speaker_mode(arm, stage2_conditioning):
    """Effective speaker term for an arm given how stage 2 trained."""
    if arm["speaker"] == "conditioning+clvi":
        if stage2_conditioning != "reference":
            raise ValueError("arm with CLVI needs a reference-conditioned stage 2 checkpoint")
        return "conditioning+clvi"
    return "conditioning" if stage2_conditioning == "reference" else "anchor"


def intelligibility_for_arm(arm, cers):
    rule = arm["intelligibility"]
    if rule.startswith("single:"):
        judge = rule.split(":", 1)[1]
        if judge not in cers:
            raise KeyError("arm needs judge %s, have %s" % (judge, sorted(cers)))
        return S.intelligibility(cers[judge])
    if rule == "mean":
        return S.mean_judge(cers)
    if rule == "min":
        return S.worst_judge(cers)
    raise ValueError("unknown intelligibility rule %s" % rule)


def speaker_value(mode, voice_sim, clvi_sim, split):
    """Raw speaker value of one rollout. ``voice_sim`` is similarity to the anchor or the conditioning clip;
    ``clvi_sim`` is set only on CLVI groups."""
    if clvi_sim is None:
        return voice_sim
    if mode != "conditioning+clvi":
        raise ValueError("CLVI group in an arm without a CLVI term")
    total = split["conditioning"] + split["clvi"]
    return (split["conditioning"] * voice_sim + split["clvi"] * clvi_sim) / total


def group_rewards(arm, mode, cfg, rollouts):
    """Rank-based rewards and GRPO advantages for one prompt's rollouts.

    ``rollouts``: [{"cers": {judge: CER}, "voice_sim": float, "clvi_sim": float or None,
    "duration_ratio": float}]. Returns (rewards, advantages, components).
    """
    split = cfg["clvi"]["speaker_split"]
    components = {
        "intelligibility": [intelligibility_for_arm(arm, r["cers"]) for r in rollouts],
        "speaker": [speaker_value(mode, r["voice_sim"], r.get("clvi_sim"), split) for r in rollouts],
        "duration": [-S.duration_penalty(r["duration_ratio"]) for r in rollouts],
    }
    rewards = S.combine_rewards(components, cfg["reward_weights"])
    return rewards, S.group_advantages(rewards), components


def speaking_rate(rows):
    """Normalised characters per second over training rows ({"text", "seconds"})."""
    chars = sum(len(S.normalise_text(r["text"])) for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    if chars == 0 or seconds <= 0:
        raise ValueError("cannot estimate speaking rate")
    return chars / seconds


def expected_seconds(text, chars_per_second):
    return max(0.5, len(S.normalise_text(text)) / chars_per_second)


def windows(history, size):
    """First and last ``size``-step means of {"reward", "silence_fraction", "token_entropy"}, or None until the
    run has 2 * size steps."""
    if len(history) < 2 * size:
        return None

    def avg(part):
        return {k: S.mean(h[k] for h in part) for k in ("reward", "silence_fraction", "token_entropy")}
    return avg(history[:size]), avg(history[-size:])
