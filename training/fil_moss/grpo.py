"""Stage 2 bookkeeping: prompt filtering, per-arm rewards and tripwire windows. Standard library only.

The arms (recipe ``stage2.arms``):
  a  intelligibility from PLD-Whisper alone, speaker reward from the conditioning clip (Align2Speak-style)
  b  mean intelligibility over the judge families, conditioning-clip speaker reward
  c  worst-judge intelligibility, conditioning-clip speaker reward
  d  worst-judge intelligibility; on CLVI batches the speaker reward is 0.15 conditioning-clip similarity +
     0.15 similarity to the speaker's real Filipino recordings when conditioned on their English clip
"""
from . import scoring as S


def filter_prompts(lines, heldout_texts, min_chars=10, max_chars=200):
    """Unpaired RL prompts. Drops empty lines, lines with digits (no verbaliser has passed gate G3, so numeric
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


def speaker_for_arm(arm, cond_sim, clvi_sim, split):
    """Raw speaker value of one rollout. ``clvi_sim`` is set only on CLVI batches (arm d)."""
    if clvi_sim is None:
        return cond_sim
    if arm["speaker"] != "conditioning+clvi":
        raise ValueError("CLVI batch in an arm without a CLVI term")
    total = split["conditioning"] + split["clvi"]
    return (split["conditioning"] * cond_sim + split["clvi"] * clvi_sim) / total


def group_rewards(arm, cfg, rollouts):
    """Rank-based rewards and GRPO advantages for one prompt's rollouts.

    ``rollouts``: [{"cers": {judge: CER}, "cond_sim": float, "clvi_sim": float or None,
    "duration_ratio": float}]. Returns (rewards, advantages, components).
    """
    split = cfg["clvi"]["speaker_split"]
    components = {
        "intelligibility": [intelligibility_for_arm(arm, r["cers"]) for r in rollouts],
        "speaker": [speaker_for_arm(arm, r["cond_sim"], r.get("clvi_sim"), split) for r in rollouts],
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
