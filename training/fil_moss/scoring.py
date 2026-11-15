"""Reward and metric arithmetic shared by stage 1 evaluation and stage 2 GRPO. Standard library only.

Every number a gate or a reward reads is computed here, so the CPU tests cover the arithmetic that the GPU
scripts rely on. Nothing here loads a model.
"""
import math
import re
import unicodedata


def normalise_text(text):
    """Same rule as ``halolib.finetune.normalise_text`` (lowercase, no stress accents or punctuation,
    apostrophes kept, single spaces). Copied because importing halolib pulls in ``datasets``; a test checks the
    two stay identical wherever halolib imports."""
    t = unicodedata.normalize("NFD", (text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = unicodedata.normalize("NFC", t).replace("’", "'").replace("‘", "'")
    return " ".join(re.sub(r"[^\w\s']", " ", t).split())


def edit_distance(a, b):
    """Levenshtein distance between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(reference, hypothesis):
    """Character error rate after normalisation. An empty reference is an error, not a zero."""
    ref, hyp = normalise_text(reference), normalise_text(hypothesis)
    if not ref:
        raise ValueError("empty reference transcript")
    return edit_distance(ref, hyp) / len(ref)


def corpus_cer(pairs):
    """Pooled CER over (reference, hypothesis) pairs: total edits over total reference characters."""
    edits = chars = 0
    for ref, hyp in pairs:
        r, h = normalise_text(ref), normalise_text(hyp)
        if not r:
            raise ValueError("empty reference transcript")
        edits += edit_distance(r, h)
        chars += len(r)
    if not chars:
        raise ValueError("no reference characters")
    return edits / chars


def intelligibility(cer_value, cap=1.0):
    """1 - CER, clipped to [0, 1]. Gibberish (CER >= cap) scores 0 under every judge."""
    return max(0.0, 1.0 - min(cer_value, cap) / cap)


def worst_judge(cers):
    """Worst-case-over-judges intelligibility from {judge: CER}. The reward of arms c and d."""
    if not cers:
        raise ValueError("no judges")
    return min(intelligibility(v) for v in cers.values())


def mean_judge(cers):
    """Mean intelligibility over judges. The reward of arm b."""
    if not cers:
        raise ValueError("no judges")
    return sum(intelligibility(v) for v in cers.values()) / len(cers)


def judge_gap(cers):
    """max - min CER across judges. Logged as a reward-hacking alarm, never used as a reward term:
    gibberish drives the gap to zero, so penalising it would reward gibberish."""
    return max(cers.values()) - min(cers.values())


def ranks(values):
    """Average ranks in [0, 1] (0 = worst, 1 = best, ties share the mean rank). Group of one -> 0.5."""
    n = len(values)
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2 / (n - 1)
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def group_advantages(rewards):
    """GRPO advantages: reward minus group mean, divided by group std. A group with no spread has no signal
    and gets all-zero advantages rather than a division by zero."""
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    if var < 1e-12:
        return [0.0] * n
    sd = math.sqrt(var)
    return [(r - mean) / sd for r in rewards]


def duration_ratio(generated_seconds, reference_seconds):
    if reference_seconds <= 0:
        raise ValueError("reference duration must be positive")
    return generated_seconds / reference_seconds


def duration_penalty(ratio):
    """|log duration ratio|: 0 at the reference pace, symmetric for too fast and too slow."""
    if ratio <= 0:
        return float("inf")
    return abs(math.log(ratio))


def silence_fraction(samples, sample_rate, frame_ms=20, top_db=40.0):
    """Fraction of frames whose RMS is more than ``top_db`` below the clip's loudest frame.
    All-zero audio is entirely silent."""
    frame = max(1, int(sample_rate * frame_ms / 1000))
    rms = []
    for start in range(0, len(samples), frame):
        chunk = samples[start:start + frame]
        rms.append(math.sqrt(sum(x * x for x in chunk) / len(chunk)))
    if not rms:
        return 1.0
    peak = max(rms)
    if peak == 0:
        return 1.0
    floor = peak * 10 ** (-top_db / 20)
    return sum(1 for r in rms if r < floor) / len(rms)


def token_entropy(tokens):
    """Shannon entropy (bits) of a token sequence's unigram distribution, e.g. codebook 0 of a rollout.
    A collapsing policy repeats a few codes and this falls."""
    if not tokens:
        return 0.0
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def combine_rewards(components, weights):
    """Rank-based reward per rollout in one group.

    ``components`` maps a name to one raw value per rollout (higher is better; pass a negated penalty).
    Each component is ranked within the group and the ranks are weighted, so no component needs a
    hand-set scale such as 1 - CER/30. Weights must be positive; components without a weight are refused
    so a typo cannot silently drop a term.
    """
    if set(components) != set(weights):
        raise ValueError("components and weights differ: %s vs %s" % (sorted(components), sorted(weights)))
    if any(w <= 0 for w in weights.values()):
        raise ValueError("weights must be positive")
    n = {len(v) for v in components.values()}
    if len(n) != 1:
        raise ValueError("every component needs one value per rollout")
    total = sum(weights.values())
    ranked = {k: ranks(v) for k, v in components.items()}
    size = n.pop()
    return [sum(weights[k] * ranked[k][i] for k in components) / total for i in range(size)]


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("mean of nothing")
    return sum(values) / len(values)
