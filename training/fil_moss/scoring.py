"""Metric arithmetic for MOSS evaluation. Standard library only.

Every number a gate or report reads is computed here, so the CPU tests cover the arithmetic that the GPU scripts
rely on. Nothing here loads a model.
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


def judge_gap(cers):
    """max - min CER across judges: how much the verdict depends on which judge you ask."""
    return max(cers.values()) - min(cers.values())


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


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("mean of nothing")
    return sum(values) / len(values)


def median(values):
    v = sorted(values)
    if not v:
        raise ValueError("median of nothing")
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def mean_pairwise(vectors, dot):
    """Mean pairwise similarity of a set of unit vectors: the voice-consistency score of unconditioned
    generations (one voice across utterances scores high)."""
    n = len(vectors)
    if n < 2:
        return None
    total = sum(dot(vectors[i], vectors[j]) for i in range(n) for j in range(i + 1, n))
    return total / (n * (n - 1) / 2)


def grouped_cer(rows, key):
    """{group: pooled CER} for rows {"group": ..., "ref": ..., "hyp": ...}; groups from ``key(row)``."""
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append((r["ref"], r["hyp"]))
    return {g: corpus_cer(p) for g, p in groups.items()}


# ------------------------------------------------------------------------------------- stage 3 (GRPO) rewards

def intelligibility(cer_value, cap=1.0):
    """1 - CER, clipped to [0, 1]. Gibberish (CER >= cap) scores 0 under every judge."""
    return max(0.0, 1.0 - min(cer_value, cap) / cap)


def worst_judge(cers):
    """Worst-case-over-judges intelligibility from {judge: CER}: a rollout cannot score by fooling one judge."""
    if not cers:
        raise ValueError("no judges")
    return min(intelligibility(v) for v in cers.values())


def mean_judge(cers):
    if not cers:
        raise ValueError("no judges")
    return sum(intelligibility(v) for v in cers.values()) / len(cers)


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
    """GRPO advantages: reward minus group mean over group std. No spread -> all zero (no signal)."""
    n = len(rewards)
    m = sum(rewards) / n
    var = sum((r - m) ** 2 for r in rewards) / n
    if var < 1e-12:
        return [0.0] * n
    sd = math.sqrt(var)
    return [(r - m) / sd for r in rewards]


def duration_penalty(ratio):
    """|log duration ratio|: 0 at the expected pace, symmetric for too fast and too slow."""
    if ratio <= 0:
        return float("inf")
    return abs(math.log(ratio))


def token_entropy(tokens):
    """Shannon entropy (bits) of a token sequence's unigram distribution (codebook 0 of a rollout).
    A collapsing policy repeats a few codes and this falls."""
    if not tokens:
        return 0.0
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def combine_rewards(components, weights):
    """Rank-based reward per rollout in one group: each component (higher is better) is ranked within the
    group and the ranks are weighted, so no component needs a hand-set scale. Components and weights must
    match exactly so a typo cannot silently drop a term."""
    if set(components) != set(weights):
        raise ValueError("components and weights differ: %s vs %s" % (sorted(components), sorted(weights)))
    if any(w <= 0 for w in weights.values()):
        raise ValueError("weights must be positive")
    sizes = {len(v) for v in components.values()}
    if len(sizes) != 1:
        raise ValueError("every component needs one value per rollout")
    total = sum(weights.values())
    ranked = {k: ranks(v) for k, v in components.items()}
    size = sizes.pop()
    return [sum(weights[k] * ranked[k][i] for k in components) / total for i in range(size)]
