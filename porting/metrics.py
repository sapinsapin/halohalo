"""Scoring shared by every toolchain venv and the browser check's server:
standard library only, so it imports anywhere.

The normalisation is halolib.finetune.normalise_text, copied rather than
imported because the porting venvs do not carry halolib's dependencies. Keep
the two in step: lowercase, strip stress accents and punctuation, single
spaces, apostrophes kept. CER and WER are corpus-level (total edits over
total reference length), as jiwer computes them.
"""

import re
import unicodedata


def normalise_text(text: str) -> str:
    t = unicodedata.normalize("NFD", (text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = unicodedata.normalize("NFC", t).replace("’", "'").replace("‘", "'")
    return " ".join(re.sub(r"[^\w\s']", " ", t).split())


def edits(a, b) -> int:
    """Levenshtein distance between two sequences (strings or word lists)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(refs, hyps) -> float:
    return sum(edits(r, h) for r, h in zip(refs, hyps)) / max(1, sum(len(r) for r in refs))


def wer(refs, hyps) -> float:
    return (sum(edits(r.split(), h.split()) for r, h in zip(refs, hyps))
            / max(1, sum(len(r.split()) for r in refs)))


def score(refs, hyps) -> dict:
    """CER and WER on normalised text, the convention the published -norm
    models are scored under, plus how often the hypothesis is identical."""
    pairs = [(normalise_text(r), normalise_text(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return {"cer": None, "wer": None, "n": 0}
    r, h = map(list, zip(*pairs))
    return {"cer": cer(r, h), "wer": wer(r, h), "n": len(pairs),
            "exact": sum(a == b for a, b in pairs) / len(pairs)}


def agreement(hyps_a, hyps_b) -> dict:
    """How far a ported model's transcripts are from the reference model's —
    parity, separate from accuracy: a port can be as accurate and still
    different, or identical and wrong in the same places."""
    pairs = [(normalise_text(a), normalise_text(b)) for a, b in zip(hyps_a, hyps_b)]
    pairs = [(a, b) for a, b in pairs if a or b]
    if not pairs:
        return {"identical": None, "cer_vs_reference": None}
    a, b = map(list, zip(*pairs))
    return {"identical": sum(x == y for x, y in pairs) / len(pairs),
            "cer_vs_reference": cer([x or " " for x in a], [y or " " for y in b])}
