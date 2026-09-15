"""
Sentence segmentation for streaming synthesis.

SpeechT5 is not a streaming model — generate_speech returns a whole waveform in
one forward pass. So the only way to get audio moving before the full utterance
is synthesized is to cut the text up and synthesize piece by piece. Segment
length therefore sets both time-to-first-audio and the granularity at which a
client disconnect can stop the work.

Regex only, deliberately: the dependency set is pinned and small, and a full
sentence tokenizer buys very little on read-speech prompts.
"""

import re

from settings import FIRST_SEG_CHARS, HARD_SEG_CHARS, MAX_SEG_CHARS

# Periods after these are not sentence ends. Filipino honorifics and address
# abbreviations first, then the English ones that show up in Taglish text.
_ABBREV = frozenset("""
    dr dra gng bb sr sra mr mrs ms atty engr arch hon prof fr rev
    brgy blg st sto sta ave blvd rd cor
    no nos pp vs etc jr iii mgr sen rep gov mayor kgg
""".split())

# A period/!/? (possibly repeated, possibly followed by a closing quote or
# bracket) that is followed by whitespace or end of string.
_BOUNDARY = re.compile(r"[.!?…]+(?=[\"'’”)\]]*(?:\s|$))")

_WORD = re.compile(r"[\w'’-]+", re.UNICODE)


def _split_on(pattern: str, seg: str, limit: int) -> list[str]:
    """Greedily pack pieces of seg (split by pattern) into <= limit chunks."""
    parts, buf = [], ""
    for piece in re.split(pattern, seg):
        if not piece:
            continue
        cand = f"{buf} {piece}".strip() if buf else piece
        if buf and len(cand) > limit:
            parts.append(buf)
            buf = piece
        else:
            buf = cand
    if buf:
        parts.append(buf)
    return parts


def _soft_wrap(seg: str, limit: int) -> list[str]:
    """Split an over-long sentence, but only ever at a clause boundary.

    Cutting mid-clause is worse than a long segment. These checkpoints do not
    predict their stop token reliably, and a fragment that ends on a dangling
    conjunction is exactly where that shows: measured on speecht5_tts-pld-fil,
    a whitespace-split fragment ending "...ginagamit na" produced 7.6s of audio
    for 84 characters (11 chars/s) against 22 chars/s for the clause-aligned
    segment beside it — it finishes the phrase and then rambles.

    So an over-limit segment with no comma in it is returned over-limit. Only a
    clause longer than HARD_SEG_CHARS gets split on whitespace, as damage
    control against a pathological input rather than as normal behaviour.
    """
    if len(seg) <= limit:
        return [seg]
    parts = _split_on(r"(?<=[,;:—])\s+", seg, limit)
    out = []
    for part in parts:
        if len(part) <= HARD_SEG_CHARS:
            out.append(part)
            continue
        ws = _split_on(r"\s+", part, limit)
        if all(len(w) <= limit for w in ws) and len(ws) > 1:
            out.extend(ws)
        else:   # one enormous token: hard cut rather than loop
            out.extend(part[i:i + limit] for i in range(0, len(part), limit))
    return out or [seg]


def segment(text: str,
            first_limit: int = FIRST_SEG_CHARS,
            limit: int = MAX_SEG_CHARS) -> list[str]:
    """Split text into synthesis segments, shortest one first."""
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t:
        return []

    sentences, start = [], 0
    for m in _BOUNDARY.finditer(t):
        head = t[start:m.end()]
        # The token immediately before the punctuation decides whether this is
        # a real boundary: an abbreviation, an initial, or a digit means no.
        words = _WORD.findall(t[start:m.start()])
        last = words[-1].lower() if words else ""
        if last in _ABBREV or last.isdigit() or (len(last) == 1 and last.isalpha()):
            continue
        sentences.append(head.strip())
        start = m.end()
    tail = t[start:].strip()
    if tail:
        sentences.append(tail)

    segs = [s for sentence in sentences for s in _soft_wrap(sentence, limit)]
    # Shorten the opening segment to cut time-to-first-audio, but only if a
    # clause boundary makes that possible — never by cutting mid-clause.
    if segs and len(segs[0]) > first_limit:
        head = _split_on(r"(?<=[,;:—])\s+", segs[0], first_limit)
        if len(head) > 1 and len(head[0]) <= first_limit:
            segs[:1] = head
    return [s for s in segs if s.strip()]
