"""
Syllabification for Philippine orthographies — the TTS text frontend and the
syllable output units for CTC (plan items T5 / R2).

Why syllables. Measured on PLD read speech (docs/pld_models_plan.md §1.4):
Whisper, Llama-3 and Qwen3 BPE all cost 1.8-2.1 tokens per word on these
languages and all three split one stem five different ways across its
inflections (sulat -> sul|at, sum|ul|at, m|ags|us|ul|at ...). A rule
syllabifier costs the same sequence length (2.0-2.2 units/word) with a pooled
inventory of 4,414 types where the top 2,000 cover 99.5 % of tokens, and it
never splits a stem two ways. Philippine orthographies are near-phonemic and
syllable structure is (C)V(C), so the rules below are reliable in a way they
would not be for English.

Taglish. English words are NOT syllabified: English orthography is not
phonemic, so "download" would become dow-nlo-ad. They fall back to characters
by default, which keeps every unit inside a closed vocabulary — necessary for
a CTC head, where an unseen unit cannot be emitted at all.

Detection is deliberately conservative, because a false positive is worse
than a false negative: spelling out a frequent particle (Cebuano *man*,
Filipino *may*) corrupts far more tokens than leaving one English word
syllabified. Two rules keep that from happening — the word list applies only
from four characters up, and the cluster test runs after `ng` is folded to a
single symbol, so *mangga*, *tanggap* and *langgam* are not mistaken for
English consonant clusters.

What no text frontend can mark: stress and glottal stop, contrastive in
Tagalog and unwritten.

    from halolib.syllables import syllabify, units, fallback_rate

    syllabify("pinagsusulatan")   -> ['pi','nag','su','su','la','tan']
    units("Magandang umaga po")   -> ['ma','gan','dang','|','u','ma','ga','|','po']
    units("download mo")          -> ['d','o','w','n','l','o','a','d','|','mo']
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

__all__ = ["syllabify", "units", "inventory", "is_english_like",
           "fallback_rate", "VOWELS", "DIGRAPHS", "WORD_DELIM"]

# Vowels including the acute/grave-marked forms PLD prompts use for stress.
VOWELS = set("aeiou") | set("áéíóú") | set("àèìòù") | set("âêîôû") | set("äëïöü")

# Consonant digraphs that behave as one segment.
#   ng      phonemic velar nasal in every Philippine language here
#   ts      Spanish/English loans (tsokolate, tsinelas)
#   ly, ny  palatalised loans (kalye, ninyo)
# Deliberately short: over-listing digraphs misfires on native words more
# often than it helps.
DIGRAPHS = ("ng", "ts", "ly", "ny")

WORD_DELIM = "|"

# English spelling patterns that do not occur in Philippine orthography.
# Applied to a form where `ng` has been folded to one symbol, so that the
# three-consonant rule does not fire on mangga / tanggap / hinggil.
_ENGLISH_CLUSTERS = re.compile(
    r"(th|sh|ch|ck|qu|wh|gh|ph|tion|sion|ould|augh|eigh|ee|oa|ow$|ew|"
    r"[^aeiouŋ]y$|dge|mb$|kn|wr|[bcdfghjklmnpqrstvwxz]{3})")
_NG = re.compile(r"ng")
# Letters absent from native Philippine orthographies. One can appear in a
# proper noun (Cebu, Jose); two is a strong signal.
_NON_PH_LETTERS = re.compile(r"[cfjqvxz]")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Minimum length before the word list is consulted. Below it, short English
# words and Philippine particles collide (man, may, get, set, so, no), and the
# particles are far more frequent in these corpora.
_LEXICON_MIN_LEN = 4

# Frequent English words that survive the spelling heuristic because they are
# phonotactically legal in Filipino. Taglish leans on these, especially the
# technology vocabulary.
_ENGLISH_COMMON = frozenset("""
about also always area been being best better data does done down even ever
every first from give good great have help here info into just keep know last
like line link load long look made make many more most much must need never
next only open order other over page part people place play point post read
real right same save send show side since size small some soon still stop such
sure take team tell than that their them then there these they thing think this
those time today took very view want week well went were what when where which
while will with word work world would year your
account application battery browser button cancel click computer content
delete device download email error file files folder format image install
internet keyboard laptop login logout message mobile monitor mouse network
offline online password phone photo picture podcast printer profile program
screen search server settings share smartphone software submit subscribe
system tablet update upload username video website window wireless
""".split())


def _english_form(word: str) -> str:
    return _NG.sub("ŋ", word)


def is_english_like(word: str, lexicon: frozenset[str] | set[str] | None = None) -> bool:
    """Does this word look like English rather than a Philippine word or an
    assimilated loan? Assimilated loans are respelled phonemically (kotse,
    tsismis, kalye) and should syllabify normally."""
    w = word.lower()
    if len(w) >= _LEXICON_MIN_LEN:
        if w in _ENGLISH_COMMON:
            return True
        if lexicon and w in lexicon:
            return True
    form = _english_form(w)
    if _ENGLISH_CLUSTERS.search(form):
        return True
    return len(_NON_PH_LETTERS.findall(w)) >= 2


def _segment(word: str) -> list[str]:
    """Split into phonological segments, keeping digraphs whole."""
    segs, i, n = [], 0, len(word)
    while i < n:
        two = word[i:i + 2]
        if two in DIGRAPHS:
            segs.append(two)
            i += 2
        else:
            segs.append(word[i])
            i += 1
    return segs


def _is_vowel(seg: str) -> bool:
    base = unicodedata.normalize("NFD", seg)[:1]
    return seg in VOWELS or base in VOWELS


def syllabify(word: str) -> list[str]:
    """One Philippine word -> its syllables. (C)V(C): an onset takes at most
    one consonant segment, and a consonant closes the syllable only when the
    next segment is also a consonant (or the word ends) — which is what puts
    the boundary in mag-su-su-lat rather than ma-gsu-su-lat.

    Reconstruction is exact: ''.join(syllabify(w)) == w.lower()."""
    w = word.lower().strip()
    if not w:
        return []
    segs = _segment(w)
    if not any(_is_vowel(s) for s in segs):
        return [w]

    out: list[str] = []
    cur = ""
    j, n = 0, len(segs)
    while j < n:
        cur += segs[j]
        if _is_vowel(segs[j]):
            nxt_is_c = j + 1 < n and not _is_vowel(segs[j + 1])
            if nxt_is_c and (j + 2 >= n or not _is_vowel(segs[j + 2])):
                cur += segs[j + 1]          # close the syllable
                j += 1
            out.append(cur)
            cur = ""
        j += 1
    if cur:                                  # trailing consonants
        if out:
            out[-1] += cur
        else:
            out.append(cur)
    return out


def units(text: str, english: str = "chars", lexicon=None,
          delim: str | None = WORD_DELIM) -> list[str]:
    """A sentence -> a flat unit sequence with word delimiters.

    english: 'chars' (default) spells English-looking words out letter by
    letter, keeping the vocabulary closed; 'word' emits them whole, which
    needs an open vocabulary and suits a codec-LM more than a CTC head;
    'syllable' forces the Philippine rules onto them (for measurement only).
    """
    out: list[str] = []
    for i, word in enumerate(_WORD.findall(text)):
        if delim and i:
            out.append(delim)
        if english != "syllable" and is_english_like(word, lexicon):
            out.extend(list(word.lower()) if english == "chars" else [word.lower()])
        else:
            out.extend(syllabify(word))
    return out


def fallback_rate(texts, lexicon=None) -> dict:
    """Share of words taking the English fallback. The plan's gate: measure
    it, cap it, and report it beside any result that depends on the frontend."""
    words = eng = 0
    for t in texts:
        for w in _WORD.findall(t):
            words += 1
            eng += bool(is_english_like(w, lexicon))
    return {"words": words, "english": eng,
            "rate": (eng / words) if words else 0.0}


def inventory(texts, english: str = "chars", lexicon=None) -> Counter:
    """Frequency-ranked unit inventory over a corpus — the vocabulary source
    for the CTC head and the codec-LM tokenizer extension."""
    counts: Counter = Counter()
    for t in texts:
        counts.update(units(t, english=english, lexicon=lexicon, delim=None))
    return counts


if __name__ == "__main__":
    family = ["sulat", "sumulat", "sinulat", "magsusulat", "kasulatan",
              "pinagsusulatan", "nagsulat", "gisulat", "isulat", "manunulat",
              "ngayon", "kinabukasan", "mangga", "tsismis", "kalye", "unsa",
              "ámong", "bumaba", "porsiyento", "tanggap", "langgam", "hinggil"]
    for w in family:
        parts = syllabify(w)
        assert "".join(parts) == w.lower(), (w, parts)
        print(f"{w:16} {'-'.join(parts)}")

    # Philippine words that must NOT take the English fallback, including the
    # particles that collide with short English words and the ngg/nggg runs.
    for w in ["man", "may", "ang", "sa", "mangga", "tanggap", "langgam",
              "hinggil", "ngayon", "yung", "nga", "imong", "kalye", "tsismis"]:
        assert not is_english_like(w), f"false positive: {w}"
    # English that must take it.
    for w in ["download", "upload", "file", "website", "password", "the",
              "internet", "message"]:
        assert is_english_like(w), f"missed: {w}"
    print("\nfallback classification OK")

    for s in ["Magandang umaga po sa inyong lahat.",
              "Download mo yung file sa website.",
              "Unsa man ang imong ngalan?",
              "Tanggapin mo ang mangga."]:
        print(s)
        print("   ", " ".join(units(s)))
    print()
    print("fallback:", fallback_rate(["Download mo yung file", "Magandang umaga po"]))
