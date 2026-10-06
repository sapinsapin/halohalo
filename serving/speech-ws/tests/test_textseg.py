"""Segmenter checks. Run: python tools/test_textseg.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from textseg import segment

def check(name, got, want):
    assert got == want, f"{name}\n  got  {got}\n  want {want}"
    print(f"ok  {name}")

check("plain two sentences",
      segment("Magandang umaga po. Kumusta ka?"),
      ["Magandang umaga po.", "Kumusta ka?"])

check("honorific is not a boundary",
      segment("Si Dr. Santos ang doktor namin."),
      ["Si Dr. Santos ang doktor namin."])

check("barangay abbreviation",
      segment("Nasa Brgy. Kalatagan po kami."),
      ["Nasa Brgy. Kalatagan po kami."])

check("initials",
      segment("Ang pangalan niya ay J. C. Diamante."),
      ["Ang pangalan niya ay J. C. Diamante."])

check("decimal number",
      segment("Ang timbang ay 3.5 kilo."),
      ["Ang timbang ay 3.5 kilo."])

check("whitespace collapse + empty",
      segment("  Oo   naman.  "), ["Oo naman."])

check("empty input", segment(""), [])
check("none input", segment(None), [])

# A long sentence with no clause boundary is left whole on purpose: cutting it
# mid-clause makes the unreliable stop token ramble (see _soft_wrap docstring).
no_comma = ("Ang halamang ito ay matatagpuan sa buong kapuluan ng Pilipinas "
            "at ginagamit na panggamot ng maraming pamilya sa probinsya.")
check("no clause boundary -> not split", segment(no_comma), [no_comma])

# With a comma, the opening segment is shortened to cut time-to-first-audio.
with_comma = ("Napansin niyang hugis puso ang mga dahon nito, at sinabi niya na ito "
              "ay ginagamit na panggamot ng maraming pamilya sa probinsya.")
segs = segment(with_comma)
assert len(segs) == 2, segs
assert segs[0] == "Napansin niyang hugis puso ang mga dahon nito,", segs[0]
assert len(segs[0]) <= 90, len(segs[0])
print(f"ok  split at the comma: first={len(segs[0])} chars, {len(segs)} segs")

# Every clause-split segment respects the cap, and no text may be lost.
para = " ".join([with_comma] * 3)
segs = segment(para)
import re
norm = lambda s: re.sub(r"\s+", "", s)
assert norm("".join(segs)) == norm(para), "text lost or duplicated"
assert all(len(s) <= 200 for s in segs), [len(s) for s in segs]
print(f"ok  paragraph: {len(segs)} segs, max {max(len(s) for s in segs)} chars, no text lost")

# A clause past HARD_SEG_CHARS is whitespace-split as damage control.
runon = " ".join(["salita"] * 60)          # 419 chars, no punctuation at all
segs = segment(runon)
assert len(segs) > 1 and all(len(s) <= 200 for s in segs), [len(s) for s in segs]
print(f"ok  run-on past hard cap split into {len(segs)} segs")

# One enormous token must not loop forever.
segs = segment("a" * 700)
assert segs and all(len(s) <= 200 for s in segs), [len(s) for s in segs]
print(f"ok  unsplittable token hard-cut into {len(segs)} segs")

print("\nall segmenter checks passed")
