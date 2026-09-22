"""Regenerate asr_registry.json — the list of ASR models the Transcribe tab
offers, with the numbers each model's card reports.

The dashboard tables are live from the Hub API, but the demo needs more than a
listing: which architecture to load a repo with, how big it is, and which
evaluation split its headline CER came from. That last one cannot be guessed —
the same corpus scores very differently on an overlapping split and on the
frozen speaker- and prompt-disjoint one, so the tab has to say which it is
showing. Run this after pushing new models:

  python make_asr_registry.py

Needs HF_TOKEN in the environment only for private repos; the public ones
resolve without it.
"""

import json
import os
import re
from datetime import date
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ORG = os.environ.get("DASHBOARD_ORG", "sapinsapin")
TOKEN = os.environ.get("HF_TOKEN")
OUT = Path(__file__).parent / "asr_registry.json"

# ONNX exports need onnxruntime, which this Space does not ship
SKIP = re.compile(r"-ONNX$")

# cards name the language as they please; the demo keys everything on the
# corpus's own codes, where Filipino is `fil`
LANG_ALIAS = {"tl": "fil", "tgl": "fil", "en": "eng"}

# What one model may weigh and still be offered in the Transcribe tab. A free
# CPU Space has 16 GB; the tab keeps two ~1 GB baselines resident plus one big
# slot, and the TTS and VC tabs hold ~1.5 GB more, so the big slot gets ~8 GB.
# whisper-large-v3 (5.8 GiB) fits and was tested there; the 7B CTC encoder
# (24.2 GiB) would take the Space down the moment someone picked it. Models
# over the budget stay in the registry, so the tab can say they exist.
MAX_WEIGHTS_GIB = 8.0
DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def weights_gib(safetensors):
    """Bytes the weights occupy as stored, from the Hub's per-dtype counts —
    what from_pretrained will hold in RAM, since nothing here downcasts."""
    params = getattr(safetensors, "parameters", None) or {}
    return sum(DTYPE_BYTES.get(dt, 4) * n for dt, n in params.items()) / 2**30


def card_of(repo):
    path = hf_hub_download(repo, "README.md", token=TOKEN)
    return Path(path).read_text(encoding="utf-8")


def frontmatter(card, key):
    m = re.search(rf"^{key}:\s*(.+)$", card.split("---")[1], re.M) \
        if card.startswith("---") else None
    return m.group(1).strip() if m else None


def metrics(card):
    """The `| cer | 0.1704 |` table every card written by push_bakeoff.py and
    compare_and_push_asr.py carries."""
    got = dict(re.findall(r"^\|\s*(cer|wer)\s*\|\s*([0-9.]+)\s*\|\s*$", card, re.M))
    return {k: float(v) for k, v in got.items()}


def split_of(card):
    """Which test split the card's numbers came from. 'frozen-disjoint' means
    no speaker and no prompt sentence is shared with training; 'in-domain'
    means the split overlaps, which flatters the model — by a factor of seven
    on Cebuano (see docs/pld_sota_track.md)."""
    return ("frozen-disjoint" if re.search(r"frozen-disjoint|prompt-disjoint", card)
            else "in-domain")


def units_of(name, kind):
    if kind != "ctc":
        return "bpe"
    return "syllable" if "syllable" in name else "char"


def main():
    api = HfApi(token=TOKEN)
    models = list(api.list_models(
        author=ORG, expand=["pipeline_tag", "tags", "safetensors", "private"]))
    rows = []
    for m in models:
        name = m.id.split("/", 1)[1]
        if m.pipeline_tag != "automatic-speech-recognition" or m.private:
            continue
        if SKIP.search(name):
            continue
        tags = m.tags or []
        card = card_of(m.id)
        lang = LANG_ALIAS.get(frontmatter(card, "language"),
                              frontmatter(card, "language"))
        if not lang:
            print(f"  skipped {name}: card states no language")
            continue
        kind = "ctc" if "ctc" in tags else "whisper"
        licence = next((t.split(":", 1)[1] for t in tags
                        if t.startswith("license:")), "unknown")
        base = next((t.split(":", 1)[1] for t in tags
                     if t.startswith("base_model:") and "finetune" not in t), "")
        gib = weights_gib(m.safetensors) if m.safetensors else None
        rows.append({
            "repo": m.id,
            "name": name,
            "language": lang,
            "kind": kind,
            "units": units_of(name, kind),
            "params": getattr(m.safetensors, "total", None) if m.safetensors else None,
            "weights_gib": round(gib, 2) if gib is not None else None,
            # an unknown size is not offered: hiding a model costs a menu entry,
            # offering one that is too big costs the whole Space
            "runnable": gib is not None and gib <= MAX_WEIGHTS_GIB,
            "licence": licence,
            "base_model": base,
            "split": split_of(card),
            # -norm models train and score with stress accents and punctuation
            # removed; their numbers are lower for that reason and the tab
            # has to say so beside them
            "normalised": name.endswith("-norm"),
            **metrics(card),
        })

    for r in rows:
        if not r["runnable"]:
            why = ("no size metadata" if r["weights_gib"] is None else
                   f"{r['weights_gib']:.1f} GiB > {MAX_WEIGHTS_GIB:.0f} GiB budget")
            print(f"  listed but not offered: {r['name']} ({why})")
    rows.sort(key=lambda r: (r["language"], r["params"] or 0, r["name"]))
    OUT.write_text(json.dumps(
        {"as_of": date.today().isoformat(), "models": rows}, indent=2) + "\n",
        encoding="utf-8")
    print(f"{OUT}: {len(rows)} models across "
          f"{len({r['language'] for r in rows})} languages")


if __name__ == "__main__":
    main()
