"""
Frozen speaker- AND prompt-disjoint evaluation splits for PLD.

Why this exists. PLD is a prompt corpus: many speakers read the same sentence
list, so a speaker-disjoint split still puts every test *sentence* in the
training set, and the model is scored on text it has already memorised. Every
CER on the dataset card today is measured that way and is an in-domain number.
Plan finding F11; see docs/pld_sota_track.md §6.

The construction. Hold out a fraction of speakers and, independently, a
fraction of prompts. Then

    test  = rows whose speaker IS held out AND whose prompt IS held out
    train = rows whose speaker is NOT held out AND whose prompt is NOT held out
    dropped = everything else

The dropped remainder is the price of doing this honestly; it is reported, not
hidden. Because test is an intersection, small speaker/prompt fractions give a
very small test set, so the defaults are deliberately larger than the usual
10 %.

The spec stores the held-out speaker ids and normalised prompt strings rather
than a row list, so it stays small, stays readable, and applies identically to
the local raw corpus and to the Hub parquet.

    python -m halolib.splits --languages bcl ceb fil ...      # build and freeze
    python -m halolib.splits --show ceb

    from halolib.splits import load_spec, assign
    spec = load_spec("ceb")
    assign(row, spec)      # -> "train" | "test" | None
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

__all__ = ["norm_prompt", "build_spec", "load_spec", "assign", "SPLIT_DIR"]

# Specs live beside the code, not beside the training runs. This used to be
# derived from FINETUNE_DIR, so any job that redirected its run directory (the
# ASR fleet, the Nebius bootstrap) silently lost the specs and fell back to the
# overlapping random split — seen on 2026-09-16. PLD_SPLIT_DIR overrides it.
# The directory is git-ignored: specs hold normalised PLD prompt text, and the
# corpus is CC-BY-NC under a research-only pledge, so it must not be pushed to
# a public repository.
SPLIT_DIR = Path(os.environ.get(
    "PLD_SPLIT_DIR", Path(__file__).resolve().parent.parent / "splits"))

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

SEED = 13
TEST_SPEAKER_FRAC = 0.20
TEST_PROMPT_FRAC = 0.25


def norm_prompt(text: str) -> str:
    """Prompt identity for disjointness: case, punctuation and the accent
    marks PLD uses inconsistently (únsa / unsa) must not make two readings of
    the same sentence look like different prompts."""
    t = unicodedata.normalize("NFD", (text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()


def build_spec(rows, language: str, seed: int = SEED,
               speaker_frac: float = TEST_SPEAKER_FRAC,
               prompt_frac: float = TEST_PROMPT_FRAC) -> dict:
    """rows: dicts with speaker_id and sentence (local index or Hub row)."""
    by_speaker = defaultdict(int)
    prompts = set()
    for r in rows:
        by_speaker[r["speaker_id"]] += 1
        prompts.add(norm_prompt(r["sentence"]))
    prompts.discard("")

    rng = random.Random(f"{seed}:{language}")
    speakers = sorted(by_speaker)
    prompt_list = sorted(prompts)
    rng.shuffle(speakers)
    rng.shuffle(prompt_list)

    n_spk = max(1, round(len(speakers) * speaker_frac))
    n_pr = max(1, round(len(prompt_list) * prompt_frac))
    test_speakers = set(speakers[:n_spk])
    test_prompts = set(prompt_list[:n_pr])

    spec = {
        "language": language,
        "seed": seed,
        "speaker_frac": speaker_frac,
        "prompt_frac": prompt_frac,
        "test_speakers": sorted(test_speakers),
        "test_prompts": sorted(test_prompts),
    }
    counts = {"train": 0, "test": 0, "dropped": 0}
    for r in rows:
        counts[assign(r, spec) or "dropped"] += 1
    spec["stats"] = {
        "rows": len(rows), "speakers": len(speakers), "prompts": len(prompt_list),
        "test_speakers": len(test_speakers), "test_prompts": len(test_prompts),
        **counts,
    }
    return spec


def load_spec(language: str, split_dir: Path | None = None) -> dict:
    p = (split_dir or SPLIT_DIR) / f"pld_{language}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — build it with: python -m halolib.splits "
            f"--languages {language}")
    spec = json.loads(p.read_text(encoding="utf-8"))
    spec["_speakers"] = set(spec["test_speakers"])
    spec["_prompts"] = set(spec["test_prompts"])
    return spec


def assign(row, spec: dict) -> str | None:
    """'train', 'test', or None for the disjointness remainder."""
    spk = spec.get("_speakers") or set(spec["test_speakers"])
    pr = spec.get("_prompts") or set(spec["test_prompts"])
    s_test = row["speaker_id"] in spk
    p_test = norm_prompt(row["sentence"]) in pr
    if s_test and p_test:
        return "test"
    if not s_test and not p_test:
        return "train"
    return None


def _load_rows(languages):
    """Read the raw corpus index; falls back to the Hub metadata if absent."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    root = Path(os.environ.get(
        "PLD_RAW", "/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD"))
    if root.exists() and os.environ.get("PLD_SOURCE") != "hub":
        from halolib.pld import index_corpus
        entries, _ = index_corpus(root)
        rows = [e for e in entries if not e.get("text_is_prompt")]
    else:
        from datasets import load_dataset
        ds = load_dataset("sapinsapin/pld", split="train",
                          token=os.environ.get("HF_TOKEN"))
        ds = ds.remove_columns([c for c in ds.column_names
                                if c not in ("speaker_id", "sentence", "language",
                                             "text_is_prompt", "speech_type")])
        rows = [r for r in ds if not r["text_is_prompt"]]
    by_lang = defaultdict(list)
    for r in rows:
        if languages is None or r["language"] in languages:
            by_lang[r["language"]].append(r)
    return by_lang


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--speaker-frac", type=float, default=TEST_SPEAKER_FRAC)
    ap.add_argument("--prompt-frac", type=float, default=TEST_PROMPT_FRAC)
    ap.add_argument("--show", default=None, help="print an existing spec and exit")
    args = ap.parse_args()

    if args.show:
        s = load_spec(args.show)
        print(json.dumps(s["stats"], indent=1))
        print("example held-out prompts:", s["test_prompts"][:3])
        return

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    by_lang = _load_rows(set(args.languages) if args.languages else None)
    print(f"{'lang':5} {'rows':>8} {'spk':>5} {'prompts':>8} "
          f"{'train':>8} {'test':>7} {'dropped':>8} {'test%':>6}")
    for lang in sorted(by_lang):
        spec = build_spec(by_lang[lang], lang, args.seed,
                          args.speaker_frac, args.prompt_frac)
        (SPLIT_DIR / f"pld_{lang}.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
        s = spec["stats"]
        print(f"{lang:5} {s['rows']:8d} {s['speakers']:5d} {s['prompts']:8d} "
              f"{s['train']:8d} {s['test']:7d} {s['dropped']:8d} "
              f"{s['test'] / max(s['rows'], 1) * 100:5.1f}%")
    print(f"\nwritten to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
