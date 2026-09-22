"""
External text for an ASR language model, with the leak check the plan makes
non-optional (docs/asr_decoder_plan.md, D1).

  python scripts/build_lm_text.py --language ceb
  -> $FINETUNE_DIR/ctc_lm_eval/ceb_lm_text.txt   (train transcripts + web text)

PLD's prompts are read sentences that may exist on the web, and the frozen test
split holds out prompts, not just speakers. An LM that has read a test sentence
turns WER into a memory test. So every external sentence is checked against
every test prompt: any sentence sharing a 5-gram of words with a test prompt is
dropped, and the count is reported. The train transcripts are clean by
construction and go in whole.

The web text is the org's own FineWeb-2 ingest for the language
(sapinsapin/halo-<lang>, plan item T1). Normalised the same way the acoustic
model's references are: lowercase, no stress accents, no punctuation.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
from halolib.finetune import normalise_text  # noqa: E402

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs"))
OUT = FINETUNE_DIR / "ctc_lm_eval"
N = 5


def ngrams(words, n=N):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--language", required=True)
    ap.add_argument("--max-sentences", type=int, default=2_000_000)
    args = ap.parse_args()
    from datasets import load_dataset

    meta = json.loads((OUT / f"{args.language}_meta.json").read_text())
    test_grams = set()
    for r in meta["refs"]:
        test_grams |= ngrams(normalise_text(r).split())
    train = (OUT / f"{args.language}_train_text.txt").read_text(encoding="utf-8").splitlines()

    # The ingests exist locally (T1, 2026-09-15) and are unpushed pending
    # speaker review; read the parquet directly rather than wait for that.
    local = FINETUNE_DIR / "halo_fw2" / args.language / "train.parquet"
    if local.exists():
        ds = load_dataset("parquet", data_files=str(local), split="train")
    else:
        ds = load_dataset(f"sapinsapin/halo-{args.language}", split="train",
                          token=os.environ.get("HF_TOKEN"))
    col = "text_cleaned" if "text_cleaned" in ds.column_names else "text"
    kept, dropped, seen, total = [], 0, set(), 0
    for doc in ds[col]:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", doc or ""):
            s = normalise_text(sent)
            w = s.split()
            if len(w) < 3 or s in seen:
                continue
            total += 1
            if ngrams(w) & test_grams:
                dropped += 1
                continue
            seen.add(s)
            kept.append(s)
            if len(kept) >= args.max_sentences:
                break
        if len(kept) >= args.max_sentences:
            break

    out = OUT / f"{args.language}_lm_text.txt"
    out.write_text("\n".join(normalise_text(t) for t in train) + "\n" + "\n".join(kept) + "\n",
                   encoding="utf-8")
    print(f"{args.language}: {len(train)} train sentences + {len(kept)} web sentences "
          f"({dropped} of {total} dropped for sharing a {N}-gram with a test prompt) -> {out}")


if __name__ == "__main__":
    main()
