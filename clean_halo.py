"""
Clean the `text` column of a sapinsapin/halo-* dataset into `text_cleaned`.

Usage:
    python clean_halo.py sapinsapin/halo-hil
    python clean_halo.py sapinsapin/halo-tgl
    python clean_halo.py sapinsapin/halo-bcl
"""

import os
import sys
from multiprocessing import cpu_count
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import DatasetCard, login

from halolib import clean_text, is_usable
from halolib.fineweb import push_with_retry

load_dotenv(Path(__file__).parent / ".env")
login(token=os.environ["HF_TOKEN"])

if len(sys.argv) < 2:
    print("Usage: python clean_halo.py <hf_repo>")
    sys.exit(1)

REPO      = sys.argv[1]
LANG_CODE = REPO.split("-")[-1]
NUM_PROC  = max(1, cpu_count() - 1)

print(f"Repo     : {REPO}")
print(f"Language : {LANG_CODE}")
print(f"Workers  : {NUM_PROC}")

print("\nLoading dataset...")
ds = load_dataset(REPO)

# detect and normalize text column
TEXT_COL = next((c for c in ds["train"].column_names if c in ("text", "content")), None)
if TEXT_COL is None:
    print(f"ERROR: no 'text' or 'content' column. Columns: {ds['train'].column_names}")
    sys.exit(1)
if TEXT_COL != "text":
    print(f"Renaming '{TEXT_COL}' -> 'text'...")
    ds = ds.rename_column(TEXT_COL, "text")

def process(batch):
    batch["text_cleaned"] = [clean_text(t) if t else "" for t in batch["text"]]
    return batch

def is_usable_row(row):
    return is_usable(row["text_cleaned"])

print(f"Cleaning with {NUM_PROC} workers...")
ds = ds.map(process, batched=True, batch_size=1000, num_proc=NUM_PROC)
ds = ds.filter(is_usable_row, num_proc=NUM_PROC)
print(ds)

push_with_retry(ds, REPO, token=os.environ["HF_TOKEN"])

print("Updating dataset card...")
card = DatasetCard.load(REPO)
card.content = card.content.rstrip() + f"""

# {REPO.split("/")[-1]}

## Dataset Summary

`{REPO.split("/")[-1]}` is a web-scraped `{LANG_CODE}` text corpus for LLM pre-training,
compatible with [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
and [sapinsapin/BantayWika](https://huggingface.co/datasets/sapinsapin/BantayWika).

## Cleaning Pipeline

- Navigation menus, markdown tables, bare URLs, image markdown removed
- WordPress, Blogger, Scribd, SlideShare boilerplate stripped
- Cookie banners, social share buttons, site chrome removed
- Residual HTML tags and markdown link syntax cleaned (label text kept)
- Documents < 30 words or < 40% Latin characters filtered out

## Language

`{LANG_CODE}`
"""
card.push_to_hub(REPO, token=os.environ["HF_TOKEN"])
print("Done.")
