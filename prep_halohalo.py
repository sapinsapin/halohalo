"""
Convert a cleaned sapinsapin/halo-* dataset into a FineWeb-compatible dataset
and push it to a target Hugging Face repo. Supports appending with deduplication.

Usage:
    python prep_halohalo.py <source_repo> <target_repo> [--append]

Examples:
    python prep_halohalo.py sapinsapin/halo-hil sapinsapin/fineweb-hil
    python prep_halohalo.py sapinsapin/halo-hil sapinsapin/halohalo --append
    python prep_halohalo.py sapinsapin/halo-tgl sapinsapin/halohalo --append
    python prep_halohalo.py sapinsapin/halo-bcl sapinsapin/halohalo --append
"""

import os
import sys
from multiprocessing import cpu_count
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import DatasetCard, login

from halolib.fineweb import add_fineweb_columns, append_to, push_with_retry

load_dotenv(Path(__file__).parent / ".env")
login(token=os.environ["HF_TOKEN"])

if len(sys.argv) < 3:
    print("Usage: python prep_halohalo.py <source_repo> <target_repo> [--append]")
    sys.exit(1)

SOURCE_REPO = sys.argv[1]
TARGET_REPO = sys.argv[2]
APPEND      = "--append" in sys.argv
LANG_CODE   = SOURCE_REPO.split("-")[-1]
SOURCE_NAME = SOURCE_REPO.split("/")[-1]
NUM_PROC    = max(1, cpu_count() - 1)

print(f"Source   : {SOURCE_REPO}")
print(f"Target   : {TARGET_REPO}")
print(f"Language : {LANG_CODE}")
print(f"Append   : {APPEND}")
print(f"Workers  : {NUM_PROC}")

print("\nLoading source dataset...")
ds = load_dataset(SOURCE_REPO)
print(ds)

# use text_cleaned as the canonical text, drop raw text
print("Replacing raw text with text_cleaned...")
ds = ds.map(lambda x: {"text": x["text_cleaned"]}, num_proc=NUM_PROC)
ds = ds.remove_columns(["text_cleaned"])

print(f"Adding FineWeb columns with {NUM_PROC} workers...")
ds = add_fineweb_columns(ds, source=SOURCE_NAME, language=LANG_CODE, num_proc=NUM_PROC)

if APPEND:
    print(f"\nDeduplicating and appending to {TARGET_REPO}...")
    ds = append_to(ds, TARGET_REPO, num_proc=NUM_PROC)

print(ds)

# ---------------------------------------------------------------------------
# Compute statistics from the pushed split (train)
# ---------------------------------------------------------------------------

def compute_stats(dataset) -> dict:
    token_counts = dataset["token_count"]
    languages    = dataset["language"]
    total_docs   = len(token_counts)
    total_tokens = sum(token_counts)
    avg_tokens   = total_tokens / total_docs if total_docs else 0
    min_tokens   = min(token_counts) if token_counts else 0
    max_tokens   = max(token_counts) if token_counts else 0
    sources      = Counter(dataset["source"])
    lang_counts  = Counter(languages)
    lang_tokens  = {}
    for tc, lang in zip(token_counts, languages):
        lang_tokens[lang] = lang_tokens.get(lang, 0) + tc
    return {
        "total_docs":   total_docs,
        "total_tokens": total_tokens,
        "avg_tokens":   round(avg_tokens, 1),
        "min_tokens":   min_tokens,
        "max_tokens":   max_tokens,
        "sources":      sources,
        "languages":    lang_counts,
        "lang_tokens":  lang_tokens,
    }

print("\nComputing statistics...")
stats = compute_stats(ds["train"])
print(f"  Documents : {stats['total_docs']:,}")
print(f"  Tokens    : {stats['total_tokens']:,}")
print(f"  Avg tokens: {stats['avg_tokens']}")

# build sources table rows
source_rows = "\n".join(
    f"| `{src}` | {count:,} |"
    for src, count in sorted(stats["sources"].items(), key=lambda x: -x[1])
)

# build language table rows
lang_rows = "\n".join(
    f"| `{lang}` | {count:,} | {stats['lang_tokens'].get(lang, 0):,} |"
    for lang, count in sorted(stats["languages"].items(), key=lambda x: -x[1])
)
lang_rows += f"\n| **Total** | **{stats['total_docs']:,}** | **{stats['total_tokens']:,}** |"

print("\nPushing to hub...")
push_with_retry(ds, TARGET_REPO, token=os.environ["HF_TOKEN"])

# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

print("\nUpdating dataset card...")
try:
    card = DatasetCard.load(TARGET_REPO)
    # extract just the YAML frontmatter block
    content = card.content
    if content.startswith("---"):
        end = content.index("---", 3) + 3
        frontmatter = content[:end]
    else:
        frontmatter = (
            f"---\nlanguage:\n- {LANG_CODE}\n"
            f"task_categories:\n- text-generation\n"
            f"tags:\n- fineweb\n- philippine-languages\n---"
        )
except Exception:
    frontmatter = (
        f"---\nlanguage:\n- {LANG_CODE}\n"
        f"task_categories:\n- text-generation\n"
        f"tags:\n- fineweb\n- philippine-languages\n---"
    )

card = DatasetCard(frontmatter + f"""

# {TARGET_REPO.split("/")[-1]}

## Dataset Summary

`{TARGET_REPO.split("/")[-1]}` is a Pretraining text corpus for Philippine languages,
assembled from web-scraped data. It is compatible with [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) for LLM Pretraining.

## Source Data

Derived from the following cleaned datasets:

| Source | Documents |
|---|---|
{source_rows}


Each source dataset was cleaned using `clean_halo.py` to remove web boilerplate, navigation menus,
markdown noise, HTML artifacts, and low-quality documents before being included here.

## Processing

1. **Cleaning** (`clean_halo.py`) — strips boilerplate, HTML, markdown noise; filters documents
   with fewer than 30 words or less than 40% Latin characters
2. **FineWeb formatting** (`prep_halohalo.py`) — adds `source`, `language`, `token_count`,
   `content_hash`; deduplicates against existing documents using MD5 content hashing

Processing code is available at [github.com/sapinsapin/halohalo](https://github.com/sapinsapin/halohalo).

## Statistics

| Metric | Value |
|---|---|
| Total documents | {stats['total_docs']:,} |
| Total tokens | {stats['total_tokens']:,} |
| Avg tokens per document | {stats['avg_tokens']:,} |
| Min tokens | {stats['min_tokens']:,} |
| Max tokens | {stats['max_tokens']:,} |

### Languages

| Language | Documents | Word Count |
|---|---|---|
{lang_rows}

## Schema

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Cleaned document text |
| `id` | `str` | Unique document identifier |
| `source` | `str` | Source dataset name |
| `language` | `str` | ISO 639-3 language code |
| `token_count` | `int` | Whitespace-tokenized word count |
| `content_hash` | `str` | MD5 hash of text for deduplication |
| `url` | `str` | Source URL |
| `date` | `str` | Crawl date |
| `dump` | `str` | CommonCrawl dump identifier |
| `title` | `str` | Page title |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{TARGET_REPO}")
print(ds["train"][0])
```
""")

card.push_to_hub(TARGET_REPO, token=os.environ["HF_TOKEN"])
print("Done.")
