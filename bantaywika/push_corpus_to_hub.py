import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import DatasetCard

load_dotenv(Path(__file__).parent / ".env")

OUTPUT_DIR = os.environ["CORPUS_OUTPUT_DIR"]
HF_REPO    = os.environ["HF_CORPUS_REPO"]
HF_TOKEN   = os.environ["HF_TOKEN"]

dataset = load_dataset(
    "json",
    data_files={
        "train": str(Path(OUTPUT_DIR) / "train.jsonl"),
        "test":  str(Path(OUTPUT_DIR) / "test.jsonl"),
    },
)

print(dataset)

dataset.push_to_hub(HF_REPO, token=HF_TOKEN)

CARD = """\
---
language:
- fil
- ceb
- ilo
task_categories:
- text-generation
- fill-mask
tags:
- fineweb
- philippine-languages
- tagalog
- cebuano
- ilocano
- pretraining
---

# BantayWika

A FineWeb-compatible pretraining text corpus for Philippine languages, derived from the **Bantay-Wika** corpus collected by the University of the Philippines Sentro ng Wikang Filipino (UP-SWF) and the UP Digital Signal Processing (DSP) Laboratory.

The Bantay-Wika (Language Watch) project was started in 1994 by UP-SWF to track how the Philippine national language is used and develops, particularly in Philippine media. The first phase (1994–2004) involved manual collection and frequency counts from eleven major Philippine tabloids. The project was revived in March 2010 with UP-SWF partnering with the UP-DSP Laboratory, expanding to include Cebuano and Ilocano sources.

## Citation

If you use this dataset, please cite the original Bantay-Wika work:

```
Ilao, Joel, and Jovy Peregrino. "Bantay-Wika: towards a Better Understanding of
the Dynamics of Filipino Culture and Linguistic Change." 2011.
https://www.academia.edu/3034996

Ilao, Joel P., Timothy Israel D. Santos, and Rowena Cristina L. Guevara.
"Comparative analysis of actual language usage and selected grammar and
orthographical rules for Filipino, Cebuano-Visayan and Ilokano: a Corpus-based
Approach." Digital Signal Processing Laboratory, EEEI, University of the
Philippines – Diliman.
https://scholar.google.com/citations?view_op=view_citation&hl=en&user=zC_dDnsAAAAJ&authuser=1&citation_for_view=zC_dDnsAAAAJ:u-x6o8ySG0sC
```

## Schema

Each record follows the FineWeb document schema:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Document text |
| `id` | `str` | Unique document identifier |
| `source` | `str` | Corpus source label |
| `language` | `str` | ISO 639-3 language code |
| `token_count` | `int` | Whitespace-tokenized word count |

## Sources

| Source | Language | Description |
|---|---|---|
| `filipiniana` | `fil` | Classic Tagalog literary texts (1879–1930s) |
| `proj_gutenberg` | `fil` | Project Gutenberg Tagalog texts |
| `newspaper_fil` | `fil` | Filipino-language newspaper articles (2005–2007) |
| `palito` | `fil` | Tagalog literary and religious texts |
| `filnet_novels` | `fil` | 100 Nobelang Tagalog — Filipino novel collection |
| `transcribed` | `fil` | Transcribed Filipino literary works |
| `isip_ceb` | `ceb` | Cebuano news articles |
| `isip_ilk` | `ilo` | Ilocano news articles |

## Quality Filters

- Minimum 50 words per document
- ≥40% Latin/ASCII characters (removes non-target-language noise)
- Exact-match deduplication by MD5 content hash

## Cleaning

- HTML entity decoding
- Control character removal
- Tokenizer artifact repair (`n~g` → `ng`, split clitics rejoined)
- Footnote/reference marker removal
- Metadata and section header lines dropped
- Cebuano parenthetical Tagalog translations stripped
- Whitespace normalization

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo}")
print(ds["train"][0])
```

## Code

Processing code is available at https://github.com/sapinsapin/pretraining

## Splits

| Split | Description |
|---|---|
| `train` | 90% of documents |
| `test` | 10% of documents |
""".format(repo=HF_REPO)

card = DatasetCard(CARD)
card.push_to_hub(HF_REPO, token=HF_TOKEN)

print("Done.")
