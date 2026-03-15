# BantayWika

A FineWeb-compatible pretraining text corpus for Philippine languages, derived from the **Bantay-Wika** corpus collected by the University of the Philippines Sentro ng Wikang Filipino (UP-SWF) and the UP Digital Signal Processing (DSP) Laboratory.

The Bantay-Wika (Language Watch) project was started in 1994 by UP-SWF to track how the Philippine national language is used and develops, particularly in Philippine media. The first phase (1994–2004) involved manual collection and frequency counts from eleven major Philippine tabloids. The project was revived in March 2010 with UP-SWF partnering with the UP-DSP Laboratory, expanding to include Cebuano and Ilocano sources.

## Dataset

Published on Hugging Face: **[sapinsapin/BantayWika](https://huggingface.co/datasets/sapinsapin/BantayWika)**

### Sources

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

### Schema

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Document text |
| `id` | `str` | Unique document identifier |
| `source` | `str` | Corpus source label |
| `language` | `str` | ISO 639-3 language code |
| `token_count` | `int` | Whitespace-tokenized word count |

### Splits

| Split | Docs |
|---|---|
| `train` | 25,040 |
| `test` | 2,783 |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("sapinsapin/BantayWika")
print(ds["train"][0])
```

## Processing

- `bantaywika/process_corpus.py` — crawls source corpora, applies cleaning, outputs `train.jsonl` / `test.jsonl`
- `bantaywika/push_corpus_to_hub.py` — uploads to Hugging Face
- `bantaywika/update_corpus_card.py` — updates the dataset card only

### Quality Filters

- Minimum 50 words per document
- ≥40% Latin/ASCII characters
- Exact-match MD5 deduplication

### Cleaning Steps

- HTML entity decoding
- Control character removal
- Tokenizer artifact repair (`n~g` → `ng`, split clitics rejoined)
- Footnote/reference marker removal
- Metadata and section header lines dropped
- Cebuano parenthetical Tagalog translations stripped
- Whitespace normalization

## Citation

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
