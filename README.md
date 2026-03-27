# halohalo
Pre-training dataset pipeline for Philippine Languages

## Setup

```bash
wsl bash setup.sh
```

Requires a `.env` file with:

```
CORPUS_DIR=/path/to/FilipinoSpeechCorpus
CORPUS_TEXT_DIR=/path/to/_Corpora_Main/Corpora
OUTPUT_DIR=/path/to/fsc_output
CORPUS_OUTPUT_DIR=/path/to/corpus_output
HF_REPO=sapinsapin/filipinospeechcorpus
HF_CORPUS_REPO=sapinsapin/BantayWika
HF_TOKEN=your_hf_token
```

---

## Speech Dataset — Filipino Speech Corpus (FSC)

`process_fsc.py` processes the Filipino Speech Corpus into a Hugging Face Parquet dataset with raw 16kHz mono audio segments, compatible with Whisper fine-tuning pipelines.

Published dataset: [sapinsapin/filipinospeechcorpus](https://huggingface.co/datasets/sapinsapin/filipinospeechcorpus)

- Parses Transcriber XML (`.trs`) files across 3 speech types: read, spontaneous, machine
- Slices source WAV recordings into sentence-level segments at 16kHz mono
- Outputs `audio`, `sentence`, `duration`, `speaker_id`, `gender`, `age_group`, `speech_type`
- Schema matches [Mozilla Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) for drop-in Whisper fine-tuning compatibility

```bash
source venv/bin/activate && python process_fsc.py
source venv/bin/activate && python push_to_hub.py
```

How `sapinsapin/filipinospeechcorpus` was produced:

```bash
# .env
CORPUS_DIR=/mnt/c/halohalo/FilipinoSpeechCorpus
OUTPUT_DIR=/mnt/d/backup/dsp_bkp/Speech_Corpora/fsc_hf
HF_REPO=sapinsapin/filipinospeechcorpus

source venv/bin/activate
python process_fsc.py
python push_to_hub.py
```

---

## Text Corpus — BantayWika (FineWeb-compatible)

`bantaywika/process_corpus.py` processes Philippine text corpora into a FineWeb-compatible JSONL dataset.

Published dataset: [sapinsapin/BantayWika](https://huggingface.co/datasets/sapinsapin/BantayWika)

Sources: Filipiniana, Project Gutenberg, newspaper corpora, Palito, FilNet, ISIP (Cebuano + Ilocano).

```bash
source venv/bin/activate && python bantaywika/process_corpus.py && python bantaywika/push_corpus_to_hub.py
```

How `sapinsapin/BantayWika` was produced:

```bash
# .env
CORPUS_TEXT_DIR=/mnt/c/Users/carrot/Dropbox/_Corpora_Main/Corpora
CORPUS_OUTPUT_DIR=/mnt/d/backup/dsp_bkp/Speech_Corpora/corpus_hf
HF_CORPUS_REPO=sapinsapin/BantayWika

source venv/bin/activate
python bantaywika/process_corpus.py
python bantaywika/push_corpus_to_hub.py
```

See [`bantaywika/README.md`](bantaywika/README.md) for full details.

---

## Web Corpus — halo-* (FineWeb-compatible)

Pipeline for cleaning and preparing web-scraped Philippine language datasets from CommonCrawl into FineWeb-compatible format.

Published datasets:
- [sapinsapin/halo-hil](https://huggingface.co/datasets/sapinsapin/halo-hil) — Hiligaynon
- [sapinsapin/halo-tgl](https://huggingface.co/datasets/sapinsapin/halo-tgl) — Tagalog
- [sapinsapin/halo-bcl](https://huggingface.co/datasets/sapinsapin/halo-bcl) — Bikol
- [sapinsapin/halohalo](https://huggingface.co/datasets/sapinsapin/halohalo) — combined FineWeb corpus

### Step 1 — Clean

`clean_halo.py` strips web boilerplate, HTML, and markdown noise from the raw `text` column and adds a `text_cleaned` column.

```bash
source venv/bin/activate
python clean_halo.py sapinsapin/halo-hil
python clean_halo.py sapinsapin/halo-tgl
python clean_halo.py sapinsapin/halo-bcl
```

### Step 2 — Prepare (FineWeb)

`prep_halohalo.py` adds FineWeb-compatible columns (`source`, `language`, `token_count`, `content_hash`) and pushes to a target repo. Supports appending with MD5-based deduplication.

```bash
source venv/bin/activate
python prep_halohalo.py sapinsapin/halo-hil sapinsapin/halohalo
python prep_halohalo.py sapinsapin/halo-tgl sapinsapin/halohalo --append
python prep_halohalo.py sapinsapin/halo-bcl sapinsapin/halohalo --append
```

How `sapinsapin/halohalo` was produced:

```bash
source venv/bin/activate

# clean each source dataset
python clean_halo.py sapinsapin/halo-hil
python clean_halo.py sapinsapin/halo-tgl
python clean_halo.py sapinsapin/halo-bcl

# combine into a single FineWeb-compatible repo
python prep_halohalo.py sapinsapin/halo-hil sapinsapin/halohalo
python prep_halohalo.py sapinsapin/halo-tgl sapinsapin/halohalo --append
python prep_halohalo.py sapinsapin/halo-bcl sapinsapin/halohalo --append
```

---

## halolib

Reusable library used for preprocessing web-mined data used in `clean_halo.py` and `prep_halohalo.py`.

```
halolib/
├── cleaner.py    — clean_text(), is_usable()
└── fineweb.py    — add_fineweb_columns(), dedup_against(), append_to(), push_with_retry()
```

```python
from halolib import clean_text, is_usable
from halolib.fineweb import add_fineweb_columns, append_to, push_with_retry
```
