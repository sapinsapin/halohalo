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
LIVESTREAM_DIR=/path/to/livestream_raw
LIVESTREAM_OUTPUT_DIR=/path/to/livestream_output
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

## Speech Dataset — Philippine Language Dataset (PLD)

`process_pld_parquet.py` packages the UP-DSP Philippine Language Dataset — a
multilingual corpus of prompted recordings across Philippine languages — into a
Hugging Face Parquet dataset, using the same shard-and-push design as FSC.

Published dataset: `sapinsapin/pld` — **334,268 utterances · 448.2 hours ·
980 speakers · 10 languages** (Bikol, Kapampangan, Cebuano, Filipino,
Hiligaynon, Ilocano, Waray, English, Pangasinan, Tausug). Created **private**
by default, since PLD is a third-party corpus; pass `--public` once
redistribution terms are confirmed.

PLD needs no segmentation stage: it ships one WAV per prompt with the text
stored inline in each session `.log`, so the pipeline is index → shard → upload.

- Parses `Key = Value` session headers plus utterance rows, tolerating the
  corpus's own quirks: a UTF-8 BOM, the misspelled `SpekaerDialect` key,
  `NOT_RECORDED` placeholders, and transcripts containing embedded quotes
- Classifies prompts into `read` / `isolated` / `digits` / `spontaneous`
- Stores audio as 16kHz mono FLAC (lossless, about half the size of WAV)
- Resumable: each uploaded shard is journaled, so an interrupted run restarts
  where it stopped and can still rebuild the dataset card

Two corpus properties that materially affect training, both encoded as columns:

- **`text_is_prompt`** — `spontaneous` rows store the *elicitation question*
  put to the speaker, not a transcript of their answer. The same question
  repeats verbatim across speakers while the audio is 20-90s of free speech.
  Always filter these out of supervised training.
- **`language` vs `corpus_language`** — the English word/sentence lists
  (`EngW.txt`, `EngSen.txt`) are read by the same speakers, so they are labeled
  `language = "eng"` while `corpus_language` keeps the Philippine collection
  they came from. Filtering on `language` alone stays correct.

```bash
# .env
PLD_DIR=/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD
PLD_WORK_DIR=/mnt/d/halohalo/pld_shards
HF_PLD_REPO=sapinsapin/pld

source venv/bin/activate
python stats_pld.py                      # corpus statistics, no upload
python process_pld_parquet.py            # all languages → Hub
python process_pld_parquet.py --languages BIK --no-push   # local dry run
```

---

## Speech Dataset — Diarized Livestream Corpus (halo-livestream)

`process_livestream.py` is a staged, incremental pipeline that turns zipped
`{id}.json` + `{id}.mp4` diarized livestream recordings into TTS/ASR-ready
datasets. Built for hundreds of source files — drop new zips into
`LIVESTREAM_DIR` and re-run; only new files are processed.

Stages: **parse** (segment + dedup + 16kHz decode) → **align** (MMS CTC forced
alignment + silero VAD, GPU) → **qc** (faster-whisper round-trip CER + audio
metrics + overlap flag, GPU) → **export** (gated 16kHz ASR + 24kHz TTS sets,
audiofolder + parquet shards streamed to the Hub with resume).

```bash
source venv/bin/activate
python process_livestream.py                        # all stages, new files only
python process_livestream.py --stages export --push # publish to the Hub
python stats_livestream.py                          # quality report
```

```
# .env
LIVESTREAM_DIR=/mnt/d/halohalo/LivestreamCorpus/raw
LIVESTREAM_OUTPUT_DIR=/mnt/d/backup/dsp_bkp/Speech_Corpora/livestream_hf
LIVESTREAM_HF_REPO=sapinsapin/halo-livestream
```

Requires `ffmpeg`; GPU stages need the alignment/QC stack (see
[`docs/livestream_pipeline.md`](docs/livestream_pipeline.md) for the full
pipeline reference: stage design, export gates, resume semantics, gap
analysis).

---

## Text Corpus — BantayWika (FineWeb-compatible 1990-2012)

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

## Web Corpus — halo-halo (FineWeb-compatible Webscraped Data 2026)

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

## Finetuning workflows

Two dataset-swappable finetuning scripts consume the published corpora. Both
share the adapter in `halolib/finetune.py`, which normalizes either corpus to
`(audio@16k, text, speaker_id)` — switch datasets with a flag:

```bash
# TTS — SpeechT5 + x-vector speaker conditioning (fits an 8GB card, fp32 + grad ckpt)
python finetune_tts.py --dataset fsc --max-steps 1000
python finetune_tts.py --dataset livestream --max-samples 500

# ASR — Whisper-small seq2seq with WER/CER eval
python finetune_asr.py --dataset fsc
python finetune_asr.py --dataset livestream
```

Runs land in `$FINETUNE_DIR/{tts,asr}_{dataset}/` with checkpoints and (for
TTS) post-train synthesized sample WAVs. FSC is the default TTS corpus (read
speech); the halo-livestream TTS config becomes useful once enough files are
processed for its gated set to grow past demo size.

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
