# halohalo
Pre-training dataset pipeline for Philippine Languages

## Setup

```bash
bash setup.sh
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

## Speech Dataset — Filipino Speech Corpus (FSC)

`process_fsc_parquet.py` processes the Filipino Speech Corpus into a Hugging Face Parquet dataset with raw 16kHz mono audio segments.

Published dataset: [sapinsapin/filipinospeechcorpus](https://huggingface.co/datasets/sapinsapin/filipinospeechcorpus)

- Parses Transcriber XML (`.trs`) files across 3 speech types: read, spontaneous, machine
- Slices source WAV recordings into sentence-level segments at 16kHz mono
- Outputs `audio`, `sentence`, `duration`, `num_words`, `speaker_id`, `gender`, `age_group`, `speech_type`, `source_file`
- Parallel processing via `ProcessPoolExecutor` + background upload thread
- Resume-safe: logs completed shards to `/tmp/fsc_shards/progress.jsonl`

```bash
source venv/bin/activate && python process_fsc_parquet.py
```

Monitor progress:

```bash
tail -f /tmp/fsc_shards/run.log
```

## Text Corpus — BantayWika (FineWeb-compatible)

See [`bantaywika/`](bantaywika/) for the full pipeline.

`bantaywika/process_corpus.py` processes Philippine text corpora from `CORPUS_TEXT_DIR` into FineWeb-compatible JSONL and `bantaywika/push_corpus_to_hub.py` uploads to Hugging Face.

Published dataset: [sapinsapin/BantayWika](https://huggingface.co/datasets/sapinsapin/BantayWika)

A FineWeb-compatible pretraining text corpus for Filipino, Cebuano, and Ilocano assembled from literary, news, and transcribed sources. See [`bantaywika/README.md`](bantaywika/README.md) for full details.

```bash
source venv/bin/activate && python bantaywika/process_corpus.py && python bantaywika/push_corpus_to_hub.py
```
