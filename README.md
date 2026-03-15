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

## Text Corpus — BantayWika (FineWeb-compatible)

See [`bantaywika/`](bantaywika/) for the full pipeline.

`bantaywika/process_corpus.py` processes Philippine text corpora from `CORPUS_TEXT_DIR` into FineWeb-compatible JSONL and `bantaywika/push_corpus_to_hub.py` uploads to Hugging Face.

Published dataset: [sapinsapin/BantayWika](https://huggingface.co/datasets/sapinsapin/BantayWika)

A FineWeb-compatible pretraining text corpus for Filipino, Cebuano, and Ilocano assembled from literary, news, and transcribed sources. See [`bantaywika/README.md`](bantaywika/README.md) for full details.

```bash
source venv/bin/activate && python bantaywika/process_corpus.py && python bantaywika/push_corpus_to_hub.py
```
