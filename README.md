# halohalo
Pre-training dataset for Philippine Languages

## Setup

```bash
wsl bash setup.sh
```

## Source Data

Currently processes the **Filipino Speech Corpus (FSC)**, a read and spontaneous speech corpus with hand-transcribed `.trs` transcription files.

The corpus is organized as:
- **Volumes 1–4** — read speech WAV files
- **Volume 5** — spontaneous speech WAV files
- **Volume 6** — transcriptions in `.trs` format (Transcriber XML)
  - `hand transcribed read speech/` — sentence-level alignments
  - `hand transcribed spontaneous speech/` — sentence-level alignments
  - `machine pre-segmented transcribed read speech/` — sentence-level alignments
  - `FSC_handsegmented_102008/` — phoneme/syllable-level alignments (excluded)

Only sentence-level `.trs` files are used. The phoneme-level files in `FSC_handsegmented_102008/` are excluded as they contain sub-word fragments unsuitable for ASR training.

## Processing

`process_fsc.py` parses the `.trs` files and extracts audio segments:

- Segments are sliced from the source WAV using timestamps in the `.trs` files
- Audio is resampled to **16kHz mono**
- Segments shorter than **0.3s** or longer than **30s** (Whisper's context window) are discarded
- Silence markers, noise tags, and empty segments are skipped
- Output is split **90% train / 10% test**

Each entry in `metadata.jsonl` contains:

| Field | Description |
|---|---|
| `file_name` | relative path to the segmented WAV |
| `sentence` | transcription text |
| `language` | `fil` |
| `duration` | segment duration in seconds |
| `speaker_id` | speaker identifier from filename |
| `gender` | `male` or `female` |
| `age_group` | age range of speaker |
| `speech_type` | `read`, `spontaneous`, or `machine` |

## Dataset Format

The dataset is published to Hugging Face using the `audiofolder` loader, which produces:

| Column | Type | Description |
|---|---|---|
| `audio` | `Audio(sampling_rate=16000)` | audio array + sampling rate |
| `sentence` | `str` | transcription text |
| + metadata columns | | speaker_id, gender, age_group, etc. |

This schema is directly compatible with Whisper fine-tuning pipelines. The `input_features` (log-mel spectrogram) and `labels` (tokenized transcription) are computed on-the-fly during training:

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Filipino", task="transcribe")

def preprocess(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
```

## Publishing

```bash
wsl bash setup.sh
wsl bash -c "source venv/bin/activate && python process_fsc.py && python push_to_hub.py"
```

## Text Corpus (FineWeb-compatible)

`process_corpus.py` processes Philippine text corpora from `CORPUS_TEXT_DIR` into FineWeb-compatible JSONL and `push_corpus_to_hub.py` uploads to Hugging Face.

```bash
wsl bash -c "source venv/bin/activate && python process_corpus.py && python push_corpus_to_hub.py"
```

Requires a `.env` file with:

```
CORPUS_DIR=/path/to/FilipinoSpeechCorpus
CORPUS_TEXT_DIR=/path/to/_Corpora_Main/Corpora
OUTPUT_DIR=/path/to/output
HF_REPO=sapinsapin/filipinospeechcorpus
HF_CORPUS_REPO=sapinsapin/philippine-text-corpus
HF_TOKEN=your_hf_token
```
