import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import DatasetCard

load_dotenv(Path(__file__).parent / ".env")

dataset = load_dataset(
    "audiofolder",
    data_dir=os.environ["OUTPUT_DIR"],
    name="filipino-speech-corpus",
)

print(dataset)

dataset.push_to_hub(
    os.environ["HF_REPO"],
    token=os.environ["HF_TOKEN"],
)

CARD = """---
language:
- fil
task_categories:
- automatic-speech-recognition
---

# Filipino Speech Corpus

## Dataset Summary

The Filipino Speech Corpus (FSC) is a read and spontaneous speech dataset for Filipino (Tagalog), originally developed for speech research in the Philippines. This version processes the original corpus into a format compatible with Mozilla Common Voice and Whisper fine-tuning pipelines.

## Common Voice Compatibility

This dataset follows the same schema as [Mozilla Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0), the standard benchmark dataset used in OpenAI's Whisper fine-tuning examples and the Hugging Face `transformers` ASR pipeline.

The key columns are:
- `audio` — `Audio(sampling_rate=16000)` object, matching Whisper's expected input sampling rate
- `sentence` — transcription text, matching the field name used by Common Voice and Whisper training scripts

This matters because Whisper fine-tuning scripts from Hugging Face and OpenAI are written expecting this exact schema. By conforming to it, this dataset can be dropped into any existing Whisper fine-tuning pipeline without modification — the `WhisperProcessor` computes `input_features` (log-mel spectrogram) and `labels` (tokenized transcription) on-the-fly from `audio` and `sentence` respectively.

## Supported Tasks

- Automatic Speech Recognition (ASR)
- Speech-to-Text

## Languages

Filipino (`fil`) / Tagalog

## Dataset Structure

Each example contains:
- `audio` — 16kHz mono WAV segment
- `sentence` — hand-transcribed text
- `speaker_id` — anonymized speaker identifier
- `gender` — `male` or `female`
- `age_group` — age range of the speaker
- `speech_type` — `read`, `spontaneous`, or `machine`
- `duration` — segment duration in seconds

## Data Splits

| Split | Description |
|---|---|
| `train` | 90% of all segments |
| `test` | 10% of all segments |

## Source Data

The original FSC contains recordings from over 150 speakers across multiple sessions. Transcriptions are in Transcriber XML (`.trs`) format with sentence-level alignments. Phoneme and syllable-level alignments are excluded.

Audio segments are filtered to between 0.3s and 30s (Whisper's maximum context window). Silence markers, noise tags, and empty transcriptions are discarded.

## Whisper Fine-tuning

This dataset is directly compatible with Whisper fine-tuning pipelines via the `audiofolder` loader:

```python
from datasets import load_dataset
from transformers import WhisperProcessor

dataset = load_dataset("sapinsapin/filipinospeechcorpus")
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

## Licensing

Please refer to the original Filipino Speech Corpus license terms before use.
"""

card = DatasetCard(CARD)
card.push_to_hub(
    os.environ["HF_REPO"],
    token=os.environ["HF_TOKEN"],
)
