---
language:
- tl
- en
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- filipino
- tagalog
- taglish
- code-switching
- speech
- spontaneous-speech
- livestream
---

# halo-livestream

Phrase-level segments from diarized Taglish (Tagalog–English code-switched)
livestream recordings, processed for ASR and TTS training. Part of the
halohalo Philippine-language dataset family.

Two configs:

| Config | Sample rate | Contents |
|---|---|---|
| `data/asr` | 16kHz mono | All gated segments, Whisper-ready |
| `data/tts` | 24kHz mono | Strict subset: forced-aligned, non-overlapping, quality-gated |

## Usage

**ASR / Whisper fine-tuning:**
```python
from datasets import load_dataset, Audio
ds = load_dataset("sapinsapin/halo-livestream", data_dir="data/asr")
# precise-timing subset:
ds = ds.filter(lambda x: x["alignment"] == "forced" and x["asr_cer"] <= 0.25)
```

**TTS:**
```python
ds = load_dataset("sapinsapin/halo-livestream", data_dir="data/tts")
# per-speaker corpora via the namespaced speaker_id column
```

## Schema

| Field | Type | Description |
|---|---|---|
| `audio` | `Audio` | Mono WAV segment (16kHz asr / 24kHz tts) |
| `sentence` | `str` | Human transcription (bracket tags stripped) |
| `language` | `str` | `tgl-eng` (Taglish) or ISO 639-3 code |
| `duration` | `float` | Segment duration (s) |
| `speaker_id` | `str` | `{recording_id}#S{n}` — unique across recordings |
| `gender` | `str` | From source speaker profile |
| `role` | `str` | Speaker role description from source metadata |
| `speech_type` | `str` | `spontaneous` |
| `source` | `str` | Source recording id |
| `start`, `end` | `float` | Position in the source recording (s) |
| `alignment` | `str` | `forced` (MMS CTC) / `interpolated` / `exact` |
| `align_score` | `float` | Forced-alignment confidence (0–1), null if not forced |
| `asr_cer` | `float` | CER between transcript and Whisper large-v3 round-trip |
| `overlap` | `bool` | Heuristic overlapped-speech flag |
| `speech_ratio` | `float` | Fraction of segment covered by VAD speech |
| `snr_db` | `float` | Speech/nonspeech energy ratio within the source block |
| `lufs` | `float` | Integrated loudness |
| `clip_ratio` | `float` | Fraction of clipped samples |

## Processing

Block-level diarized transcripts were segmented per speaker turn, then
upgraded with MMS-300M CTC forced alignment (romanization-based — handles
code-switching), silero-VAD boundary snapping, and a faster-whisper large-v3
round-trip CER score per segment. See the
[pipeline documentation](https://github.com/sapinsapin/halohalo) for stage
details and export gate thresholds.

## Splits

File-level split (`md5(recording_id) % 10`): speakers never cross splits.
`test` populates as the corpus grows.

## Quality caveats

- Spontaneous multi-party livestream audio: background music, notification
  sounds, and overlapping speech occur. Overlap is flagged heuristically, not
  removed from the ASR config.
- Transcripts are human-quality but unverified; `asr_cer` is provided as a
  per-segment quality signal — filter to taste.
- Speaker attribution errors in the source cannot be fully detected
  automatically (right words, wrong speaker label).
- Source audio is lossy 44.1kHz AAC; the 24kHz TTS config is suitable for
  training but is not studio-grade.
- Speech is from public livestream broadcasts and may contain real names and
  handles. Contact the maintainer for takedown requests.

## Licensing

Research use. Refer to the source platform's terms for the underlying
broadcast content.
