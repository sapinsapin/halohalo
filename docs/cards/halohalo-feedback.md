---
language:
- bcl
- ceb
- eng
- fil
- hil
- ilo
- pag
- pam
- tsg
- war
pretty_name: halohalo model feedback
size_categories:
- n<1K
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- philippines
- philippine-languages
- human-feedback
- rlhf
- evaluation
- speech
---

# halohalo — model feedback

Human judgements on outputs from the
[halohalo speech demo](https://huggingface.co/spaces/sapinsapin/halohalo-dashboard),
collected to build evaluation sets and preference data for the Philippine-language
speech models.

**This dataset is private.** It contains audio submitted by people using the
demo, including their own microphone recordings, so it is not published.

## How rows get here

Every tab in the demo has a *"Was this any good?"* control. When someone clicks
👍 or 👎, one row is appended with the full context of that example — the model,
the language, the input, and the output. Audio is included only when the rater
leaves the **"Include the audio"** box ticked; unticking it stores text alone.

Nothing is recorded unless someone clicks a rating button. Running a model
without rating it leaves no trace here.

## Schema

`data/ratings.jsonl` — one JSON object per rating:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Row id; also the stem of any audio files |
| `ts` | `str` | UTC timestamp, second resolution |
| `space` | `str` | Space the rating came from |
| `rating` | `str` | `good` / `bad` |
| `task` | `str` | `asr` / `tts` / `s2s` |
| `language` | `str` | ISO 639-3; for `s2s` this is the **target voice's** language |
| `model` | `str` | Exact model repo that produced the output |
| `voice` | `str` | Speaker preset used (TTS and voice conversion) |
| `input_text` | `str` | Text the user typed (TTS) |
| `output_text` | `str` | Text the model produced (ASR) |
| `reference_text` | `str` | Ground truth, when the input was a preloaded clip |
| `comment` | `str` | Optional free-text note from the rater |
| `audio_kept` | `bool` | Whether the rater allowed audio to be stored |
| `input_audio` | `str` | Path under `data/`, or null |
| `output_audio` | `str` | Path under `data/`, or null |

Audio is 16 kHz mono WAV under `data/audio/`.

## Reading it

```python
from datasets import load_dataset

fb = load_dataset("json", data_files="data/ratings.jsonl", split="train")
bad_asr = fb.filter(lambda r: r["task"] == "asr" and r["rating"] == "bad")
```

`reference_text` is the useful column for ASR triage: where it is present you
can compute the actual error against ground truth and check whether the rater's
judgement agrees with the metric.

## Intended use

- **Evaluation** — a human-judged test set per language, which the corpus's own
  random 90/10 split cannot provide (its speakers overlap between train and test).
- **Preference data** — once the same input has been rated across model
  versions, rows can be paired into preferred/rejected examples for RLHF or DPO.
- **Error analysis** — the free-text comments point at failure modes that
  aggregate WER/CER hides, such as prosody or truncated endings.

Treat the ratings as noisy: they come from whoever visited the demo, with no
annotator agreement measured.

## Related

- Demo — [halohalo-dashboard](https://huggingface.co/spaces/sapinsapin/halohalo-dashboard)
- Training corpus — [pld](https://huggingface.co/datasets/sapinsapin/pld)
- Code — [github.com/sapinsapin/halohalo](https://github.com/sapinsapin/halohalo)
