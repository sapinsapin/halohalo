---
title: Halohalo Dashboard
emoji: 🍧
colorFrom: green
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: true
license: mit
short_description: Live org dashboard + speech demo for 10 Philippine languages
sdk_version: 6.22.0
models:
- sapinsapin/whisper-small-pld-bcl
- sapinsapin/whisper-small-pld-ceb
- sapinsapin/whisper-small-pld-eng
- sapinsapin/whisper-small-pld-fil
- sapinsapin/whisper-small-pld-hil
- sapinsapin/whisper-small-pld-ilo
- sapinsapin/whisper-small-pld-pag
- sapinsapin/whisper-small-pld-pam
- sapinsapin/whisper-small-pld-tsg
- sapinsapin/whisper-small-pld-war
- sapinsapin/speecht5_tts-pld-bcl
- sapinsapin/speecht5_tts-pld-ceb
- sapinsapin/speecht5_tts-pld-eng
- sapinsapin/speecht5_tts-pld-fil
- sapinsapin/speecht5_tts-pld-hil
- sapinsapin/speecht5_tts-pld-ilo
- sapinsapin/speecht5_tts-pld-pag
- sapinsapin/speecht5_tts-pld-pam
- sapinsapin/speecht5_tts-pld-tsg
- sapinsapin/speecht5_tts-pld-war
- sapinsapin/speecht5_vc-pld
tags:
- philippines
- philippine-languages
- low-resource
- speech
- multilingual
---

# halohalo — org dashboard and speech demo

Two things in one Space:

1. **A live dashboard** of everything in the
   [sapinsapin](https://huggingface.co/sapinsapin) org — Philippine-language
   corpora (speech, web text, literary text) and the models finetuned on them.
2. **An interactive demo** of the org's speech models for **ten Philippine
   languages**: Bikol, Cebuano, Filipino, Hiligaynon, Ilocano, Kapampangan,
   Pangasinan, Tausug, Waray and Philippine English.

| Tab | Model | What it does |
|---|---|---|
| 🎙️ Transcribe | `whisper-small-pld-<lang>` | Record, upload, or load a preloaded clip and transcribe it |
| 🔊 Synthesize | `speecht5_tts-pld-<lang>` | Type text, hear it spoken by one of that language's speakers |
| 🎭 Convert voice | `speecht5_vc-pld` | Speak, hear yourself in another speaker's voice |

Both input tabs take **microphone recording** as well as file upload, and
Transcribe ships two preloaded clips per language with reference transcripts
so you can compare the model against ground truth without recording anything.

## Notes

The dashboard holds no hardcoded repo list — each page load queries the Hub
API, so new datasets and models appear automatically. Private repos are
included in the counts and listed as rows with their names withheld.

This runs on a free CPU Space, so the first request per language downloads a
model (~1GB for ASR) and inference takes a few seconds; models are cached
after that. The speech tabs import their ML dependencies lazily, so the
dashboard keeps working even if those tabs cannot start.

The models are **baselines**, not state-of-the-art: they were trained on
prompted read speech recorded in controlled sessions, so accuracy drops on
spontaneous or noisy audio. ASR ranges from 5.9% WER on Philippine English to
40% on Kapampangan — the [dataset
card](https://huggingface.co/datasets/sapinsapin/pld) has the full table.
Voice-conversion output length is bounded relative to the input, because that
checkpoint does not yet predict its stop token reliably.

## Configuration

- `DASHBOARD_ORG` (env, optional) — org to display; defaults to `sapinsapin`.
- `HF_TOKEN` (Space secret, optional) — a **read-scoped** token that can see
  the org's private repos, making the private rows fully live. Without it the
  app falls back to `private_manifest.json`, a name-free snapshot holding only
  the count of private repos per type (regenerate it when private repos are
  added or removed — or just set the secret).

## Credits

Preloaded clips and voice presets come from the Philippine Language Dataset,
collected by the **UP Diliman Digital Signal Processing Laboratory**, and are
included solely to demonstrate the models. Code:
[github.com/sapinsapin/halohalo](https://github.com/sapinsapin/halohalo).
