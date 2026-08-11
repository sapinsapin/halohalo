---
title: halohalo — Philippine speech models
emoji: 🇵🇭
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: true
short_description: ASR, TTS and voice conversion for 10 Philippine languages
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

# halohalo — Philippine speech models

Run 21 speech models for **ten Philippine languages** from the browser:
speech recognition, speech synthesis, and any-to-any voice conversion.

Languages: Bikol, Cebuano, Filipino, Hiligaynon, Ilocano, Kapampangan,
Pangasinan, Tausug, Waray, and Philippine English. For most of them these are,
as far as we know, the first public models of their kind.

| Tab | Model | What it does |
|---|---|---|
| 🎙️ Transcribe | `whisper-small-pld-<lang>` | Record, upload, or load a preloaded clip and transcribe it |
| 🔊 Synthesize | `speecht5_tts-pld-<lang>` | Type text, hear it spoken in a voice from that language |
| 🎭 Convert voice | `speecht5_vc-pld` | Speak, hear yourself in another speaker's voice |

Every tab takes **microphone input**; the Transcribe tab also ships two
preloaded clips per language with reference transcripts, so you can compare
the model against ground truth without recording anything.

## Notes

This runs on a free CPU Space, so the first request per language downloads a
model (~1GB for ASR) and inference takes a few seconds. Models are cached
after that.

These are **baselines**, not state-of-the-art. They were trained on prompted
read speech recorded in controlled sessions, so accuracy drops on spontaneous
or noisy audio. Reported ASR performance ranges from 5.9% WER on Philippine
English to 40% on Kapampangan — see the
[dataset card](https://huggingface.co/datasets/sapinsapin/pld) for the full
table and caveats.

Voice-conversion output length is bounded relative to the input, because that
checkpoint does not yet predict its stop token reliably.

## Links

- **Models** — [all 21 on the Hub](https://huggingface.co/sapinsapin)
- **Org dashboard** — [halohalo-dashboard](https://huggingface.co/spaces/sapinsapin/halohalo-dashboard)
- **Code** — [github.com/sapinsapin/halohalo](https://github.com/sapinsapin/halohalo)

## Credits

Preloaded clips and voice presets come from the Philippine Language Dataset,
collected by the **UP Diliman Digital Signal Processing Laboratory**, and are
included solely to demonstrate the models. Please credit UP-DSP if you build
on this work.
