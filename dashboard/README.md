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
- sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_ceb
- sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_pam
- sapinsapin/omniASR_W2V_1B_SSL-ctc-syllable-pld_ceb
- sapinsapin/omniASR_W2V_1B_SSL-ctc-syllable-pld_pam
- sapinsapin/whisper-large-v3-pld-ceb
- sapinsapin/whisper-large-v3-pld-pam
- sapinsapin/whisper-small-fsc
- sapinsapin/whisper-small-fsc-pld-fil
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
- sapinsapin/orpheus-3b-0.1-pretrained-char-pld-bcl
- sapinsapin/orpheus-3b-0.1-pretrained-char-pld-ceb
- sapinsapin/orpheus-3b-0.1-pretrained-char-pld-eng
- sapinsapin/orpheus-3b-0.1-pretrained-char-pld-fil
- sapinsapin/orpheus-3b-0.1-pretrained-char-pld-hil
- sapinsapin/orpheus-3b-0.1-pretrained-char-pld-ilo
- sapinsapin/orpheus-3b-0.1-pretrained-char-pld-pam
- sapinsapin/orpheus-3b-0.1-pretrained-char-pld-tsg
- sapinsapin/orpheus-3b-0.1-pretrained-char-pld-war
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
| 📚 Datasets / 🤖 Models | the whole org, live | Every repo, with a **Status** column: ★ marks the model to use per language |
| 🎙️ Transcribe | 28 ASR models, chosen per language | Opens on the **★ RECOMMENDED** model; record, upload, or load a preloaded clip |
| 🔊 Synthesize (retired baseline) | `speecht5_tts-pld-<lang>` (private) | Type text and hear it — the retired SpeechT5 voices, kept because a CPU runs them |
| 🎧 Compare voices — the published TTS | `orpheus-3b-0.1-pretrained-char-pld-<lang>`, Qwen3-TTS base, SpeechT5, MMS-TTS | The same unseen sentences from a person and each TTS system, with scores |
| 🎭 Convert voice | `speecht5_vc-pld` | Speak, hear yourself in another speaker's voice |

Both input tabs take **microphone recording** as well as file upload, and
Transcribe ships two preloaded clips per language with reference transcripts
so you can compare the model against ground truth without recording anything.

## Which ASR model to pick

**Take the one marked ★ RECOMMENDED** — the dropdown opens on it. For nine of
the ten languages it is `whisper-large-v3-pld-<lang>-norm`, the most accurate
model this org has published, scored on speakers and sentences it never saw:

| language | CER | WER |
|---|---|---|
| Bikol | 4.6 % | 15.2 % |
| Cebuano | 10.8 % | 22.5 % |
| Filipino | 5.0 % | 12.2 % |
| Hiligaynon | 9.3 % | 18.7 % |
| Ilocano | 5.7 % | 20.6 % |
| Kapampangan | 5.1 % | 19.8 % |
| Pangasinan | 16.0 % | 30.7 % |
| Tausug | 6.6 % | 21.0 % |
| Waray | 7.8 % | 21.0 % |

It writes **lowercase, without punctuation or accent marks**: PLD marks stress
on about a third of words, and scoring with those counted as errors adds ten
points or more of word error for reasons unrelated to recognition. At 6 GB it
takes several minutes to load on this CPU the first time. Philippine English
has no recommended model yet — its run is being fixed.

The **fast baseline** is whisper-small (242M, ~1 GB, seconds per clip). Its
card reports an *in-domain* number — 2.6 % CER on Cebuano — from a split that
shares speakers and sentences with training; on unseen speakers the same
family scores several times worse. It is quick, not better. **Research**
entries (the non-normalised bake-off models, the Omnilingual CTC heads) are
there to compare.

**Two splits do not compare.** *In-domain* shares speakers and prompt
sentences with training; *frozen-disjoint* shares neither. The Transcribe tab
prints the split, and "normalised" where it applies, next to every number.

A **7B** encoder was trained too
([omniASR_W2V_7B_SSL-ctc-char-pld_ceb](https://huggingface.co/sapinsapin/omniASR_W2V_7B_SSL-ctc-char-pld_ceb)):
17.0 % CER on Cebuano, level with the 1B — scale did not help. At 24 GiB it
cannot load on this Space, so the tab names it without offering it.

## Orpheus: the newest TTS, heard rather than run

Nine languages now have an **Orpheus 3B** voice, a LoRA on a codec language
model. It is far more intelligible than the SpeechT5 voices the Synthesize tab
runs. Whether it beats Meta's MMS-TTS depends on the judge — see the second
table. Each system spoke the same 50 held-out recordings' sentences per
language, and an ASR judge scored the audio (round-trip CER, lower is better;
the human recording is the judge's own floor):

| | human | SpeechT5 | MMS-TTS | Orpheus 3B |
|---|---|---|---|---|
| Bikol | 0.6 % | 73.8 % | 7.9 % | **4.4 %** |
| Cebuano | 3.4 % | 47.0 % | 42.9 % | **6.1 %** |
| English (PH) | 0.0 % | 31.5 % | 2.1 % | **0.8 %** |
| Filipino | 1.9 % | 32.4 % | 5.9 % | **5.9 %** |
| Hiligaynon | 0.3 % | 108.2 % | 8.7 % | **5.2 %** |
| Ilocano | 0.9 % | 75.9 % | 13.7 % | **7.6 %** |
| Kapampangan | 1.3 % | 95.2 % | 7.4 % | **4.3 %** |
| Tausug | 0.0 % | 43.2 % | — | **12.5 %** |
| Waray | 0.7 % | 35.7 % | 7.7 % | **7.3 %** |

Every row was scored by one judge, so a row compares; across rows the judge
changes (whisper-large-v3 for Cebuano and Kapampangan, whisper-small
elsewhere). Three caveats travel with this table: 50 sentences cannot resolve
a one-point gap, so Filipino and Waray are ties with MMS; the judges are our
own models trained on the same corpus, so an independent judge is still owed;
and Orpheus conditions on a speaker label instead of cloning a voice, so its
speaker similarity (0.23–0.46) is mostly *below* SpeechT5's (0.34–0.53).
Pangasinan has no Orpheus voice: retrained at a quarter of the steps it scored the same 36.5 %, so 1,445 clips is simply too few.

**Qwen3-TTS 1.7B, untrained on any Philippine language**, is in Compare voices for Cebuano and Kapampangan as a zero-shot row: 8.8 % / 3.5 % CER with speaker similarity **0.77 / 0.78**, the closest voice match measured on anything here. Our own finetune of it is broken and not shown.

**A second judge disagrees.** Re-transcribed by Meta's MMS-1b-all instead of
our PLD-trained Whisper judges, the same audio ranks the other way round in
most languages (round-trip CER; MMS-1b-all has no Tausug model):

| | human | MMS-TTS | Orpheus 3B | Qwen3-TTS base |
|---|---|---|---|---|
| Bikol | 6.1 % | 9.2 % | 10.0 % | |
| Cebuano | 11.1 % | 18.1 % | 16.0 % | 13.9 % |
| English (PH) | 4.1 % | 3.9 % | 7.1 % | |
| Filipino | 6.6 % | 8.9 % | 12.0 % | |
| Hiligaynon | 3.4 % | 7.4 % | 9.7 % | |
| Ilocano | 11.6 % | 14.7 % | 17.5 % | |
| Kapampangan | 8.0 % | 11.2 % | 11.5 % | 9.5 % |
| Pangasinan | 9.3 % | 8.5 % | 19.5 % | |
| Waray | 4.8 % | 9.3 % | 14.0 % | |

Each judge plausibly favours audio from its own family: ours was fine-tuned on
the recordings Orpheus learned from, and MMS-1b-all comes from the project that
made MMS-TTS. So **Orpheus against MMS-TTS is unresolved**; people listening is
what settles it. What holds under both judges: Orpheus is intelligible and the
only Tausug TTS, and the untrained Qwen3-TTS base beats MMS-TTS on Cebuano and
Kapampangan.

It is not in the Synthesize tab because it cannot be, on free hardware:
measured on two CPU threads, it generates ~1.5 audio tokens a second — **about
2.7 minutes per three-second sentence**, holding 6.5 GiB. The Compare voices
tab plays clips rendered offline with the evaluation's exact settings instead.
Running it live would take a GPU Space.

## Rate what you hear

Each tab has a **"Was this any good?"** control. 👍 / 👎 plus an optional note
is stored in a private dataset,
[halohalo-feedback](https://huggingface.co/datasets/sapinsapin/halohalo-feedback),
together with the model, language, input and output — enough to build
human-judged evaluation sets and, over time, preference pairs for RLHF.

Audio is included only if you leave the **"Include the audio"** box ticked, and
**nothing is stored unless you click a rating button** — running a model
without rating it leaves no trace. Ratings need a write-scoped token in the
`FEEDBACK_TOKEN` (or `HF_TOKEN`) Space secret; without one the buttons still
work but the panel says ratings are not being saved.

## Notes

The dashboard holds no hardcoded repo list — each page load queries the Hub
API, so new datasets and models appear automatically. Private repos are
included in the counts and listed as rows with their names withheld.

This runs on a free CPU Space, so the first request for a model downloads and
loads it — ~1 GB for whisper-small, 3.6 GB for a 1B CTC model, 5.8 GB for
whisper-large-v3 — which takes **minutes**. Transcribing after that takes
seconds. Measured on 8 CPU cores with the files already on disk, for a 3.5 s
clip:

| model | load | transcribe |
|---|---|---|
| whisper-small (242M) | 168 s | 2.8 s |
| omniASR 1B + CTC | 230 s | 1.9 s |
| whisper-large-v3 (1.5B) | 364 s | 16.8 s |

The CTC models transcribe fastest despite being four times the size, because
CTC is a single forward pass with no autoregressive decoding. Since loading is
the expensive part, the cache is two-tier: **two whisper-small models stay
resident and the multi-gigabyte ones share a single slot**, so switching back
to a baseline is instant while switching between two big models reloads. That
is what keeps a 3.6 GB and a 5.8 GB model from being in memory together. The
speech tabs import their ML dependencies lazily, so the dashboard keeps
working even if those tabs cannot start.

Most of these models are **baselines**, not state-of-the-art: they were
trained on prompted read speech recorded in controlled sessions, so accuracy
drops on spontaneous or noisy audio. The in-domain WERs run from 3.7 % on
Philippine English to 22.8 % on Kapampangan, and the honest frozen-disjoint
WERs are far higher — 36.8 % for Cebuano's best model. Treat the in-domain
figures on the [dataset
card](https://huggingface.co/datasets/sapinsapin/pld) as an upper bound on
what you will see, not a forecast. Voice-conversion output length is bounded
relative to the input, because that checkpoint does not yet predict its stop
token reliably.

The models table has a **Licence** column because it is not uniform: the
corpus is CC-BY-NC, so everything trained on it is research-only regardless of
what the base model permitted, and some older repos are still labelled
`apache-2.0` from before that was settled.

## Configuration

- `asr_registry.json` — which ASR models the Transcribe tab offers, what to
  load each with, and the error rate and split its card reports. Regenerate it
  with `python make_asr_registry.py` after pushing new ASR models; without the
  file the tab falls back to one whisper-small per language. Models whose
  weights exceed 8 GiB are recorded with `runnable: false` and named in the
  tab but never loaded — offering one would take the Space down.
- `tts_compare.json` and `samples/compare/` — the Compare voices tab.
  Regenerate with `python make_tts_compare.py --eval <tts_eval dir> --results
  <results.json>` from the outputs of `scripts/tts_eval.py`; clips are stored
  as Ogg Vorbis (~25 KB each).
- `DASHBOARD_ORG` (env, optional) — org to display; defaults to `sapinsapin`.
- `HF_TOKEN` (Space secret, optional) — a **read-scoped** token that can see
  the org's private repos, making the private rows fully live. Without it the
  app falls back to `private_manifest.json`, a name-free snapshot holding only
  the count of private repos per type (regenerate it when private repos are
  added or removed — or just set the secret).

## Credits

Preloaded clips, voice presets and the human recordings in Compare voices come
from the Philippine Language Dataset, collected by the **UP Diliman Digital
Signal Processing Laboratory**, and are included solely to demonstrate the
models. The MMS-TTS clips were generated with Meta's
[`facebook/mms-tts`](https://huggingface.co/facebook/mms-tts) models
(CC-BY-NC 4.0) as a reference point. Code:
[github.com/sapinsapin/halohalo](https://github.com/sapinsapin/halohalo).
