---
language:
- tl
- fil
- en
pretty_name: halo-livestream (Taglish code-switched speech)
size_categories:
- n<1K
task_categories:
- automatic-speech-recognition
- text-to-speech
multilinguality:
- multilingual
tags:
- filipino
- tagalog
- taglish
- code-switching
- spontaneous-speech
- livestream
- conversational
- philippines
configs:
- config_name: asr
  data_files:
  - split: train
    path: data/asr/train/*.parquet
  default: true
- config_name: tts
  data_files:
  - split: train
    path: data/tts/train/*.parquet
---

# halo-livestream

**Real Taglish code-switching from livestreams — every segment carries forced-alignment confidence, ASR round-trip CER, SNR, loudness and overlap flags.**

<div align="center">

**62 segments · 3 speakers · seed release**

[![Code](https://img.shields.io/badge/pipeline-github-black)](https://github.com/sapinsapin/halohalo)

</div>

> ### 🌱 This is a seed release — 62 segments, about 7 minutes
>
> It exists to publish the **pipeline and the schema**, not to be a training
> corpus. Nothing here is big enough to train on. What is worth your time is the
> per-segment quality metadata below — and the processing code, which scales to
> as many recordings as you feed it.
>
> If you want volume today, use [`sapinsapin/pld`](https://huggingface.co/datasets/sapinsapin/pld)
> (448 h) or [`sapinsapin/filipinospeechcorpus`](https://huggingface.co/datasets/sapinsapin/filipinospeechcorpus) (65 h).

Why it exists: Filipino speakers switch between Tagalog and English *mid-clause*,
constantly. Studio corpora don't capture it because prompts are written in one
language. This pipeline targets natural code-switched speech and attaches enough
quality signal per segment that you can set your own bar instead of trusting an
opaque "clean" label.

---

## Quickstart

```python
from datasets import load_dataset

asr = load_dataset("sapinsapin/halo-livestream", "asr", split="train")
tts = load_dataset("sapinsapin/halo-livestream", "tts", split="train")

print(asr[0]["sentence"])
# 'Yes mi. Saglit lang, saglit lang pa- paalis paalis- paalis ako eh.'
```

Filter to precisely-timed, cleanly-transcribed segments — the pattern this
dataset is really shipping:

```python
good = asr.filter(lambda x:
    x["alignment"] == "forced"      # MMS CTC alignment, not interpolated
    and x["asr_cer"] <= 0.25        # transcript agrees with Whisper round-trip
    and not x["overlap"]            # no overlapping speech
    and x["speech_ratio"] >= 0.6    # mostly speech, not silence
)
```

---

## The two configs

| Config | Rate | Segments | Audio | Purpose |
|---|---|---|---|---|
| `asr` | 16 kHz | 53 | ~6.4 min | All gated segments, Whisper-ready |
| `tts` | 24 kHz | 9 | ~1 min | Strict subset: forced-aligned, non-overlapping, tighter quality gates |

The `tts` config is deliberately small — it is what survives TTS-grade gating,
and the ratio (9 of 53) is itself the useful signal about how much livestream
audio is actually usable for synthesis.

| | `asr` | `tts` |
|---|---|---|
| Forced-aligned | 49 / 53 | 9 / 9 |
| Median round-trip CER | 0.203 | 0.087 |
| Mean segment length | 7.3 s | 6.3 s |

---

## Schema

Standard fields:

| Field | Type | Description |
|---|---|---|
| `audio` | `Audio` | Mono segment (16 kHz `asr` / 24 kHz `tts`) |
| `sentence` | `str` | Human transcription, bracket tags stripped |
| `language` | `str` | `tgl-eng` (Taglish) or ISO 639-3 |
| `duration` | `float` | Segment length (s) |
| `speaker_id` | `str` | `{recording_id}#S{n}` — unique across recordings |
| `gender`, `role` | `str` | From the source speaker profile |
| `speech_type` | `str` | `spontaneous` |
| `source` | `str` | Source recording id |
| `start`, `end` | `float` | Position within the source recording (s) |

**Quality metadata — the part that makes this dataset useful.** Every segment is
scored, so you can pick a threshold instead of accepting someone else's:

| Field | Type | Description |
|---|---|---|
| `alignment` | `str` | `forced` (MMS-300M CTC) / `interpolated` / `exact` |
| `align_score` | `float` | Forced-alignment confidence 0–1; `null` when not forced |
| `asr_cer` | `float` | CER between the human transcript and a faster-whisper large-v3 round trip — **the single best "is this transcript right" signal** |
| `overlap` | `bool` | Heuristic overlapping-speech flag |
| `speech_ratio` | `float` | Fraction of the segment covered by VAD speech |
| `snr_db` | `float` | Speech/non-speech energy ratio |
| `lufs` | `float` | Integrated loudness |
| `clip_ratio` | `float` | Fraction of clipped samples |

---

## How it was built

1. **Diarized transcripts** at block level, segmented per speaker turn.
2. **Forced alignment** with MMS-300M CTC, romanization-based — this is the part
   that survives code-switching, where a Tagalog-only or English-only aligner
   drifts at every switch point.
3. **Boundary snapping** with silero-VAD, so segments start and end on speech.
4. **Round-trip scoring**: transcribe each segment with faster-whisper large-v3
   and record CER against the human transcript.
5. **Gating** into the `asr` and `tts` configs by alignment quality, overlap,
   CER, and loudness/clipping thresholds.

Pipeline source: [`process_livestream.py`](https://github.com/sapinsapin/halohalo/blob/main/process_livestream.py)
· docs: [`docs/livestream_pipeline.md`](https://github.com/sapinsapin/halohalo/blob/main/docs/livestream_pipeline.md)

---

## Models trained on this data

None — 62 segments is far too little, and publishing a model trained on it would
be misleading.

The trainers accept this dataset with a flag (`--dataset livestream`), so once
the corpus grows the recipe is already wired:

```bash
python finetune_asr.py --dataset livestream --push   # → whisper-small-halohaloLS
python finetune_tts.py --dataset livestream --push   # → speecht5_tts-halohaloLS
```

Working baselines on the sibling Filipino corpus, for reference:
[`speecht5_tts-fsc`](https://huggingface.co/sapinsapin/speecht5_tts-fsc) ·
[`whisper-small-fsc`](https://huggingface.co/sapinsapin/whisper-small-fsc)

---

## Limitations

- **Tiny.** 62 segments from a single source recording and 3 speakers. Any metric
  computed on it is noise.
- **Transcripts are human but imperfect** — median round-trip CER is 0.203 on the
  `asr` config, which reflects both genuine transcription variance and the fact
  that Taglish orthography is unstandardised (`nag-aano` / `nagaano` / `nag aano`).
- **CER is the honest metric here**, not WER: word-level scoring punishes
  legitimate spelling variation in code-switched text.
- **Numerals are not verbalized** — digits appear as digits in transcripts.
- **Speaker roles come from source metadata** and are not independently verified.
- Livestream audio carries background music, stream artifacts, and variable mic
  quality; `snr_db` and `clip_ratio` are there so you can see it.

---

## Related datasets

Part of the **halohalo** Philippine-language speech family:

| Dataset | What it covers | Scale |
|---|---|---|
| **halo-livestream** *(this one)* | Taglish code-switched livestream speech | 62 segments · seed |
| [`sapinsapin/pld`](https://huggingface.co/datasets/sapinsapin/pld) | 10 Philippine languages, prompted | 334k utterances · 448 h |
| [`sapinsapin/filipinospeechcorpus`](https://huggingface.co/datasets/sapinsapin/filipinospeechcorpus) | Filipino studio read + spontaneous | 305k segments · 65 h |

---

## Terms and ethics

**No license is asserted over the source audio.** It is livestream content whose
rights belong to the original broadcasters; this repo publishes derived segments
and metadata for research use only. Check the source terms before redistributing,
and treat this as research material, not a licensed corpus.

Speakers are identified only by an opaque
`{recording_id}#S{n}` key, with no names, handles, or channel identifiers in the
data. If you are a speaker in this data and want a segment removed, open a
discussion on this repo and it will be taken down.

Do not use this data to identify, profile, or synthesize the voice of any
individual speaker without their consent.

---

## Contributing

The pipeline is the point — it scales to whatever you feed it, and every stage
is documented.

- Run it on your own recordings: [github.com/sapinsapin/halohalo](https://github.com/sapinsapin/halohalo)
- Report bad segments via the **Community** tab (include `source` and `start`)
- Taglish/code-switching evaluation sets are badly needed; contributions welcome
