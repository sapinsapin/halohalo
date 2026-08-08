---
language:
- fil
- tl
license: mit
pretty_name: Filipino Speech Corpus (FSC)
size_categories:
- 100K<n<1M
task_categories:
- automatic-speech-recognition
- text-to-speech
task_ids:
- keyword-spotting
tags:
- filipino
- tagalog
- philippines
- speech
- read-speech
- spontaneous-speech
- low-resource
citation: >-
  @article{sagumdevelopment, title={DEVELOPMENT OF A FILIPINO SPEECH CORPUS},
  author={Sagum, Ramil} }
---

# Filipino Speech Corpus (FSC)

**Studio-recorded Filipino read, spontaneous, and word-level speech — 125 speakers, packaged as ready-to-stream Parquet.**

<div align="center">

**313,322 transcribed segments · 65.1 hours · 125 speakers · 16kHz mono**

[![Models](https://img.shields.io/badge/finetuned_models-2-blue)](https://huggingface.co/sapinsapin)
[![Code](https://img.shields.io/badge/pipeline-github-black)](https://github.com/sapinsapin/halohalo)

</div>

This is the Filipino Speech Corpus (Sagum), recorded in a controlled setting and
hand/machine transcribed with Transcriber. This repo
repackages the original `.wav` + `.trs` volumes as segment-level Parquet with
inline audio, so you can stream it without downloading and parsing XML.

> **New here? Start with the [30-second quickstart](#30-second-quickstart), then read
> [Before you train](#-before-you-train) — the segment length distribution will
> surprise you.**

---

## 30-second quickstart

```python
from datasets import load_dataset

ds = load_dataset("sapinsapin/filipinospeechcorpus", split="train", streaming=True)
row = next(iter(ds))

print(row["sentence"], row["duration"], row["speech_type"])
# audio arrives decoded as a numpy array at 16kHz
```

Full download (6.9 GB):

```python
ds = load_dataset("sapinsapin/filipinospeechcorpus")   # train + test
```

---

## ⚠️ Before you train

**This corpus is mostly single words, not sentences.** The median segment is
**0.62 seconds** and 95% of segments are under 1.3 s, because 56% of the corpus
is machine-pre-segmented word tokens intended for unit-selection synthesis and
keyword work. If you load it and train directly, you will train on isolated
words.

For sentence-level ASR or TTS you want the read/spontaneous portion with a
length filter — which leaves roughly **8,500 utterances (~7 hours)**:

```python
ds = ds.filter(lambda x:
    x["speech_type"] in ("read", "spontaneous")
    and 1.5 <= x["duration"] <= 30.0
    and x["num_words"] >= 3
)
```

Three more things worth knowing before you spend GPU hours:

| Gotcha | Detail | What to do |
|---|---|---|
| Extreme outliers | Longest segment is **1,640 s** (27 min); shortest is ~0 s | Always bound `duration` |
| Speaker overlap | Split is a random 90/10 over *segments*, so speakers appear in both | Re-split on `speaker_id` for speaker-disjoint eval |
| Narrow demographics | 97.5% of segments come from the 20–27 age band | Don't claim age robustness |

---

## What's inside

All statistics on this card are computed from the source corpus transcriptions
(313,322 segments); 305,246 of those survive into the published Parquet — the
rest are empty turns, control markers, or turns whose audio is missing.

**Splits (published rows)**

| Split | Segments |
|---|---|
| `train` | 274,730 |
| `test` | 30,516 |
| **total** | **305,246** |

**Speech types** — the three source volumes, and the reason the length
distribution is bimodal:

| `speech_type` | Segments | Share | Speakers | What it is |
|---|---|---|---|---|
| `machine` | 175,854 | 56.1% | 64 | Machine pre-segmented word tokens from read speech (Volume 6) |
| `read` | 130,001 | 41.5% | 50 | Hand-transcribed read speech — paragraphs, sentences, word lists |
| `spontaneous` | 7,467 | 2.4% | 65 | Hand-transcribed free speech (Volume 5) |

**Duration**

| | |
|---|---|
| Median | 0.62 s |
| Mean | 0.75 s |
| p95 | 1.21 s |
| Min / Max | ~0 s / 1,639.6 s |
| Total | 65.1 h |

**Speakers** — 125, near-balanced by gender (51.8% male / 48.2% female of
segments). Age skews hard young: 97.5% in `20-27`, 1.6% in `28-35`, 0.9% in
`36-43`.

---

## Schema

| Field | Type | Description |
|---|---|---|
| `audio` | `Audio(16000)` | 16 kHz mono segment, decoded on access |
| `sentence` | `str` | Transcription as written by the annotator |
| `duration` | `float` | Segment length in seconds |
| `num_words` | `int` | Whitespace word count |
| `speaker_id` | `str` | Speaker number from the filename |
| `gender` | `str` | `male` / `female` / `unknown` — recorded by observation |
| `age_group` | `str` | `20-27`, `28-35`, `36-43`, `44-51`, `52-60`, `unknown` |
| `speech_type` | `str` | `read` / `spontaneous` / `machine` |
| `source_file` | `str` | Original `.trs` stem, for tracing back to the corpus |

Speaker metadata is decoded from the corpus filename convention
(`09_xx00xxxx_15A` → speaker 09, male, age band 20–27, session 1, set 5A).

---

## Models trained on this data

Reference finetunes, trained by the same pipeline, so you can hear/measure what
the corpus supports before committing to your own run:

| Model | Task | Base | Notes |
|---|---|---|---|
| [`sapinsapin/speecht5_tts-fsc`](https://huggingface.co/sapinsapin/speecht5_tts-fsc) | Text-to-speech | `microsoft/speecht5_tts` | 1,000 steps on 1,867 read clips · eval loss 0.443 · [listen to samples](https://huggingface.co/sapinsapin/speecht5_tts-fsc/tree/main/samples) |
| [`sapinsapin/whisper-small-fsc`](https://huggingface.co/sapinsapin/whisper-small-fsc) | Speech recognition | `openai/whisper-small` | Filipino ASR finetune, WER/CER reported on the held-out split |

Both are **demonstration baselines on a single 8 GB GPU**, not
state-of-the-art — they exist to prove the data path end to end and to give you
a known-good starting configuration.

Reproduce either in one command:

```bash
python finetune_tts.py --dataset fsc --push
python finetune_asr.py --dataset fsc --push
```

---

## How it was built

1. Parse Transcriber `.trs` XML from *Volume 6 (Transcriptions)* — three
   directories, one per `speech_type`.
2. Slice each turn out of the corresponding `.wav` at its annotated
   `start`/`end`, resample to 16 kHz mono.
3. Drop empty turns and Transcriber control markers (`..`, `{...}` events).
4. Decode speaker/gender/age from the filename convention.
5. Shard to Parquet with audio inline, 90/10 random split.

Of 313,322 transcribed segments in the source, 305,246 survive; the remainder
are empty turns, control markers, or turns whose audio is missing.

Pipeline source: [`process_fsc_parquet.py`](https://github.com/sapinsapin/halohalo/blob/main/process_fsc_parquet.py)

---

## Limitations

- **Not a sentence corpus.** See [Before you train](#-before-you-train).
- **Read speech is scripted.** Prosody reflects reading, not conversation.
- **Age and register are narrow** — young adults, mostly in one setting.
- **Transcription conventions vary** between the hand-transcribed and
  machine-pre-segmented volumes; the machine volume is word-aligned output, not
  editorial transcription.
- **No dialect labels.** The corpus is Filipino/Tagalog; regional variation is
  not annotated.
- Original recording notes flag per-speaker irregularities (e.g. a wrong prompt
  set given to speaker 66, and several speakers withdrawn from the corpus).

---

## Related datasets

Part of the **halohalo** Philippine-language speech family:

| Dataset | What it covers | Scale |
|---|---|---|
| **filipinospeechcorpus** *(this one)* | Filipino studio read + spontaneous speech | 305k segments · 65 h |
| [`sapinsapin/pld`](https://huggingface.co/datasets/sapinsapin/pld) | **10 Philippine languages**, prompted recordings | 334k utterances · 448 h |
| [`sapinsapin/halo-livestream`](https://huggingface.co/datasets/sapinsapin/halo-livestream) | Taglish code-switched livestream speech | seed release |

---

## License, source and citation

The recordings originate from the Filipino Speech Corpus developed by Ramil
Sagum. This repackaging is distributed for research use — **cite the original
corpus**, not this repo:

```bibtex
@article{sagumdevelopment,
  title={DEVELOPMENT OF A FILIPINO SPEECH CORPUS},
  author={Sagum, Ramil}
}
```

Paper: [Development of a Filipino Speech Corpus](http://www.wins.or.kr/DataPool/Board/xxxx/18xx/1812/DEVELOPMENT%20OF%20A%20FILIPINO%20SPEECH%20CORPUS.pdf)

If you represent the corpus authors and want the terms or attribution changed,
please open a discussion on this repo.

---

## Contributing

Philippine languages are under-served in speech ML, and this family is built in
the open so others can pick it up. Useful contributions:

- Report bad segments via the **Community** tab (include `source_file`)
- Share finetunes trained on it — tag `sapinsapin/filipinospeechcorpus` in your
  model card and it will appear in this repo's "used by" list
- Improve the pipeline: [github.com/sapinsapin/halohalo](https://github.com/sapinsapin/halohalo)
