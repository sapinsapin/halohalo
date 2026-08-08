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
license: other
license_name: up-dsp-research
pretty_name: Philippine Language Dataset (PLD)
size_categories:
- 100K<n<1M
task_categories:
- automatic-speech-recognition
- text-to-speech
multilinguality:
- multilingual
tags:
- philippines
- philippine-languages
- low-resource
- multilingual
- speech
- bikol
- cebuano
- kapampangan
- ilocano
- hiligaynon
- waray
- pangasinan
- tausug
---

# Philippine Language Dataset (PLD)

**Ten Philippine languages, 980 speakers, 448 hours of prompted speech — one of the largest multilingual Philippine speech collections available as Parquet.**

<div align="center">

**334,268 utterances · 448.2 hours · 980 speakers · 10 languages · 16kHz mono**

[![Code](https://img.shields.io/badge/pipeline-github-black)](https://github.com/sapinsapin/halohalo)

</div>

Collected by the **University of the Philippines Diliman Digital Signal
Processing Laboratory**. Every row is one prompted recording: the corpus ships
pre-segmented WAVs with the prompt text stored inline in each session log, so
no forced alignment or segmentation was applied here.

Most Philippine language speech data stops at Tagalog. This one covers Bikol,
Cebuano, Kapampangan, Hiligaynon, Ilocano, Waray, Pangasinan and Tausug at
scale — languages with tens of millions of speakers and almost no public ASR/TTS
data.

> **Read [`text_is_prompt`](#-read-this-before-training) before you train.** 2,586 rows
> carry an elicitation *question* instead of a transcript, and silently training
> on them will poison your model.

---

## 30-second quickstart

```python
from datasets import load_dataset

ds = load_dataset("sapinsapin/pld", split="train", streaming=True)
row = next(iter(ds))

print(row["language_name"], "|", row["sentence"])
print(row["speech_type"], row["duration"], "s")
```

The training-ready filter, in full:

```python
ds = load_dataset("sapinsapin/pld", split="train")
ds = ds.filter(lambda x: not x["text_is_prompt"] and 0.3 <= x["duration"] <= 30.0)
```

One language at a time:

```python
bikol = ds.filter(lambda x: x["language"] == "bcl")
```

---

## ⚠️ Read this before training

### `text_is_prompt` — 2,586 rows have no transcript

Rows where `speech_type == "spontaneous"` do **not** carry a transcript. The
session logs store the *elicitation question* that was put to the speaker —
e.g. *"Saen an dream destination mo?"* — while the audio is 20–90 seconds of
their free-speech answer. The same question text repeats verbatim across
different speakers.

They are kept because the audio is genuine spontaneous speech (32 hours of it,
valuable for pretraining, VAD, diarization, or re-transcription), but they are
poison for supervised `(audio, text)` training:

```python
ds = ds.filter(lambda x: not x["text_is_prompt"])
```

### Other things to know

| Gotcha | Detail | What to do |
|---|---|---|
| Prompts, not transcripts | Text is what the speaker was *asked* to read; no one verified they read it exactly | Treat as weakly-supervised; round-trip ASR to score |
| Speaker overlap | Random 90/10 split over utterances, so speakers appear in both splits | Re-split on `speaker_id` for speaker-disjoint eval |
| `eng` is not native English | English word/sentence lists read by Filipino L2 speakers | Use `corpus_language` to see which collection they came from |
| Half the corpus is single words | 164k `isolated` rows average 2.3 s | Filter on `speech_type` for sentence-level work |

---

## What's inside

### Languages

| Code | Language | Utterances | Hours |
|---|---|---|---|
| `bcl` | Bikol | 62,488 | 95.8 |
| `pam` | Kapampangan | 57,595 | 84.1 |
| `ceb` | Cebuano | 56,928 | 58.5 |
| `fil` | Filipino | 50,993 | 52.1 |
| `ilo` | Ilocano | 29,688 | 51.1 |
| `hil` | Hiligaynon | 30,965 | 39.8 |
| `war` | Waray | 21,526 | 31.4 |
| `eng` | English | 14,024 | 20.8 |
| `pag` | Pangasinan | 5,566 | 8.7 |
| `tsg` | Tausug | 4,495 | 5.9 |

English word and sentence lists (`EngW.txt`, `EngSen.txt`) were read by the same
speakers. Those rows are labeled `language = "eng"`, while `corpus_language`
retains the Philippine language collection they came from, so per-language
filters stay clean either way.

### Speech types

| Type | Utterances | Hours | Mean | What it is |
|---|---|---|---|---|
| `read` | 158,121 | 302.9 | 6.9 s | Full prompted sentences — news, medical, literature, education, tourism. **Best material for TTS.** |
| `isolated` | 164,447 | 107.3 | 2.3 s | Single words and short phrases from word lists |
| `spontaneous` | 2,586 | 32.2 | 44.8 s | Free speech — **prompt-only text**, see the warning above |
| `digits` | 9,114 | 5.9 | 2.3 s | Spoken digit strings |

### Splits

| Split | Rows |
|---|---|
| `train` | 300,842 |
| `test` | 33,426 |

---

## Schema

| Field | Type | Description |
|---|---|---|
| `audio` | `Audio(16000)` | 16 kHz mono, FLAC-compressed in storage |
| `sentence` | `str` | Prompt text read by the speaker (see `text_is_prompt`) |
| `duration` | `float` | Seconds |
| `num_words` | `int` | Whitespace word count |
| `language` | `str` | ISO 639-3 of the spoken content (`eng` for English lists) |
| `language_name` | `str` | Human-readable language name |
| `corpus_language` | `str` | Language collection the session belongs to |
| `speech_type` | `str` | `read` / `isolated` / `digits` / `spontaneous` |
| `prompt_category` | `str` | Prompt list, e.g. `News`, `Medical`, `BodyParts` |
| `prompt_source` | `str` | Original prompt filename |
| `text_is_prompt` | `bool` | `true` when text is an elicitation question, not a transcript |
| `speaker_id` | `str` | Language-namespaced speaker key, e.g. `BIK_0800` |
| `gender` | `str` | `male` / `female` / `unknown` |
| `age` | `int` | Speaker age, `-1` when unrecorded |
| `speaker_dialect` | `str` | Self-reported dialect |
| `mother_dialect` / `father_dialect` | `str` | Parents' dialects — useful for contact/variation studies |
| `profession` | `str` | Self-reported profession |
| `session_id` | `str` | Recording session identifier |
| `session_environment` | `str` | Recording environment note |
| `source_file` | `str` | Original WAV stem |

The dialect fields are unusually rich for a speech corpus — speaker, mother and
father dialect are all recorded, which supports dialectometry and language-contact
work that most corpora can't.

---

## Models trained on this data

**Eleven reference finetunes** — a TTS model per language plus one multilingual
speech-to-speech model — each with listen-test samples in its `samples/`
directory. All were trained on a single 8 GB GPU with
[`finetune_tts.py`](https://github.com/sapinsapin/halohalo/blob/main/finetune_tts.py) /
[`finetune_s2s.py`](https://github.com/sapinsapin/halohalo/blob/main/finetune_s2s.py),
so they are baselines to hear and beat, not state-of-the-art:

| Language | TTS model (`microsoft/speecht5_tts` base) |
|---|---|
| Bikol | [`speecht5_tts-pld-bcl`](https://huggingface.co/sapinsapin/speecht5_tts-pld-bcl) |
| Cebuano | [`speecht5_tts-pld-ceb`](https://huggingface.co/sapinsapin/speecht5_tts-pld-ceb) |
| English (PH) | [`speecht5_tts-pld-eng`](https://huggingface.co/sapinsapin/speecht5_tts-pld-eng) |
| Filipino | [`speecht5_tts-pld-fil`](https://huggingface.co/sapinsapin/speecht5_tts-pld-fil) |
| Hiligaynon | [`speecht5_tts-pld-hil`](https://huggingface.co/sapinsapin/speecht5_tts-pld-hil) |
| Ilocano | [`speecht5_tts-pld-ilo`](https://huggingface.co/sapinsapin/speecht5_tts-pld-ilo) |
| Pangasinan | [`speecht5_tts-pld-pag`](https://huggingface.co/sapinsapin/speecht5_tts-pld-pag) |
| Kapampangan | [`speecht5_tts-pld-pam`](https://huggingface.co/sapinsapin/speecht5_tts-pld-pam) |
| Tausug | [`speecht5_tts-pld-tsg`](https://huggingface.co/sapinsapin/speecht5_tts-pld-tsg) |
| Waray | [`speecht5_tts-pld-war`](https://huggingface.co/sapinsapin/speecht5_tts-pld-war) |

**Speech-to-speech:** [`speecht5_vc-pld`](https://huggingface.co/sapinsapin/speecht5_vc-pld)
— any-to-any voice conversion across all ten languages, trained on
same-sentence cross-speaker pairs mined from PLD's shared prompt lists (the
corpus has no parallel translations, but many speakers reading the same prompt
is exactly the parallel data voice conversion needs).

Reproduce any of them in one command:

```bash
python finetune_tts.py --dataset pld --language ceb --push
python finetune_s2s.py --push
```

An ASR baseline (whisper-small per language) has **not** been trained yet — a
Bikol or Cebuano one would be the first of its kind in public. If you train
something on PLD, tag this dataset in your model card and it will appear here.

---

## How it was built

1. Walk every session directory; parse the per-session `.log` (speaker
   demographics header, then one row per utterance: WAV name, prompt list,
   prompt text).
2. Classify each utterance's `speech_type`. Explicit markers (`_Iso_`, `_Utt_`,
   `Spontaneous`, digits) are used where present; the corpus uses at least five
   naming conventions, so the ~55k rows with no marker are typed by the
   **measured median word count** of their prompt list rather than by guessing
   from the filename.
3. Repair double-encoded UTF-8 in transcripts (`hapÃºnan` → `hapúnan`) — 150 of
   166 affected lines recover; the rest are left intact rather than risk a worse
   string.
4. Resample to 16 kHz mono, encode FLAC, shard to Parquet, 90/10 random split.

1,943 rows (0.6%) reference WAVs that are not present in the archive and were
skipped.

Pipeline source: [`process_pld_parquet.py`](https://github.com/sapinsapin/halohalo/blob/main/process_pld_parquet.py)
· parser: [`halolib/pld.py`](https://github.com/sapinsapin/halohalo/blob/main/halolib/pld.py)

---

## Limitations

- **Prompted, not conversational.** Except for the 32 h spontaneous portion,
  this is people reading from lists. Prosody and vocabulary reflect that.
- **Transcripts are unverified prompts.** Nobody checked that speakers read the
  prompt exactly; expect a residual mismatch rate.
- **Coverage is uneven** — Bikol has 95.8 h, Tausug 5.9 h. Don't expect balanced
  multilingual behaviour without resampling.
- **Recording conditions vary** by session and are only loosely described in
  `session_environment`.
- **No held-out speaker split** is provided by default.
- Language codes follow ISO 639-3; `fil` and `tgl` distinctions in the wild are
  inconsistent, so filter on both if you merge with other corpora.

---

## Related datasets

Part of the **halohalo** Philippine-language speech family:

| Dataset | What it covers | Scale |
|---|---|---|
| **pld** *(this one)* | 10 Philippine languages, prompted | 334k utterances · 448 h |
| [`sapinsapin/filipinospeechcorpus`](https://huggingface.co/datasets/sapinsapin/filipinospeechcorpus) | Filipino studio read + spontaneous | 305k segments · 65 h |
| [`sapinsapin/halo-livestream`](https://huggingface.co/datasets/sapinsapin/halo-livestream) | Taglish code-switched livestream speech | seed release |

---

## License, source and citation

Collected by the **UP Diliman Digital Signal Processing Laboratory**. This is a
repackaging for research use; the underlying corpus terms are those of UP-DSP.
**Please credit the original collectors**, and contact UP-DSP for terms covering
uses beyond research.

If you represent UP-DSP and want attribution, terms, or access changed, please
open a discussion on this repo.

---

## Contributing

Eight of these ten languages have essentially no public ASR or TTS baseline.
That is the opportunity here.

- Train a baseline on any single language and tag this dataset in your model card
- Report bad rows via the **Community** tab (include `source_file` and `session_id`)
- Improve the pipeline: [github.com/sapinsapin/halohalo](https://github.com/sapinsapin/halohalo)
