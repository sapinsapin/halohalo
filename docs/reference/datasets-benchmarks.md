# Datasets and benchmarks

What exists for Philippine languages, what we can train on, and what must stay
untouched for evaluation.

**The one rule:** anything listed under *Benchmarks* never enters a training
mix. Once a benchmark is contaminated we lose the ability to make honest
claims, and there is no way to un-contaminate it.

---

## Ours

Published under [sapinsapin](https://huggingface.co/sapinsapin); cards in
[`../cards/`](../cards/).

| Dataset | Size | Notes |
|---|---|---|
| `filipinospeechcorpus` (FSC) | ~100 h | UP-DSP read + spontaneous Filipino. Our TTS gold set and eval anchor. |
| `pld` *(private)* | 448 h, 10 languages, 980 speakers | UP-DSP. Bilingual speakers read both English and Philippine-language prompts — a rare same-speaker cross-lingual seed. Redistribution terms unresolved. |
| `halo-livestream` | growing | Spontaneous Taglish from diarized streams, QC-gated by our own pipeline. The only spontaneous code-switched speech we control. |
| `BantayWika` | text | Literary/reference corpora, FineWeb-compatible. |
| `halohalo` (+ `halo-tgl`/`hil`/`bcl`) | text | Cleaned CommonCrawl web text per language. |

**Gap we own:** no public benchmark measures Taglish code-switching. Building
one from held-out livestream + FSC spontaneous data is the highest-value eval
contribution we can make, and it's cheap relative to model training.

---

## External training data

- **Common Voice** (Mozilla) — crowdsourced read speech with a Tagalog
  (`tl`) subset. CC0, so no licence friction. Volume for Philippine languages
  is modest; verify current hours before planning around it.
- **MMS / MMS-lab data** (Meta) — models and alignment data covering 1,000+
  languages including several Philippine ones. Mostly religious readings, so
  domain is narrow and speaker diversity limited — fine for pretraining
  signal, poor for evaluating conversational speech.
- **OpenSLR** — the standard host for open speech resources; Google's
  crowdsourced Southeast Asian read-speech corpora live here. **[unverified]**
  which Philippine-language sets are currently present — check during the
  public-corpus sweep.
- **SEACrowd catalogue** — the aggregator to search first; see
  [research-groups.md](research-groups.md).
- **Bloom Library / SIL** — literacy and children's materials in very many
  languages, often with audio. Useful for the smallest languages where
  nothing else exists. Licences vary per item.
- **LibriSpeech / LibriTTS / Common Voice (en)** — the English side of the
  S2S parallel-data manufacturing described in the SOTA plan.

---

## Benchmarks — never train on these

- **FLEURS** — 102-language read speech from FLoRes translations, with a
  Filipino (`fil`) split. The default cross-paper comparison point for ASR and
  speech translation. Its size makes it tempting as training data; don't.
- **FLoRes-200** — the text translation benchmark FLEURS derives from,
  covering Philippine languages. Use for MT quality in the S2S cascade.
- **CoVoST 2 / CVSS** — speech translation and speech-to-speech translation
  benchmarks. Also the methodological template for manufacturing our own
  pseudo-parallel S2S corpus (translate transcripts, synthesize targets).
- **Our held-out splits** — FSC, per-language PLD, and livestream test sets,
  frozen before the cluster runs.

---

## Evaluation metrics worth standardizing on

| Metric | For | Why |
|---|---|---|
| **CER** | ASR, TTS round-trip | Our default over WER: Taglish orthography varies at the word level (*ng/nang*, English spellings, no standard for borrowings), so WER punishes spelling choices rather than recognition errors. |
| **WER** | ASR, comparability | Report alongside CER because everyone else does — just don't select on it. |
| **UTMOS / NISQA** | TTS | Automatic MOS prediction. Cheap, runs per checkpoint in CI. Proxies only — they correlate with human MOS but do not replace it. |
| **Whisper round-trip WER/CER** | TTS | Synthesize, re-transcribe, compare to the input text. Catches intelligibility failures and skipped content that MOS proxies miss. Already the gate in our QC stage. |
| **Speaker similarity** (ECAPA-TDNN cosine) | TTS cloning, expressive S2S | Did the voice survive? |
| **ASR-BLEU** | S2ST | Transcribe the translated speech, score BLEU against the reference translation. The standard S2ST metric despite being cascade-dependent. |
| **BLASER 2.0** | S2ST | Text-free speech translation quality; avoids the ASR-error confound in ASR-BLEU. |
| **Human MOS / CMOS** | Everything, finally | Native Taglish-fluent listeners, paid. The only metric that decides "SOTA" credibly. Recruit through the community rather than a generic crowd platform. |

---

## Practical cautions

- **Domain mismatch is the silent killer.** Religious readings, parliamentary
  speech, and read prompts all differ from spontaneous conversation in rate,
  prosody, and vocabulary. A model trained mostly on read speech scores well
  on read benchmarks and disappoints real users. Keep spontaneous data in the
  mix and in the eval.
- **`text_is_prompt` in PLD.** Spontaneous rows store the *elicitation
  question*, not a transcript of the answer. Filter them from supervised
  training — this is encoded in our loaders, but any new pipeline must
  reproduce it.
- **Speaker overlap across splits** inflates every number. Split by speaker,
  not by utterance.
- **Provenance per shard.** Keep source attribution at shard granularity so a
  source can be excised later if permission changes — already the manifest
  discipline in `halolib`.
