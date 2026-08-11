# Research groups, orgs, and communities

Who else is building speech and language AI for languages the big labs skip.
Grouped by what we'd actually borrow from them.

---

## Aalto Speech Research Group (Finland)

**<https://github.com/aalto-speech>** · Aalto University

**What it is.** A university speech group with two decades of ASR work,
publishing most of it as open code. Their focus is Finnish — a morphologically
rich language with ~5M speakers — plus Northern Sami and other Uralic
languages. Notable repositories:

| Repo | What it does |
|---|---|
| `large-scale-monolingual-speech-foundation-models` | Speech foundation models trained on ~158k hours of Finnish |
| `colloquial-Finnish-wav2vec2` | wav2vec2 finetuned for *spoken* colloquial Finnish, not the written standard |
| `finnish-parliament-scripts` | Aligning long parliamentary audio to imperfect official transcripts |
| `kaldi-sb-north-sme` | Northern Sami ASR across Kaldi, SpeechBrain, and wav2vec2 |
| `morfessor`, `flatcat` | Unsupervised morphological segmentation into subword units — **we tested this on our data, see [tokenizers.md](tokenizers.md)** |
| `subword-kaldi` | Position-dependent phones in subword lexicon FSTs |
| `multitask-wav2vec2` | Joint ASR + speech classification |
| `speaker-diarization`, `PPG2Speech`, `slate-2025` | Diarization, pronunciation editing, spoken language assessment |

**Why it matters to us.** This is the closest thing to a template for what
we're attempting, and the parallels are unusually direct:

1. **Morphology.** Finnish is agglutinative with heavy inflection; Philippine
   languages are agglutinative with affixation, infixation, and reduplication
   (*sulat → sumulat → sinulatan*). Word-level vocabularies fail the same way
   in both. Morfessor came out of this tradition — and when we ran it on PLD
   it did find real Philippine affix boundaries (`mag+susulat`,
   `pinag+susulat+an`, `ka+sulat+an`), though it is blind to the `-um-`/`-in-`
   infixes Tagalog leans on. Full analysis in [tokenizers.md](tokenizers.md).
2. **Colloquial vs standard.** Their colloquial-Finnish work exists because
   spoken Finnish diverges sharply from written Finnish. That is exactly our
   Taglish-vs-formal-Filipino problem, and it's the reason we select on CER
   rather than WER.
3. **Long audio, imperfect transcripts.** The parliament alignment scripts
   solve the same problem as our livestream `align` stage — recovering
   segment-level training data from hours-long recordings whose transcripts
   are approximate. Worth reading before we scale that stage.
4. **One language, at scale.** The 158k-hour Finnish foundation model is
   precisely the thesis of [`sota_speech_plan.md`](../sota_speech_plan.md):
   a mid-size language beats multilingual-generalist models on its own turf
   if you assemble enough of its speech. They've demonstrated it.
5. **Northern Sami as the analogue for our small languages.** Sami has far
   fewer speakers than Finnish, and their approach — transfer from the larger
   related/neighbouring language, compare Kaldi vs SSL vs end-to-end honestly
   — is the model for Tausug, Pangasinan, and Waray inside PLD.

**Caveats.** Code is research-grade and Kaldi-era in places; some repos assume
a Slurm cluster and Finnish-specific resources. Take the methods and the
papers behind them, not necessarily the scripts.

---

## AI4Bharat (India)

**<https://ai4bharat.org>** · IIT Madras

**What it is.** An academic lab building open datasets and models across 22+
Indian languages — speech (IndicWav2Vec, IndicTTS, IndicASR), text (IndicNLP
corpora, IndicTrans MT), and evaluation suites.

**Why it matters to us.** The structural situation is nearly identical to the
Philippines: one country, many languages, a large multilingual population, and
near-zero commercial incentive for the biggest vendors to serve them. They
solved the coordination problem — university anchor, government and
philanthropic funding, open releases, a shared benchmark everyone reports on.
Their data-collection playbooks and licence choices are the ones to study
before we design the community recording drive.

---

## Masakhane (Africa)

**<https://www.masakhane.io>** · distributed, grassroots

**What it is.** A participatory research community for African NLP —
hundreds of contributors, many not formally affiliated with universities,
producing datasets, translation models, and benchmarks (MasakhaNER,
MasakhaNEWS) as a collective.

**Why it matters to us.** This is the "it takes a village" thesis with
receipts. Their participatory-research paper (see
[reading-list.md](reading-list.md)) is the strongest published argument that
native-speaker communities must be co-authors of the data rather than
subjects of it — which is the operating principle for our recording drive and
for the MOS listening panels. Also a useful governance model: how to credit
contributors, and how to keep data open without extracting from the people
who produced it.

---

## SEACrowd

**<https://huggingface.co/SEACrowd>** · Southeast Asian NLP collective

**What it is.** A community effort cataloguing and standardizing Southeast
Asian language datasets — a data hub plus a loader library that gives many
scattered corpora one interface.

**Why it matters to us.** The fastest route to finding Philippine-language
data we don't already have, across Tagalog, Cebuano, Ilocano, Hiligaynon, and
the rest. Their catalogue is the first place to look during the public-corpus
sweep in the SOTA plan. *We are already a member org* — contributing our
published corpora back is low-effort and raises their visibility.

---

## Philippine institutions and communities

- **UP Diliman — Digital Signal Processing Laboratory.** The source of both
  the Filipino Speech Corpus and the Philippine Language Dataset, i.e. the
  foundation of everything we train on. The natural first partner for
  licensing questions, new collection, and native-speaker evaluation.
- **DOST-ASTI** — the government S&T agency with a track record in Filipino
  speech and language resources, and the likely route to public funding or
  compute. **[unverified]** current programme status.
- **[philippineaireport](https://huggingface.co/philippineaireport)** — the
  initiative Sapin-sapin grew out of; the ecosystem-mapping and policy side of
  the same effort.
- **[FiLLM](https://huggingface.co/FiLLM)**, **[PhLLM](https://huggingface.co/PhLLM)** —
  Philippine LLM community orgs we're already part of. Relevant for the text
  side (BantayWika, halohalo) and for finding collaborators who care about
  Filipino evaluation.

---

## Industrial labs producing usable open artifacts

Not communities to join, but the source of most of our starting checkpoints.

- **Meta AI (FAIR) speech** — MMS (1,000+ language ASR/TTS), wav2vec2/XLS-R,
  SeamlessM4T, NLLB. Almost every baseline in our SOTA plan is theirs, and
  MMS forced alignment is already in our pipeline.
- **Google / Google Research** — FLEURS and the crowdsourced SEA speech
  corpora on OpenSLR; the "crowdsource high-quality read speech in-country"
  model we'd emulate for a recording drive.
- **Canopy Labs** — Orpheus, the codec-LM TTS our single-GPU pilot builds on.
- **Alibaba / FunAudioLLM** — CosyVoice family; a leading open multilingual
  codec-LM TTS and a candidate base for the cluster run.

**Caveat for all of these:** open weights with restrictive or ambiguous
licences are a real risk for anything we redistribute. Check the licence
before a checkpoint becomes load-bearing — noted as a gating item in the SOTA
plan.
