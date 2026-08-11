# Reading list

Grouped by what you'd read it *for*. Citations are title + year so they're
searchable; author lists are trimmed to first author where given.

---

## Framing — why this work exists

- **The State and Fate of Linguistic Diversity and Inclusion in the NLP
  World** (Joshi et al., ACL 2020). Classifies the world's languages 0–5 by
  available resources and shows how badly the distribution is skewed. Tagalog
  sits mid-table; most Philippine languages sit at the bottom. The standard
  citation for "why bother", and useful language for funding conversations.
- **On Achieving and Evaluating Language-Independence in NLP** (Bender, 2011).
  Source of the "Bender Rule": *name the language you worked on*. Sounds
  trivial; the field routinely fails it, and it's why our cards state the
  language and variety explicitly.
- **Participatory Research for Low-resourced Machine Translation: A Case
  Study in African Languages** (∀ / Masakhane et al., Findings of EMNLP 2020).
  The argument that speaker communities should be co-authors of the data, with
  evidence that it produces better results. The intellectual backing for the
  community recording drive.
- **Data Statements for NLP** (Bender & Friedman, TACL 2018) and **Datasheets
  for Datasets** (Gebru et al., 2018/2021). What a dataset card owes its
  users. Worth re-reading before we publish the next corpus.

## Multilingual speech foundations

- **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
  Representations** (Baevski et al., NeurIPS 2020). The self-supervised
  pretraining result that made low-resource ASR practical.
- **XLS-R: Self-supervised Cross-lingual Speech Representation Learning at
  Scale** (Babu et al., 2021). Cross-lingual transfer at 128 languages —
  the empirical case for starting multilingual.
- **Scaling Speech Technology to 1,000+ Languages** (Pratap et al., 2023).
  MMS: ASR, TTS, and forced alignment for a thousand languages, including
  Philippine ones. We already use its aligner; read it before extending our
  alignment stage.
- **Robust Speech Recognition via Large-Scale Weak Supervision** (Radford et
  al., 2022). Whisper. The weak-supervision-at-scale argument, and the model
  underneath our pseudo-labeling.

## Low-resource technique

- **Morfessor** (Creutz & Lagus, 2007; Morfessor 2.0, Virpioja et al., 2013).
  Unsupervised morphological segmentation — the subword-unit question for
  agglutinative languages.
- **Self-training and Pre-training are Complementary for Speech Recognition**
  (Xu et al., 2020). Why pseudo-labeling *and* SSL pretraining stack rather
  than substitute.
- Aalto's Finnish and Northern Sami papers (see
  [research-groups.md](research-groups.md)) — the closest published analogue
  to our situation, especially on colloquial-vs-standard speech and
  long-audio alignment.

## TTS

- **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers**
  (Wang et al., 2023). VALL-E — the codec-LM formulation the whole current
  generation of expressive TTS descends from, including Orpheus.
- **SoundStream** (Zeghidour et al., 2021) and **High Fidelity Neural Audio
  Compression** (Défossez et al., 2022, EnCodec). The neural codecs that make
  discrete speech tokens work; background for SNAC's residual/multi-scale
  structure.
- **Flow-matching TTS** (F5-TTS and the non-autoregressive line, 2024–).
  Our robustness fallback.

## Speech translation and S2S

- **No Language Left Behind** (NLLB Team, 2022). 200-language MT; the
  translation engine for manufacturing pseudo-parallel S2S data.
- **SeamlessM4T** (Seamless Communication, 2023) and the expressive/streaming
  follow-ups. The base model for our S2S Track A, and the baseline we have to
  beat on fil↔en.
- **CVSS Corpus and Massively Multilingual Speech-to-Speech Translation**
  (Jia et al., LREC 2022). The recipe for building S2ST data by translating
  transcripts and synthesizing targets — exactly what our plan does.
- **AudioPaLM: A Large Language Model That Can Speak and Listen** (Rubenstein
  et al., 2023). The unified speech-text LLM pattern behind Track B.

## Evaluation

- **FLEURS: Few-shot Learning Evaluation of Universal Representations of
  Speech** (Conneau et al., SLT 2022). Our cross-paper comparison point;
  read it to understand exactly what the `fil` split does and doesn't cover.
- **UTMOS** (Saeki et al., 2022) and the VoiceMOS Challenge papers.
  Automatic MOS prediction and — more usefully — its documented failure modes.
- **BLASER** (2022) and BLASER 2.0. Text-free speech translation evaluation,
  avoiding the ASR-error confound in ASR-BLEU.

## Efficiency

- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021).
- **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023).
  Together these are why a 3B TTS finetune fits on an 8GB card — the basis of
  [`../orpheus_pilot_plan.md`](../orpheus_pilot_plan.md).

---

## How to use this list

Don't read it front to back. Before starting a piece of work, read the two or
three entries that bear on it — and add what you learn back here. An entry
that turns out to be wrong or superseded is worth *correcting in place*, not
silently dropping.
