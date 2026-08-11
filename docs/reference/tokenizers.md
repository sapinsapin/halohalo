# Tokenizers for Philippine languages

Whether pretrained tokenizers fit our languages, what it costs when they
don't, and what we can actually do about it with the data we have.

All numbers below were measured on PLD transcripts with
[`../../scripts/tokenizer_report.py`](../../scripts/tokenizer_report.py) —
reproduce before trusting them.

---

## TL;DR

There are **two separate problems**, and they need different fixes:

| | Problem | Severity | Fix |
|---|---|---|---|
| **A** | Whisper has a `<\|tl\|>` language token and **no token for the other nine PLD languages** | Low for per-language models, **high** for any multilingual model | Add language tokens (9 new embeddings) |
| **B** | The BPE vocabulary fits Philippine languages badly — **1.75–1.92× English fertility**, and it destroys stem identity | Affects **every** Philippine language, Tagalog included | Vocabulary extension (~8k tokens → **−34%** tokens/word) |

Morfessor is **relevant but not a drop-in tokenizer**. It's a good source of
units for problem B; it does nothing for problem A; and it structurally cannot
represent Tagalog infixation. Details below.

---

## Problem A: the language-token gap

Whisper ships 112 language codes. Of the ten languages in PLD:

```
tagalog       tl
filipino      ABSENT      cebuano       ABSENT      ilocano       ABSENT
waray         ABSENT      hiligaynon    ABSENT      bikol         ABSENT
kapampangan   ABSENT      pangasinan    ABSENT      tausug        ABSENT
```

So `run_pld_asr.sh` currently forces Bikol, Cebuano, Waray, Tausug and the
rest through `<|tl|>`, i.e. tells the model "this is Tagalog" when it isn't.

**How much does this actually matter?** Less than it looks, *for what we
built*. We train one model per language, so the language token is a constant
prefix within each model — the decoder can simply relearn what it conditions
on. It is not free (we're fighting a pretrained prior that says "expect
Tagalog"), but it is not fatal either.

It becomes a real problem the moment we want **one multilingual Philippine
model**, which is the direction the SOTA plan points. There the token is the
only signal telling the model which language it's decoding, and nine languages
sharing `<|tl|>` means it cannot condition on language at all.

**Fix:** add new special tokens (`<|ceb|>`, `<|ilo|>`, …) and initialize their
embeddings from `<|tl|>` — a sensible prior, since Tagalog is the nearest
relative already in the model. Nine new rows in the embedding matrix; cheap
and low-risk. Do it before the multilingual run, not before the per-language
ones.

---

## Problem B: subword fit

Fertility — BPE tokens spent per whitespace word — measured over PLD
transcripts with `whisper-small`'s tokenizer:

| lang | words | tok/word | chars/tok | vs English |
|---|---|---|---|---|
| eng | 138,042 | **1.099** | 4.25 | — |
| ceb | 86,127 | 2.115 | 2.39 | **1.92×** |
| ilo | 173,005 | 2.051 | 2.47 | 1.87× |
| **fil** | 118,506 | **2.025** | 2.51 | **1.84×** |
| bcl | 126,049 | 1.996 | 2.52 | 1.82× |
| pag | 34,326 | 1.985 | 2.57 | 1.81× |
| hil | 117,918 | 1.971 | 2.47 | 1.79× |
| tsg | 23,405 | 1.943 | 2.48 | 1.77× |
| pam | 103,516 | 1.935 | 2.63 | 1.76× |
| war | 127,247 | 1.918 | 2.55 | 1.74× |

**The headline finding is the `fil` row.** Tagalog is the one Philippine
language Whisper nominally supports — and it still costs 1.84× English.
A language token is not the same as vocabulary coverage. Every Philippine
language pays roughly the same penalty whether or not it's "supported".

Why this hurts ASR specifically: for Whisper the tokenizer defines what the
decoder must **generate**. Double the tokens per word means double the
autoregressive steps, double the exposure to error propagation, and slower
inference. (For TTS it matters far less — in Orpheus, text is *input*
conditioning against ~87 audio tokens per second of speech, so text fertility
is a rounding error. **The tokenizer problem is an ASR problem first.**)

### The subtler damage: stem identity

Fertility understates it. Look at what BPE does to one Tagalog stem:

```
sumulat          _sum|ul|at
sinulat          _sin|ul|at
magsusulat       _mag|s|us|ul|at
kasulatan        _kas|ul|atan
pinagsusulatan   _pin|ags|us|ul|atan
```

Every one of these words contains the stem *sulat* ("write"). The tokenizer
splits it differently in each — `ul|at`, `ul|atan`, and boundaries that cut
across the stem/affix line (`_kas|ul`, `_pin|ags`). The model cannot see that
these are the same word family; it has to learn each surface form more or less
independently. In a language that builds most of its vocabulary this way,
that's a large, silent tax on sample efficiency — exactly what you can't
afford at 448 hours.

---

## Morfessor — summary and relevance

**<https://github.com/aalto-speech/morfessor>** · BSD-2-Clause · pure Python

**What it is.** Morfessor 2.0 is an unsupervised (optionally semi-supervised)
morphological segmenter from Aalto. It takes a word list with frequency counts
and learns to split words into morph-like units using a **Minimum Description
Length** objective: it balances the cost of storing the lexicon against the
cost of coding the corpus with it. No annotated morphology required. Family
members: **Baseline** (the MDL segmenter, what we tested), **Categories-MAP**
and **FlatCat** (HMM variants that also label morphs as prefix/stem/suffix,
useful when you want the categories and not just boundaries). Ships a CLI
(`morfessor-train`, `morfessor-segment`) and a Python API; cite Virpioja,
Smit, Grönroos & Kurimo (2013) for the implementation.

Installation is `pip install morfessor` — trivial, no dependencies to fight.

### Does it actually help? A qualified yes

Trained Morfessor Baseline on PLD word lists per language. **The naive
comparison is a trap and I nearly reported it:** in-domain it gives ~1.01–1.05
morphs/word versus BPE's ~1.9–2.1, which looks like a rout. It isn't — with a
small corpus the MDL objective keeps ~77% of word *types* whole (fil: 9,009
constructions for 11,770 types). It had largely memorised the vocabulary, and
any tokenizer with a language-specific whole-word lexicon "wins" that way.

The fair test is held-out words. Train on 80% of utterances, measure on the
remaining 20%:

| lang | OOV rate | morph/word | bpe/word |
|---|---|---|---|
| fil | 17.9% | **1.178** | 1.895 |
| ceb | 14.5% | **1.261** | 2.020 |
| ilo | 1.5% | 1.067 | 2.032 |
| war | 0.5% | 1.031 | 1.918 |

It holds up. Even with 18% of Tagalog tokens unseen in training, morph
segmentation costs ~1.18 units/word against BPE's ~1.90. Morphological
segmentation genuinely generalizes here rather than merely memorising.

### The limitation that matters: infixation

Morfessor is a **concatenative** segmenter — it can only cut words into
consecutive pieces. Philippine morphology is not purely concatenative:

```
sumulat    →  Morfessor: sumulat        (unsegmented)
sinulat    →  Morfessor: sinulat        (unsegmented)
```

Both are *sulat* with an infix (`-um-`, `-in-`) inserted **inside** the stem.
Representing that requires discontinuous morphology, which Morfessor cannot
express at any hyperparameter setting. It handled the concatenative cases well
— `mag+susulat`, `pinag+susulat+an`, `ka+sulat+an` (ceb), `nag+sulat` (ceb),
`ka+gandahan` — and produced some plain errors too (`g+isulat` should be
`gi+sulat`; `kina+b+uk+la+n` is noise). So: real signal on prefixes, suffixes
and circumfixes; blind to infixes; noisy on rare words.

Since `-um-` and `-in-` are among the most common Tagalog verbal affixes, this
caps the benefit. It doesn't eliminate it.

### Verdict

**Relevant — as a source of subword units and as an analysis tool, not as a
tokenizer swap.** Concretely:

- ✅ Good candidate generator for vocabulary extension (below), because its
  units generalize to unseen words better than whole-word frequency lists.
- ✅ Genuinely useful for the **text** side — BantayWika/halohalo LLM
  tokenizer design, where we control the vocabulary from scratch.
- ✅ Cheap to try: BSD-2, pip-installable, trains on our word lists in
  minutes on CPU.
- ❌ Not a replacement for Whisper's tokenizer — swapping the vocabulary means
  retraining the decoder's embedding *and* output projection, which needs far
  more data than we have.
- ❌ Won't capture infixation; don't expect it to solve the *sulat* family.

---

## Adapting the tokenizer we have

The realistic intervention is **vocabulary extension**: keep Whisper's
tokenizer and add Philippine units as new tokens, initializing each new
embedding as the mean of the subtokens it replaces (a standard, well-behaved
initialization).

Measured benefit, adding most-frequent whole words pooled across all nine
Philippine languages:

| tokens added | tok/word | reduction | token coverage |
|---|---|---|---|
| 0 | 1.974 | — | — |
| 1,000 | 1.669 | 15.4% | 58.3% |
| 2,000 | 1.559 | 21.0% | 67.1% |
| 4,000 | 1.436 | 27.2% | 76.4% |
| **8,000** | **1.294** | **34.4%** | **85.0%** |
| 16,000 | 1.156 | 41.4% | 92.5% |
| 32,000 | 1.040 | 47.3% | 98.2% |

**~8,000 added tokens cuts decoder sequence length by a third** — bringing
Philippine fertility from 1.97 down to 1.29, within reach of English's 1.10.
Whisper-small's embedding matrix grows by 8k × 768 ≈ 6M parameters, about 2.5%
of the model. That is a very good trade.

Diminishing returns past ~16k, and every added token is an embedding that
starts untrained, so more is not automatically better — the tokens must earn
their place from data we actually have.

Two caveats on that table. It is an **upper bound**: it assumes each new
embedding learns its token perfectly, which they won't. And it is
**sample-size sensitive** — run on a 3k-utterance-per-language slice instead
of 20k, the same curve reads 40% reduction at 8k tokens because the corpus
vocabulary is smaller. Measure on the corpus you intend to train on.

### Recommended sequence

1. **Measure first, on the corpus you'll train on.** The script is in the
   repo; PLD and FSC give different answers than web text.
2. **Extend with a hybrid unit list** — frequent whole words (best fertility
   per token) plus Morfessor morphs (best generalization to unseen forms).
   Ablate the mix; don't assume.
3. **Initialize new embeddings as the mean of their subtokens**, and warm up
   with the rest of the model frozen for a few hundred steps so the new rows
   land somewhere sensible before they perturb the pretrained space.
4. **Add language tokens** at the same time, initialized from `<|tl|>` —
   before any multilingual training, and only then.
5. **Validate on CER, not just WER**, and confirm the win survives on held-out
   *speakers*, not just held-out utterances.

### What not to do

- Don't replace Whisper's tokenizer wholesale. The decoder's output layer is
  tied to that vocabulary; retraining it needs orders of magnitude more data
  than 448 hours.
- Don't extend the vocabulary for **TTS** expecting much — text is input
  conditioning there, and the win is small. Spend the effort on ASR.
- Don't tune vocabulary size on the test set. Freeze splits first.

---

## Open questions

- Does the fertility win translate to a **CER** win? Reduced sequence length
  should help, but new untrained embeddings can hurt early. **[unverified]** —
  needs the ablation, and it's cheap enough to run on one GPU.
- Would **FlatCat**'s morph categories (prefix/stem/suffix) beat Baseline as a
  unit source? Untested.
- Is a **shared** Philippine vocabulary extension better than per-language
  ones? Shared should win on the related languages (much affix vocabulary is
  cognate) and is the only option for a multilingual model.
- How do these numbers look on **spontaneous Taglish** (livestream) rather
  than read PLD prompts? Code-switching means English tokens interleave, which
  may lower measured fertility while making segmentation less consistent.

## Reproducing

```bash
python scripts/tokenizer_report.py --corpus pld
python scripts/tokenizer_report.py --corpus pld --morfessor --held-out
python scripts/tokenizer_report.py --corpus pld --extension-curve
```
