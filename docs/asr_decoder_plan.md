# A better decoder for the Omnilingual encoders

Written 2026-09-21, after omni-7B failed to beat omni-1B. Follows
docs/bakeoff_r1_results.md.

## 1. What the numbers already say

Frozen speaker- and prompt-disjoint splits, 5000 steps each:

| model | decoder | ceb CER | ceb WER | pam CER | pam WER |
|---|---|---|---|---|---|
| whisper-large-v3 (1.55B) | attention, BPE, pretrained | 16.38 | 36.80 | 5.83 | 23.16 |
| omni-1B | linear CTC, chars | 17.04 | 49.62 | 10.21 | 41.52 |
| omni-7B | linear CTC, chars | 17.02 | 49.46 | 9.48 | 38.61 |

Two things stand out, and they point the same way.

**Seven times the encoder moved nothing.** If the encoder were the limit, 7B
would show it. So the loss is downstream of the encoder.

**On Cebuano the CER is a tie and the WER is not.** Whisper and omni make the
same *number* of character errors (16.4 vs 17.0) but Whisper gets 13 more words
in a hundred right. That is the signature of a missing language model: a CTC
head emits each frame independently of the characters around it, so its errors
are sprinkled one per word — `maayo` → `maayu` — and each costs a whole word.
Whisper's decoder conditions on what it has already written, so its errors
cluster in fewer words. The acoustic evidence is about equal; the *spelling* is
not.

That is the cheapest kind of problem to have, because it can be attacked
without touching the 25 GB of weights already trained.

**One confound to clear first.** The 7B's dev CER was still falling when it
stopped — pam went 12.1 → 9.48 between steps 3000 and 5000. Some of the gap is
training budget, not decoder. Whisper ran the same 5000 steps, so the table is
fair, but "CTC has converged and lost" is not yet established.

## 2. The plan, cheapest first

Each step has a gate. Do not start the next because it is interesting; start it
because the previous one left a gap worth its cost.

### D0. Look at the errors — free, inference only

Before building anything, confirm the diagnosis on the models we have.

- Per-utterance hypotheses from omni-7B and whisper-large-v3 on the same test
  clips; align each against the reference.
- Split errors into: substitutions within a word, word-boundary errors
  (merged / split words), whole-word deletions, and insertions.
- **The number that decides D1:** of omni's wrong words, how many are within
  edit distance 1–2 of a word in the training vocabulary? That is the ceiling
  on what a lexicon or LM can repair. If it is most of them, D1 will pay; if
  the wrong words are acoustically far off, skip to D3.

Cost: minutes of GPU. **Gate:** ≥ half of word errors are near-misses.

### D1. N-gram LM fusion — CPU, no retraining

KenLM 4–5-gram over words, fused at beam search with `pyctcdecode`. Standard,
and it keeps CTC's streaming property, which is why the plan wanted CTC at all.

- LM text: PLD **train-split** transcripts plus the org's per-language text
  (the FineWeb-2 ingests, T1). Tune α (LM weight) and β (word bonus) on a slice
  of train, never on test.
- **Leak check, not optional.** PLD prompts are read sentences that may exist
  on the web. The frozen split is prompt-disjoint so train transcripts are
  clean by construction, but external text is not: drop any external sentence
  sharing a long n-gram with a test prompt, and report how many were dropped.
  An LM that has read the test set turns WER into a memory test — the same
  mistake as the contaminated fleet re-measurement, in a new place.
- Report every model **with and without** the LM, so the acoustic model and
  the LM are never confused in a table.

Cost: ~$0, an afternoon. **Gate:** ceb WER from 49 to ≤ 40. If it lands near
Whisper's 36.8, the "better decoder" may simply be this, and D3–D4 become
optional.

### D2. Train the CTC properly — ~$5

Only once D1 shows what the LM alone buys.

- 10k–15k steps for omni-1B (not 7B: same accuracy, a seventh of the cost),
  to see where the curve actually flattens.
- Intermediate CTC loss at a middle layer (InterCTC). Nearly free, consistently
  helps deep encoders, and targets exactly the conditional-independence
  weakness.
- A small subword vocabulary (unigram, 256–512) as an alternative to chars:
  units that span a syllable carry some of the context a char head lacks.
  The earlier syllable arm lost, but that was a 1.4k-unit linguistic inventory;
  a data-driven 256 is a different object.

**Gate:** beats omni-1B-char + LM from D1 by ≥ 1 CER point.

### D3. An attention decoder on the encoder — ~$10

Joint CTC/attention, the ESPnet recipe: keep the CTC head, add a 6-layer
transformer decoder over the same encoder, train with λ·CTC + (1−λ)·attention,
decode with both scores. The attention branch learns the spelling the LM in D1
supplies from outside; the CTC branch keeps alignments monotonic and stops the
hallucination that makes attention-only models dangerous.

- Risk: a decoder trained from scratch on 25k clips (≈ 30 h) has little text to
  learn a language from. Mitigate by pretraining the decoder as a text LM on
  the same corpus as D1's KenLM before attaching it.
- Loses streaming. Acceptable for research; say so on the card.

**Gate:** beats D1's best on WER without losing CER.

### D4. An LLM as the decoder — the open question, ~$15+

Meta's own answer to this problem: the Omnilingual ASR release ships LLM-ASR
variants — the same wav2vec2 encoders feeding a Llama-style decoder — and
reports them ahead of their CTC siblings. Two routes:

- **Use theirs.** Zero-shot first, as a baseline row; it covers nine of our ten
  languages. The catch is known: their checkpoints need fairseq2, which this
  repo has deliberately avoided. One contained venv, as with qwen-tts.
- **Build ours.** Frozen omni-1B encoder → a small projector → a LoRA-tuned
  LLM (the SLAM-ASR shape). Cheap to train because only the projector and
  adapter learn. Which LLM matters: it needs to have seen Cebuano and
  Kapampangan text, which most have barely.

This is the step most likely to beat Whisper and the least certain to. It only
makes sense if D1–D3 leave a gap, and it does not fit the remaining ~$30 of
credit alongside the TTS work.

## 2b. Results so far (2026-09-22)

**D0 changed the diagnosis before D1 ran.** Scoring the *same* hypotheses with
stress accents and punctuation stripped from both sides:

| Cebuano, 2781 clips | as scored, CER / WER | normalised, CER / WER |
|---|---|---|
| whisper-large-v3 | 16.39 / 36.87 | 12.15 / 24.20 |
| omni-1B | 20.11 / 51.51 | 16.41 / 39.51 |

Twelve WER points in *both* models were orthography: PLD marks stress on about
a third of words and an ASR model is not asked for it. So section 1's reading
of the CER-tie/WER-gap as "a missing language model" was partly wrong — but only
partly, because the gap between the two models survives normalisation intact
(24 vs 40). The decoder question stands; the ruler was bent.

Acted on: `halolib.finetune.normalise_text`, `--normalise` in both trainers,
and all four models continued 1500 steps on normalised labels
(`scripts/run_normalised.sh`; published as `*-norm`). Whisper ceb 10.77 / 22.53,
pam 5.08 / 19.80; omni-1B ceb 12.55 / 34.80, pam 8.46 / 35.39. Most of that
gain was the fairer scoring, not the retraining.

**D0 on the errors themselves** (omni-1B, un-normalised scoring):

| | ceb | pam |
|---|---|---|
| word errors that are substitutions | 77% | — |
| substitutions within edit distance 2 of the right word | 45% | 54% |
| … and the right word appears in the training text | 20% | 35% |
| test words never seen in training text | 32% | 21% |

The last row is the split working as designed — no test prompt appears in
training — and it caps what a train-text LM can do.

**D1, train-text 4-gram KenLM, alpha/beta tuned on 200 held-out clips:**

| omni-1B greedy → + LM | CER | WER |
|---|---|---|
| ceb | 19.98 → 19.14 | 51.26 → **46.77** (−4.5) |
| pam | 11.43 → 10.56 | 43.15 → **36.49** (−6.7) |

Real, free, streaming-safe, and exactly where D0 said it would land: pam gains
more because more of its errors were reachable. The gate was "ceb WER ≤ 40";
this LM alone does not clear it. The next step within D1 is an external-text LM
(the FineWeb-2 ingests) with the test-prompt overlap check, which is where the
remaining headroom is — the current LM has 4,864 distinct pam sentences to
learn from.

**D2, first arm: negative.** omni-1B at 15000 steps scored 16.90 CER against
17.04 at 5000, with dev CER flat and noisy over the last 6000. Training budget
is not the gap. InterCTC and a small subword vocabulary remain untested.

**Where that leaves D3/D4.** After normalisation and the LM, omni-1B trails
Whisper by ~12 WER points on both languages under like-for-like scoring. That
is the decoder's share, and it is what a joint CTC/attention decoder (D3) would
have to earn. Not funded within the current credit.

## 3. What would count as success

- **Primary:** beat whisper-large-v3 on ceb WER (36.8) with a permissively
  licensed stack. CER is already tied; WER is where the decoder shows.
- **Secondary:** pam CER under 8 — 9.48 today against Whisper's 5.83.
- Reported on the frozen splits, with and without any external LM, with the
  LM's text provenance and the leak check stated beside the number.

## 4. What this plan does not do

It does not revisit w2v-BERT, whose collapse in every arm is still undiagnosed
and is a separate bug. And it does not retrain the 7B: two languages agree that
it buys nothing over the 1B under a linear head, so every experiment here runs
on the 1B until a decoder exists that could use more encoder.
