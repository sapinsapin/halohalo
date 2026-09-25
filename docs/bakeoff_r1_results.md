# R1 foundation bake-off + R2 output-unit ablation — results

Run 2026-09-17/18 on one preemptible NVIDIA RTX PRO 6000 Blackwell (96 GB) in
Nebius uk-south2, ~13 GPU-hours, about $15. Two languages: Cebuano (worst BPE
fertility) and Kapampangan (weakest published CER). 5000 steps, effective batch
16, 25k clips per language.

Every number below is on the **frozen speaker- and prompt-disjoint split**
(`splits/pld_*.json`). They are not comparable with the in-domain CERs on the
dataset card, which share speakers and prompts between train and test.

## Results

| candidate | units | licence | ceb CER% | ceb WER% | pam CER% | pam WER% |
|---|---|---|---|---|---|---|
| whisper-large-v3 | bpe | Apache-2.0 | **16.38** | 36.80 | **5.83** | 23.16 |
| omniASR_W2V_1B_SSL | char | Apache-2.0 | 17.04 | 49.62 | 10.21 | 41.52 |
| omniASR_W2V_1B_SSL | syllable | Apache-2.0 | 22.79 | 54.12 | 14.15 | 44.26 |
| w2v-bert-2.0 | char | MIT | 93.20 | 100.00 | 93.03 | 99.99 |
| w2v-bert-2.0 | syllable | MIT | 99.74 | 100.00 | — | — |

**R1 (foundation): whisper-large-v3 wins both languages**, and by a wide margin
on Kapampangan (5.83 vs 10.21). It is Apache-2.0, so the research and
commercial tracks take the same winner here — the licence fork the plan
anticipated does not bite.

**R2 (output units): characters beat syllables** on both languages with the
same encoder (17.04 vs 22.79 ceb; 10.21 vs 14.15 pam). The 1.4k-unit syllable
vocabulary loses to 57–65 characters. That is evidence against the syllable
hypothesis as applied here, not evidence for BPE: whisper's advantage confounds
units with a different encoder, decoder and pretraining mix.

**w2v-bert-2.0 collapsed in every arm** (CER 93–100%, WER 100%): it learned
nothing, identically in both languages. This is an unresolved setup bug — a
1.0 WER at 5000 steps is not a property of the languages. Do not read it as a
finding about the encoder. Suspects, in order: learning rate (the CTC arms
share 1e-4, tuned for the omni encoders), the feature extractor, and the
`Wav2Vec2BertForCTC` head over the SeamlessM4T front end. **The bake-off is
incomplete until this is fixed and those four arms are re-run.**

## Published models

All under `sapinsapin/`, tagged `cc-by-nc-4.0` — PLD is CC-BY-NC and
research-only, so the weights inherit that regardless of the base model's
licence. Every card opens with a plain-language section and says which model
to use instead where it is not the best one.

- `whisper-large-v3-pld-ceb`, `-pam` — the bake-off winners, scored with
  accents and punctuation.
- `omniASR_W2V_1B_SSL-ctc-char-pld_ceb`, `-pld_pam` — the CTC arm.
- `omniASR_W2V_7B_SSL-ctc-char-pld_ceb` — the 7B, published as a negative
  result (below).
- The `-norm` continuations and fleet (below).

The syllable CTC arms were published and then **deleted on 2026-09-22**: they
lost the R2 ablation and the numbers stay in this document. The collapsed
w2v-bert arms were never published.

## What the run cost, and what it taught about the machine

Two bugs and one waste, all fixed in the scripts:

1. **whisper-large-v3 crashed on the first attempt.** transformers 5 loads
   weights in the checkpoint's dtype and that checkpoint ships fp16, so fp16
   weights met fp32 log-mels in generation-based eval. Fixed by loading
   `dtype=torch.float32` and letting bf16 autocast do mixed precision.
2. **`finetune_asr.py` never passed `num_proc` to the loader**, so selecting
   one language meant three single-process filter passes over PLD's ~300k rows
   — 10–30 minutes of idle GPU per run. The filtered corpus is now cached per
   (task, language, split) under `$PLD_WORK_DIR/ds_cache`.
3. **The trainers were still sized for an 8 GB card.** Whisper ran at batch 2 ×
   accum 8 with gradient checkpointing on a 96 GB card.

Measured on the card (DCGM counters, not `nvidia-smi` utilisation, which only
says a kernel was resident):

| | before | after |
|---|---|---|
| GPU memory | 13 GB / 96 | 72–80 GB / 96 |
| power | ~380 W | ~550 W |
| tensor-core active | 0.19 | 0.46 |
| prep before first step | 10–30 min | ~2 min |
| whisper step rate | 1.33 it/s | 1.60 it/s |

`torch.profiler` on the tuned Whisper config (`--profile`, trace in
`finetune_runs_nebius/`): matmuls 44% of GPU time (`mm` 32%, `addmm` 12%),
`copy_` 18% across 22k calls (mixed-precision weight casts), flash-attention
backward 9%, elementwise and optimizer kernels ~16%. GPU/CPU time ratio 2.12,
so the pipeline is not starving the GPU. Attention was already flash-backed
SDPA before the "switch to SDPA" change — transformers picks it by default, and
that suggestion bought nothing.

## The frozen splits do not retrofit (found 2026-09-19)

A7 asks for the published fleet to be re-measured on the frozen splits so it can
sit beside the bake-off numbers. It cannot be done that way.

The fleet was trained before the splits existed, on the hub's random split.
Measured on Cebuano: the frozen spec holds out 28 speakers and 866 prompts, and
**20.3% of the hub train split's 51,205 Cebuano clips belong to them** — 10,042
clips from those 28 speakers alone. Scoring those checkpoints on the frozen test
set is scoring them on speakers they trained on.

It shows up exactly as you would expect. `whisper-small-pld-ceb` scores **10.75%
CER** on the frozen ceb test set against whisper-large-v3's **16.38%** on the
same split. A 244M model does not beat a 1.55B model on held-out speech; it wins
when the speech is not held out for it.

The honest fleet column therefore needs retraining on the frozen train splits,
which is what `scripts/retrain_fleet_frozen.sh` does — whisper-small, ten
languages, roughly 2.5 GPU-hours. Until that runs, the dataset card's numbers
and any re-measured ones are both in-domain in different ways, and neither
belongs beside the bake-off's.

## Follow-ups, 2026-09-21 to 09-23

**Scale did not help.** omni-7B CTC (fp32 weights, bitsandbytes 8-bit Adam, 74.5
GiB of 95) under the same recipe: ceb **17.02** CER / 49.46 WER, pam **9.48** /
38.61 — against the 1B's 17.04 and 10.21. Seven times the encoder moved
nothing on Cebuano. That pointed at the decoder; see docs/asr_decoder_plan.md.

**Twelve WER points were orthography.** PLD marks stress on about a third of
words and keeps punctuation. Scoring the same hypotheses with both removed
moved whisper-large-v3 on ceb from 36.9 to 24.2 WER and omni-1B from 51.5 to
39.5. `halolib.finetune.normalise_text` and `--normalise` in both trainers
make it a convention; the four bake-off models were continued 1500 steps on
normalised labels (`*-norm`, published). Most of the gain was the fairer
ruler, not the retraining. The gap between the two models survives it.

**whisper-large-v3 on normalised text, all ten languages** (5000 steps from the
base, same recipe; CER / WER %, frozen split):

| bcl | ceb | eng | fil | hil | ilo | pag | pam | tsg | war |
|---|---|---|---|---|---|---|---|---|---|
| 4.6 / 15.2 | 10.8 / 22.5 | **46.7 / 79.1** | 5.0 / 12.2 | 9.3 / 18.7 | 5.7 / 20.6 | 16.0 / 30.7 | 5.1 / 19.8 | 6.6 / 21.0 | 7.8 / 21.0 |

(ceb and pam are the 1500-step continuations.) Nine are published as
`whisper-large-v3-pld-<lang>-norm` and are the recommended ASR per language.
**English is a bug, not a result** and is not published: it is the only
language the trainer sends through `language="english"` rather than `<|tl|>`,
and that path had never been exercised with `--normalise`. Undiagnosed.

## Next

Moved to docs/status_2026-09-25.md, which keeps one list of everything open
across ASR and TTS and says which of it runs on the workstation's 8 GB card.
