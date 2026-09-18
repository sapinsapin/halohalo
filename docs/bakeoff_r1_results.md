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

Everything that learned is on the Hub under `sapinsapin/`, tagged
`cc-by-nc-4.0` — PLD is CC-BY-NC and research-only, so the weights inherit that
regardless of the base model's licence:

- `whisper-large-v3-pld-ceb`, `whisper-large-v3-pld-pam`
- `omniASR_W2V_1B_SSL-ctc-char-pld_ceb`, `-pld_pam`
- `omniASR_W2V_1B_SSL-ctc-syllable-pld_ceb`, `-pld_pam`

The collapsed w2v-bert arms are deliberately not published
(`scripts/push_bakeoff.py --max-cer`).

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

## Next

- Fix and re-run the four w2v-bert arms (R1 is incomplete without them).
- Whisper's fixed 30 s mel wastes most of its compute on PLD's short prompts;
  `group_by_length` on the CTC arms and a shorter Whisper context are untested.
- The remaining eight languages. With the dataset cache and the tuned settings,
  a language costs roughly 2 GPU-hours for the Whisper arm.
