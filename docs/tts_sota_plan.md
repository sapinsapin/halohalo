# TTS on the RTX PRO 6000: Orpheus-3B and Qwen3-TTS

Written 2026-09-18, after the R1 ASR bake-off. Covers plan items S2, S3, T6
and R7, and the two missing trainers §8 lists as engineering debt.

The bake-off's lesson was not about models: **the trainers were still sized
for an 8 GB card and used 13 GB of 96.** Whisper ran batch 2 × accum 8 with
gradient checkpointing; fixing that alone moved tensor-core activity from 0.19
to 0.46. `finetune_orpheus.py` has the same problem in a worse form, so the
configuration section below comes before the experiment plan.

## 1. Why these two models

| | Orpheus-3B | Qwen3-TTS-12Hz-1.7B-Base |
|---|---|---|
| what it is | Llama-3.2-3B that emits SNAC 24 kHz codec tokens | 1.7B codec LM at 12 Hz |
| licence | Apache-2.0 weights, **Llama 3.2 base terms** | **Apache-2.0, no base-model chain** |
| our languages | none pretrained; Llama BPE covers Latin script | none pretrained |
| precedent | QLoRA proven in this repo; Sunbird's 20-language finetune of the same checkpoint | official Apache-2.0 SFT ships; per-sample speaker conditioning built in (see P3) |
| track | research now; commercial only if the Llama terms are accepted | **the commercial candidate** |

They are complementary, not redundant: Orpheus is the one with a multilingual
finetuning precedent, Qwen3-TTS is the one whose licence is clean. Running both
answers "how good can we get" and "what can we ship" in one sweep, and the
comparison is fair because both are codec LMs fed by the same frontend.

Not in scope: OmniVoice (CC-BY-NC, research ceiling only — its `fil` 7.7 h and
`ceb` 12.2 h make it a strong zero-shot **baseline row**, S4), and Higgs/Fish
(non-commercial reference rows).

## 2. Hardware configuration — the part we got wrong last time

One RTX PRO 6000 Blackwell: **96 GB, bf16, sm_120**. Preemptible ≈ $1.08/h in
uk-south2.

### What has to change in `finetune_orpheus.py`

| | now (8 GB card) | on 96 GB |
|---|---|---|
| base weights | 4-bit NF4 quantised | **bf16, unquantised** |
| trainable | LoRA r=16 on projections | LoRA r=64, or `--full` |
| batch × accum | 1 × 16 | **8 × 1** (16 does not fit — measured) |
| gradient checkpointing | on | **off** |
| optimiser | `paged_adamw_8bit` | `adamw_torch_fused` (full) / fused (LoRA) |
| SNAC tokens | re-encoded every run, librosa resampling | **cached to disk, polyphase resampling (40× faster)** |
| batching | padded to longest in batch | length-grouped |

4-bit quantisation is not free accuracy-wise — it is a memory compromise we no
longer need, and it makes every matmul slower than bf16 on a card with tensor
cores to spare.

**Measured on 2026-09-19** (`--profile`, 6 steps, Cebuano, 9,270 examples):

| | |
|---|---|
| LoRA r=64, bf16 base, batch 8, no checkpointing | **72.1 GiB of 95 GiB** |
| GPU/CPU time ratio | 1.83 |
| dominant kernels | bf16 tensor-core GEMMs (cutlass) |

That is far more than a 97M-parameter adapter should need, and the reason is
the **156,939-token vocabulary**: the logits tensor is batch x sequence x 157k,
several GiB in bf16 before the loss upcasts it. The estimate this section used
to carry — 53 GiB for a full finetune, leaving room for batch 8 — ignored that
and was wrong in the direction that matters.

Consequences, replacing the earlier plan:

- **LoRA at batch 8 is already near the card's limit.** Do not raise the batch;
  raising `--max-tokens` costs memory quadratically through attention and
  linearly through the logits.
- **A full 3.3B finetune does not fit at batch 8.** Its optimiser state alone
  is ~40 GiB on top of what is already resident. If the `--full` vs LoRA
  comparison is worth running, it needs batch 2, `adamw_bnb_8bit`, or
  gradient checkpointing back on — i.e. it is no longer free, and P2 should
  treat it as a separate costed experiment rather than the default.
- Qwen3-TTS-1.7B should be far cheaper per step regardless: its codec runs at
  12.5 Hz against SNAC's 87.5 tokens/second, and its vocabulary is a fraction
  of Orpheus's.


### The preprocessing debt is the real cost

SNAC encoding currently runs inside every training job, one clip at a time, and
is lost on preemption. For 25k clips that is tens of minutes of GPU doing
codec inference instead of training — the same failure as the ASR arm's
300k-row filter, which cost 10–30 idle minutes per run.

Fix once: encode in batches, cache to
`$PLD_WORK_DIR/snac_cache/{dataset}_{language}/`, reuse across every arm,
language and rerun. Ten languages × several arms each makes this the
highest-leverage change in this document.

## 3. Experiments

Phase ordering follows the plan's gates: the frontend decision (T6) must land
before any fleet run, or the fleet bakes in an unmeasured choice.

### P1. Frontend ablation (T6) — Cebuano and Kapampangan

The R2 ASR result (characters beat syllables, 17.0 vs 22.8 CER on ceb) does
**not** transfer automatically: a codec LM inherits a text tokenizer's priors,
which is a different mechanism from a CTC head's output alphabet. Measure it.

3 frontends (BPE / syllable / char) × 2 languages × Orpheus, 2k steps each.
Judge: round-trip CER via `whisper-large-v3-pld-{lang}` — **the new bake-off
winners, not the whisper-small fleet** — plus a listen test. Decision recorded
in this file before P2 starts. **~6 GPU-h.**

### P2. Orpheus per language, winning frontend — all ten + FSC

Full finetune where it fits, LoRA r=64 otherwise; the choice is itself measured
on ceb first (`--full` vs LoRA at equal steps), because a 3.3B full finetune on
20k clips may simply overfit. ~20k clips, 2 epochs, speaker-disjoint eval.
**~25 GPU-h** for the fleet.

### P3. Qwen3-TTS per language

**Corrected 2026-09-18 after reading the upstream source.** The plan said "the
trainer does not exist; official SFT is single-speaker only". Half of that was
wrong, and it makes this item much cheaper. `QwenLM/Qwen3-TTS` ships
`finetuning/{prepare_data,dataset,sft_12hz}.py` under Apache-2.0, and:

- **Training is already per-sample multi-speaker.** Each JSONL row carries its
  own `ref_audio`; the loop runs `model.speaker_encoder(ref_mels)` per batch
  and writes the result into the codec embedding at position 6. Nothing about
  the forward pass is single-speaker.
- **Only the *saving* step is single-speaker**: it takes the first batch's
  embedding, bakes it into `codec_embedding.weight[3000]`, and flips
  `tts_model_type` to `custom_voice` with one `spk_id`. To stay multi-speaker
  we keep the checkpoint in `base` mode, where a voice is supplied at
  inference as 3 seconds of reference audio — which is also what we want for
  the 980 PLD speakers, since it bakes no one's voice into the weights.
- `prepare_data.py` already batch-encodes audio to codes (32 at a time) into
  the JSONL, so the codec-caching work is done upstream.

What we write is therefore an **exporter**, not a trainer: PLD/FSC →
`{audio, text, language, ref_audio}` JSONL, where `ref_audio` is a different
clip from the same speaker, plus a thin wrapper that keeps our conventions
(frozen splits, run dirs, W&B, `--resume`). Vendor the three upstream files
with attribution; they are Apache-2.0. **~0.5 engineer-day + ~20 GPU-h.**

**Sequence length is the pleasant surprise.** Qwen's codec runs at 12.5 Hz
against SNAC's 87.5 tokens/second, so a 10-second clip is ~125 frames instead
of ~875. Qwen3-TTS sequences are roughly 7× shorter than Orpheus's, which means
much larger batches at the same memory and a proportionally cheaper fleet.

Hardware: upstream's loop is `batch_size=2`, grad-accum 4, bf16, FlashAttention
2 — an accessible-hardware default, the same trap as the 8 GB recipes. At 1.7B
with 125-frame sequences the card should take **batch 32–64**; confirm with
`--profile` on the first run rather than assuming.

### P4. Multilingual adapter (S3)

One model, ten languages, language tags; compared against the per-language
models on pag and tsg, the two thinnest. Answers whether the fleet can collapse
to one artefact. **~15 GPU-h.**

### P5. Baselines and judging

- OmniVoice zero-shot on fil and ceb; Higgs TTS 3 on tgl and ceb (reference rows).
- **Independent judge:** Omnilingual ASR, not our own whisper models. The
  round-trip CER we have been quoting is circular — our judge was trained on
  the same corpus as the models it scores, and doubly so if it ever becomes an
  RL reward.
- MMS-TTS is the bar in the eight languages it covers (CER 2.6–17.8); Tausug
  has no bar at all, which makes it the language where a listen test is not
  optional.

**Total ≈ 70 GPU-h ≈ $75 preemptible**, which does not fit the remaining ~$84
credit alongside the ASR work. P1 and P3-on-ceb are the part to fund first:
they produce the two decisions everything else depends on, for about $12.

Cheaper than it looks in one respect: the SNAC and Qwen codec caches are built
once per language and reused by every arm, so the marginal cost of an extra
ablation is training time only.

## 4. Success criteria

Per language, in order:

1. **Intelligible**: round-trip CER below MMS-TTS's number for that language,
   measured by the independent judge. Today's SpeechT5 fleet fails this in
   eight of ten (CER 27–284%).
2. **Speaker fidelity**: ECAPA similarity at or above SpeechT5's 0.30–0.53,
   which is the one thing the current fleet does well.
3. **Native-listener check** before any publication claim: the judge floor on
   ceb and pam is already 7.3% CER on *human* recordings, so CER alone cannot
   separate good from excellent there.

Licence on everything published: **cc-by-nc-4.0**, research-only, because PLD
is. That is independent of Orpheus's or Qwen's own licence.

## 5. Open decisions

- **Llama 3.2 terms for Orpheus** (naming requirement + AUP) — needed before
  Orpheus can enter the commercial track at all. Unowned; §10 of the SOTA plan.
- **Voice-cloning consent** for the 980 identifiable PLD speakers, and whether
  cloning ships off by default. Gate G5, also unowned.
- **Watermarking** for published TTS. Undecided.
