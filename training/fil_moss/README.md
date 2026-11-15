# Authorized Filipino MOSS fine-tuning workflow

The user authorized starting Filipino TTS fine-tuning when data is ready (2026-11-15). No further launch confirmation is required within this scope. Current state: **authorized, pending eligible data**, not an active trainer or a scheduled watcher. Continue this work under the active dataset-curation goal. Data-readiness checking and waveform export are executable. Codec encoding, stage 1 (engineering smoke), stage 2 (bounded baseline SFT and held-out evaluation) and stage 3 (GRPO on the stage 2 baseline) are wired but not yet run on real weights or data (see "Stages 1–3" below).

`recipe.json` pins the existing MOSS-Realtime model, upstream code and acoustic codec. The initial baseline starts from the released pretrained model rather than the Hiligaynon-adapted checkpoint. Use FP32 parameters/gradients, BF16 autocast and bitsandbytes AdamW8bit, batch one and LR 1e-5. The existing GB10 experiment measured about 20.12 GiB peak training allocation; remeasure with the actual Filipino lengths and retain host memory guards. This quantizes optimizer state, not model weights. QLoRA is a distinct later memory experiment.

## Data readiness

Run on the host that holds the audio:

```sh
python -m curation.tts_readiness --manifest /absolute/path/manifest.jsonl \
  --recipe training/fil_moss/recipe.json --report /absolute/path/readiness.json
```

Exit 2 means data is not ready; exit 0 only passes the data gate. Preserve Tagalog-English code-switching and exact reviewed transcripts. The initial pilot thresholds (1 hour train, 6 minutes validation and 6 minutes test, 2–15-second clips) are engineering pilot choices, not sufficient evidence of production quality or replacements for the 250-hour dataset target. Require attributed audio review, training permission evidence, stable speaker IDs and frozen disjoint splits. Validate exact eligible audio hashes and reject duplicate audio before counting hours. Reject cross-split duplicate transcripts after Unicode NFKC, case-folding and whitespace normalization; this is an exact normalized-text check, not semantic deduplication. Never manufacture speaker IDs, permissions or audio approvals from LLM text review.

`readiness.json` audits the frozen 160-row partial remote Filipino manifest: zero eligible rows. Its counts are a dated snapshot, not a claim that every other project corpus was audited. Previously curated Filipino corpora may be considered only after checking their own evidence and frozen evaluation boundaries.

## Execution sequence once eligible

The waveform export runs on the audio host and requires an entirely eligible frozen manifest meeting the pilot thresholds:

```sh
python -m curation.tts_export --manifest /absolute/path/eligible.jsonl \
  --recipe training/fil_moss/recipe.json --output /absolute/path/new-fil-run/data
```

It rechecks the data gate and audio hashes, converts to mono 24 kHz PCM16, verifies decoded durations, rejects duplicates after conversion, and writes separate train/validation/test conversations with unchanged text and full original provenance. The output directory must be new. Failed exports remove their temporary directory; successful exports freeze the recipe, canonical manifest, derived hashes and FFmpeg version. This does not encode acoustic tokens, establish audio quality, or launch training.

1. Freeze the eligible manifest, split hashes, source rights evidence, package versions and model/codec revisions.
2. Use the isolated GB10 MOSS runtime at `/home/iotim/work/hil-moss-pilot` as a tested dependency source. Keep new Filipino data/results in a separate run directory. Do not overwrite the Hiligaynon experiments.
3. Export reviewed waveform/transcript pairs into the pinned upstream conversation schema, resample to codec format with tracked derivation, and precompute 16-codebook tokens. Verify reconstruction and prevent validation/test from appearing in training.
4. Acquire `/tmp/sapinsapin-gpu.lock`; preserve the existing 25 GiB available-host-memory guard. Run one-step forward/backward/AdamW8bit, verify FP32 gradients/BF16 compute, save and reload, and generate a comparison sample before scaling.
5. Run a bounded baseline with recorded seed/order/schedule, separate best-validation and last checkpoints, and durable progress. Determine the pilot update budget from the frozen dataset rather than repeating the earlier 32 Bible clips. Do not use held-out test results for checkpoint selection.
6. Evaluate held-out loss, intelligibility, native listening, speaker consistency and code-switch handling. Measure streaming first-audio latency and real-time factor separately; offline generation time is not a streaming benchmark. No production release follows merely from training completion.

## Stages 1–3 (2026-11-15)

Stages 1 and 2 implement the execution sequence above; stage 3 adds GRPO on top of the stage 2 baseline. All three reuse what the Hiligaynon GB10 pilot measured (`hiligaynon_tts/moss/README.md`). Nothing has run on Filipino data, because the data gate still has zero eligible rows. Settings live in `recipe.json` under `encode`, `judges`, `generation`, `stage1`, `stage2` and `latency`; pass/fail logic lives in `gates.py`. Every GPU command runs under `guard.py`, which holds `/tmp/sapinsapin-gpu.lock`, keeps the 25 GiB host-memory floor and enforces a time bound.

```sh
G="python -m training.fil_moss.guard --max-seconds"
python -m curation.tts_export --manifest ELIGIBLE.jsonl --recipe training/fil_moss/recipe.json --output RUN/data
$G 3600  -- python -m training.fil_moss.encode --export RUN/data --recipe training/fil_moss/recipe.json --output RUN/codes
$G 1800  -- python -m training.fil_moss.stage1 --codes RUN/codes --out RUN/stage1
$G 14400 -- python -m training.fil_moss.stage2 train --codes RUN/codes --stage1-report RUN/stage1/report.json --out RUN/stage2
$G 1800  -- python -m training.fil_moss.stage2 latency --checkpoint RUN/stage2/best_checkpoint --codes RUN/codes --out RUN/latency.json
$G 21600 -- python -m training.fil_moss.stage3 run --arm c --stage2-report RUN/stage2/report.json --codes RUN/codes \
             --prompts PROMPTS.txt --out RUN/stage3-c          # likewise arms a and b; d needs --bilingual/--ceiling
python -m training.fil_moss.stage3 compare --reports RUN/stage3-*/report.json
```

**Encode (step 3).** Re-verifies the export hashes and encodes 16 codebooks. It records each row's validation domain (`encode.domain_key`, default `source_id`) and whether the row is Tagalog-English code-switched. It attaches a same-speaker, different-clip reference from the same split for the separate reference-conditioned experiment; singleton speakers get none. It then decodes the first clips of each split back as a reconstruction check.

**Stage 1, engineering smoke (step 4).** Five checks; all must pass before stage 2 will start:

| Check | Passes when |
| --- | --- |
| Loss parity | Per-token log-probabilities from `local_logits` reproduce upstream's loss. They agreed exactly on a tiny random MOSS model on CPU. |
| Batch parity | Per-utterance values match one at a time vs in a padded batch, so batched training and evaluation are allowed. |
| One step | Forward/backward/AdamW8bit on real codes gives FP32 gradients and BF16 linear outputs, observed with a hook as in the pilot. Loss and norm are finite and the probed weights moved. |
| Save/reload | The checkpoint reloads to identical weights and an identical validation loss. |
| Samples | Matched-seed, unconditioned generations from the pretrained and one-step models are finite and not silent. They are written as WAVs. |

**Stage 2, bounded baseline and held-out evaluation (steps 5–6).** Starts from the pinned pretrained model, not the Hiligaynon checkpoint. Uses the pilot's configuration: FP32 weights and gradients, BF16 autocast, AdamW8bit, LR 1e-5, betas (0.9, 0.95), weight decay 0.1, 20 warmup steps, linear decay, clip 1.0, `language_model.embed_tokens` frozen. Seed, order (hashed) and schedule are recorded.
- **Update budget:** `epochs` passes over the frozen training split.
- **Conditioning:** none by default, as in the pilot. `conditioning: reference` is the separate multi-speaker experiment.
- **Evaluation:** at step 0 and every 128 updates, the pilot's cadence.
- **Selection:** the lowest equally weighted mean of per-domain validation losses, the pilot's rule. With the pilot's own numbers it keeps the step-0 checkpoint, as the pilot did. `best_checkpoint` and `last_checkpoint` are kept apart, and test rows are never read.
- **Generation metrics per evaluation:**
  - pooled CER per judge, worst judge and judge gap;
  - worst-judge CER per domain, and for code-switched vs monolingual rows;
  - median trimmed-duration ratio against the real recording, and frame-cap hits;
  - voice consistency (mean pairwise ECAPA similarity) for unconditioned runs, or similarity to the target speaker's real recording for reference-conditioned runs;
  - six WAVs for native listening.
- **Report:** deltas against step 0 per domain and per judge, any domain that got worse, and `quality: native listening pending`. A completed run is not a quality claim.

**Streaming latency.** `stage2 latency` drives upstream's own streaming session and frame decoder with upstream `app.py`'s defaults. It reports first-audio latency (p50/p90) and real-time factor at batch 1, separately from training, because offline generation time is not a streaming benchmark.

**Stage 3, GRPO on the baseline.** Starts from stage 2's selected checkpoint; it refuses if stage 2 kept the pretrained model. It keeps stage 2's conditioning. Each step generates 8 rollouts for one unpaired Filipino prompt, as one batch on the BF16 mirror. Prompts are dropped if they contain digits, are duplicates, or match a validation/test transcript.
- **Scoring:** every judge, ECAPA and pacing score each rollout. Rewards are within-group ranks of intelligibility (0.6), speaker (0.3) and −|log duration ratio| (0.1), an engineering weighting.
- **Update:** the clipped surrogate plus k3 KL to the frozen stage 2 checkpoint, in micro-batches. Rollouts that hit the frame cap lose their artificial EOS label.
- **Pre-registered arms:**
  - a: PLD-Whisper alone, the in-family single-judge baseline;
  - b: mean over judges;
  - c: worst judge;
  - d: worst judge plus the cross-lingual voice reward (CLVI): condition on a consented bilingual speaker's English clip and score against their real Filipino recordings.
- **Speaker term:** with the unconditioned MOSS baseline it is an *anchor*, similarity to the centroid of the stage 2 checkpoint's own voice measured at start, so RL cannot drift the voice. With a reference-conditioned stage 2 it is similarity to the conditioning clip. Arm d runs only then, and only with at least 50 consented bilingual speakers and an EN→FIL ECAPA ceiling of at least 0.60 (`stage3 ceiling`).
- **Stop rules,** all relative to the arm's own step-0 evaluation:
  - the voice measure drops by more than 0.02;
  - balanced validation loss rises by more than 10%;
  - no worst-judge gain by step 300;
  - a tripwire fires twice (reward rising while silence rises or codec-token entropy falls).
- **Selection:** the lowest worst-judge CER among evaluations that passed the stop rules. `compare` applies the pre-registered adoption rules: c over b, then d over c. The judge gap is logged and never rewarded. Native listening stays pending.
- **Inputs it needs:** an unpaired Filipino prompt file and, for arm d only, `bilingual.jsonl` with consent.

### Using both GPUs (GB10 + RTX 3070)

- **GB10 (dwnd)** runs the FP32 policy, plus a frozen BF16 *mirror* used for every generation and re-synced before each evaluation and after every stage 3 update. Stage 3 also holds the frozen BF16 stage 2 checkpoint for the KL term. The FP32 policy cannot generate under autocast because of its static KV cache, and BF16 halves the bytes per decode step at 273 GB/s. The codec, MMS (FP32, as validated) and ECAPA also run here. Generation and teacher-forced evaluation are batched.
- **RTX 3070 (reth5et)** runs `judge_server.py`, which serves Whisper in BF16, batched (Omnilingual is a 23 ms CTC pass and stays on the GB10 with MMS). It binds to the tailnet address and requires the token in `~/.config/fil-moss/judge-token` (mode 600, same file on both hosts). Only synthesised audio goes there; real recordings and their embeddings stay on the GB10. While the 3070 transcribes evaluation batch *i*, the GB10 generates batch *i+1*. In stage 3 (`pipeline: true`) the 3070 judges step *t* while the GB10 applies step *t−1*'s update and generates step *t+1*. Rollouts are therefore one update old when trained on; their behaviour log-probabilities are recorded at generation time and the clipped ratio corrects for the lag.
  - `judges.placement` decides what runs where. If the server is down at start, or fails three times mid-run, those judges load on the GB10 and the switch is recorded in `judge_events`.
  - Under WSL2, `torch.cuda.mem_get_info` does not count other processes, so the server reads `nvidia-smi` too. On 2026-11-15 the 3070 had about 1.8 GiB truly free until its `nyo-tts@orpheus-fil` and `nyo-asr-fw@flipvox` engines were stopped (`systemctl --user start` restores them).
- **Training batch.** `stage2.batch_size` stays 1, the pilot's measured configuration. On the GB10 a step costs 0.75 s at batch 1, 0.77 s at batch 4 and 1.28 s at batch 8 (20–24 GiB peak), so batch 4 is about 3.9× the throughput. It also means 4× fewer updates; set it deliberately.

Start the judge server on reth5et (model files under `~/fil-moss-judges`, offline):

```sh
cd ~/fil-moss-judges/repo && HF_HUB_OFFLINE=1 ~/nyo-serve/.venv-tts/bin/python -m training.fil_moss.judge_server \
  --host 100.92.72.53 --port 9310 --judges whisper --token-file ~/.config/fil-moss/judge-token \
  --omni-port-dir ~/fil-moss-judges/omniASR-CTC-1B-hf --omni-port-code ~/fil-moss-judges/code
```

`bench.py` measures every component on the GB10, and the remote judges if the server answers. `timing.py` turns those measurements into a wall-clock estimate for a given dataset size and batch size.

**Not yet verified:** no stage has run on real weights. Judges use greedy Whisper, not the bake-off's temperature fallback. Omnilingual loads the bake-off's port from `~/work/hil-tts`. ECAPA (speechbrain 1.1.2) is installed in the MOSS venv and measured at 3.1 ms per clip. The latency harness follows `app.py` but has not been run. Stage 3 needs a prompt file that does not exist yet.

Tests: `python -m unittest training.fil_moss.test_scoring training.fil_moss.test_gates training.fil_moss.test_stage3 training.fil_moss.test_timing training.fil_moss.test_guard training.fil_moss.test_objective` (the objective tests need torch).

## Optional tokenizer experiment

“Tokeniser morphormer” is not yet identified. Clarification is pending and does not block the unchanged-tokenizer baseline. Search found morphology-aware text methods such as [MorphBPE](https://arxiv.org/abs/2502.00894), but that is not evidence it is the intended method or improves Filipino MOSS.

First identify the exact paper/repository and whether it changes text processing, model architecture or acoustic codes. Diagnose tokens per word/character, unknown tokens and special-token round trips on Filipino and code-switched text. Preserve existing pretrained token IDs and acoustic codec. Any supported vocabulary extension requires explicit embedding initialization and a compatible reload path. Do not replace tokenizer IDs under pretrained weights. Test a suitable method as a separate matched-data/compute ablation and adopt it only after a measurable quality benefit. A different acoustic codec is a separate retraining experiment.

NYO `us.kimi-k3` reviewed the baseline/readiness/ablation design (0.103841 credits). Its suggestions are design input, not data approval or measured improvement.

Readiness regression checks: `python -m unittest curation.test_tts_readiness`. Seven cases cover valid code-switching metadata, missing audio review, unknown permission, duplicate audio, cross-split prompt leakage, changed audio and stale review. Fixture audio bytes test gate integrity only; they do not test waveform decoding or TTS training.

Waveform export checks: `python -m unittest curation.test_tts_export`. Three real-waveform tests cover resampling/export with exact text and split preservation, unreviewed-data rejection, and cleanup after a decoded-duration mismatch. GPU codec and training checks remain pending.
