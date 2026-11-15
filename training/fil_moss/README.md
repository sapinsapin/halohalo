# Authorized Filipino MOSS fine-tuning workflow

The user authorized starting Filipino TTS fine-tuning when data is ready (2026-11-15). No further launch confirmation is required within this scope. Current state: **authorized, pending eligible data**, not an active trainer or a scheduled watcher. Continue this work under the active dataset-curation goal. Data-readiness checking and waveform export are executable. Codec encoding, stage 1 (SFT) and stage 2 (GRPO) are wired but not yet run on real weights or data (see "Stage 1 and stage 2 code" below).

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

## Stage 1 and stage 2 code (2026-11-15)

Steps 3–6 above are now wired as two stages adapted from the Qwen3-TTS post-training plan (`docs/qwen_tts_sota_plan.md` §4 and §5, applied to MOSS-TTS-Realtime). Nothing has run on Filipino data: the data gate still has zero eligible rows. Thresholds and settings live in `recipe.json` under `encode`, `judges`, `generation`, `stage1` and `stage2`; the gating logic is in `gates.py` and `grpo.py`. Each GPU stage runs under `guard.py`, which holds `/tmp/sapinsapin-gpu.lock`, keeps the 25 GiB host-memory floor and enforces a time bound.

| Module | Runs on | Does |
| --- | --- | --- |
| `encode.py` | dwnd, MOSS venv | export → 16-codebook codes; reference voice = another clip by the same speaker in the same split; codec reconstruction check |
| `stage1.py verify` | dwnd | S1: loss parity (our token log-probs vs upstream loss), 32-clip overfit with duration check, save/reload |
| `stage1.py train` | dwnd | S2: full SFT, FP32 weights + BF16 autocast + AdamW8bit, step-0 zero-shot eval, duration and worse-than-zero-shot kills, best/last checkpoints, success gate |
| `stage2.py ceiling / run / compare` | dwnd | ECAPA EN→FIL ceiling; one GRPO arm (a–d) per run; pre-registered adoption decisions |
| `scoring.py`, `gates.py`, `grpo.py`, `guard.py` | anywhere (stdlib) | CER, rank rewards, advantages, tripwires, every pass/kill decision |
| `objective.py` | torch | clipped-surrogate GRPO loss with k3 KL to the stage 1 model |

Run order once the export exists, each GPU command prefixed with `python -m training.fil_moss.guard --max-seconds N --`:

```sh
python -m training.fil_moss.encode --export RUN/data --recipe training/fil_moss/recipe.json --output RUN/codes
python -m training.fil_moss.stage1 verify --codes RUN/codes --out RUN/stage1-verify
python -m training.fil_moss.stage1 train --codes RUN/codes --verify-report RUN/stage1-verify/verify_report.json \
    --out RUN/stage1 --english ENGLISH.jsonl
python -m training.fil_moss.stage2 ceiling --bilingual BILINGUAL.jsonl --out RUN/ceiling.json
python -m training.fil_moss.stage2 run --arm a --stage1-report RUN/stage1/report.json --codes RUN/codes \
    --prompts PROMPTS.txt --out RUN/stage2-a        # then b, c, and d with --bilingual/--ceiling
python -m training.fil_moss.stage2 compare --reports RUN/stage2-*/report.json
```

**Stage 1.** `verify` must pass before `train` starts. Its parity check matters most: stage 2 optimises per-token log-probabilities gathered from `local_logits` under teacher forcing, and parity proves they reproduce upstream's training loss. On a tiny randomly initialised MOSS model on CPU (2026-11-15) the two agreed exactly, with and without a reference voice. The overfit check is the guard against the never-stops failure that broke our Qwen3-TTS finetune. `train` starts from the pinned pretrained model. Success, judged on the selected validation checkpoint: MMS-judge CER ≤ 7.5%, ECAPA similarity to the speaker's real recording ≥ 0.76, duration ratio within [0.93, 1.07], and English worst-judge CER regression ≤ 5% relative. A missing measurement, such as having no English regression set, gives `incomplete`, never `pass`. A mean duration ratio above 1.3 kills the run, and so does ending two epochs worse than zero-shot.

**Stage 2.** It refuses to start unless stage 1 verification and success both passed. The plan's kill clause would allow RL from the zero-shot base, but its dependency section cancels stage 2; the code follows the stricter rule. Each step takes one unpaired prompt and samples 8 rollouts conditioned on a training speaker's clip. Rewards are within-group ranks of intelligibility, speaker similarity and −|log duration ratio|, weighted 0.6/0.3/0.1 (an engineering choice). Arm a uses PLD-Whisper alone; b uses the mean over judges; c and d use the worst judge. Arm d adds the cross-lingual voice reward (CLVI) on 25% of batches: condition on a bilingual speaker's English clip and score against their real Filipino recordings. CLVI runs as a reward only with at least 50 consented bilingual speakers and an EN→FIL ECAPA ceiling of at least 0.60; otherwise arm d refuses to run. Prompts with digits, duplicates, or text equal to a validation/test transcript are dropped. The judge gap is logged and never rewarded. A tripwire aborts the run when the reward rises while silence rises or codec-token entropy falls; two aborts kill the arm. Further kills: similarity below 0.75, or no worst-judge gain by step 300.

**Not yet verified:** no stage has run on real weights or a GPU. Judges use Whisper greedy decoding, not the bake-off's temperature fallback. The Omnilingual judge loads the bake-off's port from `~/work/hil-tts`. ECAPA needs `pip install speechbrain==1.1.2` in the MOSS venv, which is not yet installed. Policy, reference model, codec and three judges should fit in about 40 GiB, but that is an estimate until measured. The bilingual-speaker file and the English regression set do not exist yet.

Tests: `python -m unittest training.fil_moss.test_scoring training.fil_moss.test_gates training.fil_moss.test_grpo training.fil_moss.test_objective` (the objective tests need torch and skip without it).

## Optional tokenizer experiment

“Tokeniser morphormer” is not yet identified. Clarification is pending and does not block the unchanged-tokenizer baseline. Search found morphology-aware text methods such as [MorphBPE](https://arxiv.org/abs/2502.00894), but that is not evidence it is the intended method or improves Filipino MOSS.

First identify the exact paper/repository and whether it changes text processing, model architecture or acoustic codes. Diagnose tokens per word/character, unknown tokens and special-token round trips on Filipino and code-switched text. Preserve existing pretrained token IDs and acoustic codec. Any supported vocabulary extension requires explicit embedding initialization and a compatible reload path. Do not replace tokenizer IDs under pretrained weights. Test a suitable method as a separate matched-data/compute ablation and adopt it only after a measurable quality benefit. A different acoustic codec is a separate retraining experiment.

NYO `us.kimi-k3` reviewed the baseline/readiness/ablation design (0.103841 credits). Its suggestions are design input, not data approval or measured improvement.

Readiness regression checks: `python -m unittest curation.test_tts_readiness`. Seven cases cover valid code-switching metadata, missing audio review, unknown permission, duplicate audio, cross-split prompt leakage, changed audio and stale review. Fixture audio bytes test gate integrity only; they do not test waveform decoding or TTS training.

Waveform export checks: `python -m unittest curation.test_tts_export`. Three real-waveform tests cover resampling/export with exact text and split preservation, unreviewed-data rejection, and cleanup after a decoded-duration mismatch. GPU codec and training checks remain pending.
