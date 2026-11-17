# Authorized Filipino MOSS fine-tuning workflow

The user authorized starting Filipino TTS fine-tuning when data is ready (2026-11-15). No further launch confirmation is required within this scope. Current state: **authorized, pending eligible data**, not an active trainer or a scheduled watcher. Continue this work under the active dataset-curation goal. Data-readiness checking and waveform export are executable; codec encoding and training integration remain to be wired and validated.

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

## Optional tokenizer experiment

“Tokeniser morphormer” is not yet identified. Clarification is pending and does not block the unchanged-tokenizer baseline. Search found morphology-aware text methods such as [MorphBPE](https://arxiv.org/abs/2502.00894), but that is not evidence it is the intended method or improves Filipino MOSS.

First identify the exact paper/repository and whether it changes text processing, model architecture or acoustic codes. Diagnose tokens per word/character, unknown tokens and special-token round trips on Filipino and code-switched text. Preserve existing pretrained token IDs and acoustic codec. Any supported vocabulary extension requires explicit embedding initialization and a compatible reload path. Do not replace tokenizer IDs under pretrained weights. Test a suitable method as a separate matched-data/compute ablation and adopt it only after a measurable quality benefit. A different acoustic codec is a separate retraining experiment.

NYO `us.kimi-k3` reviewed the baseline/readiness/ablation design (0.103841 credits). Its suggestions are design input, not data approval or measured improvement.

Readiness regression checks: `python -m unittest curation.test_tts_readiness`. Seven cases cover valid code-switching metadata, missing audio review, unknown permission, duplicate audio, cross-split prompt leakage, changed audio and stale review. Fixture audio bytes test gate integrity only; they do not test waveform decoding or TTS training.

Waveform export checks: `python -m unittest curation.test_tts_export`. Three real-waveform tests cover resampling/export with exact text and split preservation, unreviewed-data rejection, and cleanup after a decoded-duration mismatch. GPU codec and training checks remain pending.
