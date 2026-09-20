#!/usr/bin/env bash
# Take a cold VM to the point where a TTS run starts at step 0.
#
#   bash scripts/prewarm_tts.sh ceb pam
#
# Three things have to happen before the first training step, and left to a
# training job they happen in series with the GPU idle:
#   1. ~7 GB of model weights download          (network-bound)
#   2. PLD's 300k rows are filtered to one language (CPU-bound, ~10 min at
#      num_proc=1, which is why --cloud sets 8)
#   3. every clip is resampled to 24kHz and encoded to SNAC tokens
#
# This runs (1) against (2)+(3) instead of after them, and writes both caches
# to the persistent disk, so every later arm of the ablation — three frontends
# per language — starts immediately. Re-running is cheap: everything checks for
# its cache first.
#
# Measured on 2026-09-19: the first uncached Cebuano run spent ~10 minutes
# filtering and several more encoding before it reached a training step.
set -euo pipefail
cd "$(dirname "$0")/.."

LANGS=${*:-ceb pam}
PY=venv/bin/python3
# Must match the training run's --max-samples: the SNAC cache is keyed by
# corpus size, so a prewarm at any other size is one the run never reads.
SAMPLES=${SAMPLES:-20000}

echo "=== prewarming: $LANGS"

# Weights first and in the background: it is the only network-bound step, so it
# should overlap the CPU and GPU work rather than precede it.
$PY - <<'PY' &
import os
from huggingface_hub import snapshot_download

for repo in ("unsloth/orpheus-3b-0.1-pretrained", "hubertsiuzdak/snac_24khz"):
    snapshot_download(repo, token=os.environ.get("HF_TOKEN"))
    print(f"  weights: {repo}", flush=True)
PY
weights=$!

for lang in $LANGS; do
    echo "=== caches: $lang"
    # --cache-only stops before the base model loads: encoding audio does not
    # need 3.3B parameters resident, and on a preemptible VM the shorter the
    # unprotected window the better
    $PY finetune_orpheus.py --cache-only --cloud --dataset pld --language "$lang" \
        --max-samples "$SAMPLES"
done

wait $weights
echo "=== prewarm done; a training run now starts at step 0"
