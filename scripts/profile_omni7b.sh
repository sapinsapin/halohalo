#!/usr/bin/env bash
# Does the 7B Omnilingual encoder fit on one 96 GB card, and in which recipe?
#
#   bash scripts/profile_omni7b.sh            # waits for tts_fleet, then profiles
#
# Profile only: ~10 steps each, no checkpoints, so it is safe on a disk that a
# real 7B run (42 GB per checkpoint with optimiser state) would fill.
#
# The arithmetic says fused AdamW cannot work: 7B x (4 weights + 4 grads + 8
# Adam) = 112 GB before a single activation. Two recipes might:
#
#   A. fp32 weights + bitsandbytes 8-bit Adam   28 + 28 + 14 = 70 GB + activations
#   B. bf16 weights + bitsandbytes 8-bit Adam   14 + 14 + 14 = 42 GB + activations
#
# A is preferred — small updates round away in bf16 weights — so B only runs if
# A does not fit. Arithmetic was wrong about Orpheus by 20 GB (the 157k-token
# logits), which is why this measures instead of assuming.
set -uo pipefail
cd "$(dirname "$0")/.."

while tmux has-session -t tts_fleet 2>/dev/null; do sleep 60; done
echo "=== fleet finished $(date -u +%T); profiling omni-7b"

COMMON="--encoder omni-7b --language ceb --units char --max-samples 25000
        --batch-size 4 --grad-accum 4 --optim adamw_bnb_8bit
        --num-proc 8 --dataloader-workers 8 --profile"

for recipe in "A:" "B:--bf16-weights"; do
    name=${recipe%%:*}; flag=${recipe#*:}
    echo "--- recipe $name ${flag:-(fp32 weights)} start $(date -u +%T)"
    # shellcheck disable=SC2086
    FINETUNE_DIR=/mnt/data/profiles venv/bin/python3 finetune_ctc.py $COMMON $flag \
        2>&1 | tr '\r' '\n' | grep -vE 'examples/s|it/s\]$' \
        | grep -E "peak GPU|GPU kernel|GPU/CPU|out of memory|OutOfMemory|Error|=== .* steps" \
        | tee /tmp/omni7b_$name.txt
    if grep -q "peak GPU" /tmp/omni7b_$name.txt; then
        echo "=== recipe $name FITS"
        break
    fi
    echo "=== recipe $name did not fit"
done
echo "=== profile done $(date -u +%T)"
