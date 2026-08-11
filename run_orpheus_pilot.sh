#!/usr/bin/env bash
# halo-tts pilot: QLoRA a 3B Orpheus codec-LM on FSC (see
# docs/orpheus_pilot_plan.md). Waits for the GPU to be free before starting —
# two trainers on an 8GB card OOMs both, and the PLD ASR fleet has priority.
set -u
cd "$(dirname "$0")"

LOGDIR=finetune_runs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/orpheus_pilot.log"

if [ -f ts-wandb.txt ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < ts-wandb.txt)"
    export WANDB_PROJECT="${WANDB_PROJECT:-halohalo-tts}"
fi

echo "=== orpheus pilot queued $(date -u +%H:%M:%S) ===" | tee "$LOG"

# 1. wait for any other trainer to exit
while pgrep -f 'finetune_(asr|tts|s2s)\.py' > /dev/null; do
    echo "  $(date -u +%H:%M:%S) waiting: another finetune holds the GPU" >> "$LOG"
    sleep 300
done

# 2. wait for VRAM to actually drain (driver frees lazily)
for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && break
    echo "  $(date -u +%H:%M:%S) waiting: ${used}MiB still in use" >> "$LOG"
    sleep 60
done
echo "=== GPU free (${used}MiB), starting $(date -u +%H:%M:%S) ===" >> "$LOG"

# 3. smoke test the whole path on a handful of clips before the long run
echo "--- smoke test ---" >> "$LOG"
if ! venv/bin/python3 -u finetune_orpheus.py --dataset fsc \
        --max-samples 24 --max-steps 2 --grad-accum 1 >> "$LOG" 2>&1; then
    echo "SMOKE TEST FAILED — not starting the full run" >> "$LOG"
    exit 1
fi
echo "--- smoke test OK ---" >> "$LOG"

# 4. the real run
echo "--- full run $(date -u +%H:%M:%S) ---" >> "$LOG"
venv/bin/python3 -u finetune_orpheus.py --dataset fsc \
    --max-samples "${ORPHEUS_SAMPLES:-2000}" \
    --max-steps "${ORPHEUS_STEPS:-1500}" \
    --push >> "$LOG" 2>&1
status=$?
echo "=== pilot exit=$status $(date -u +%H:%M:%S) ===" >> "$LOG"
exit $status
