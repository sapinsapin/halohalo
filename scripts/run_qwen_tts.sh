#!/usr/bin/env bash
# P3: Qwen3-TTS multi-speaker SFT, queued behind whatever holds the card.
#
#   bash scripts/run_qwen_tts.sh ceb pam
#
# Smoke first, on purpose. sft_qwen_tts.py has never run: its forward is
# upstream's, but the per-row reference handling and the base-mode save are
# ours, and a ten-step run finds a shape error in a minute where a full run
# would find it after the 7B job has handed over the card and nobody is
# watching. A failed smoke stops the queue; the idle watchdog then stops the VM.
set -uo pipefail
cd "$(dirname "$0")/.."

LANGS=${*:-ceb}
PY=venv_qwen/bin/python3
RUNS=${FINETUNE_DIR:-/mnt/data/finetune_runs}
SUMMARY="$RUNS/qwen3tts_summary.txt"

while tmux has-session -t omni7b 2>/dev/null; do sleep 60; done
echo "=== qwen3-tts start $(date -u +%F' '%T) langs='$LANGS'" | tee -a "$SUMMARY"

first=${LANGS%% *}
if ! $PY scripts/sft_qwen_tts.py --language "$first" --max-steps 10 2>&1 \
        | tr '\r' '\n' | grep -vE '^\s*$' | tail -25 | tee /tmp/qwen_smoke.txt \
        | grep -q "smoke test: stopping"; then
    echo "smoke FAILED $(date -u +%T) — see below" | tee -a "$SUMMARY"
    tail -12 /tmp/qwen_smoke.txt | tee -a "$SUMMARY"
    exit 1
fi
grep -E "step 10 " /tmp/qwen_smoke.txt | tee -a "$SUMMARY"

for lang in $LANGS; do
    if [ -f "$RUNS/qwen3tts_pld_${lang}/result.json" ]; then
        echo "$lang: done already" | tee -a "$SUMMARY"; continue
    fi
    echo "--- $lang start $(date -u +%T)" | tee -a "$SUMMARY"
    $PY scripts/sft_qwen_tts.py --language "$lang" 2>&1 | tr '\r' '\n' | grep -vE '^\s*$'
    if [ -f "$RUNS/qwen3tts_pld_${lang}/result.json" ]; then
        echo "$lang: OK $(date -u +%T)" | tee -a "$SUMMARY"
    else
        echo "$lang: FAILED $(date -u +%T)" | tee -a "$SUMMARY"
    fi
done
echo "=== qwen3-tts done $(date -u +%F' '%T)" | tee -a "$SUMMARY"
