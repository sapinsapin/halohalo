#!/usr/bin/env bash
# D2 of docs/asr_decoder_plan.md, first arm: was the CTC model simply stopped
# too early? omni-7B's dev CER was still falling at 5000 steps (pam: 12.1 ->
# 9.48 over the last 2000), so part of the gap to Whisper may be training
# budget rather than the decoder. This separates the two.
#
#   bash scripts/run_decoder_d2.sh ceb
#
# omni-1B, not 7B: two languages agree the 7B buys nothing under a linear head,
# at seven times the cost per step. Everything but the step count matches the
# bake-off, so the 5000-step row already in the table is the control.
#
# Own FINETUNE_DIR: the bake-off's ctc_omni-1b_char_pld_<lang> has the same run
# name and must not be overwritten or resumed from.
#
# InterCTC and a small subword vocabulary are D2's other two arms. They need
# trainer changes and are not here yet.
set -uo pipefail
cd "$(dirname "$0")/.."

LANGS=${*:-ceb}
STEPS=${STEPS:-15000}
OUT=${D2_DIR:-/mnt/data/decoder_d2}
WAIT_FOR=${WAIT_FOR:-qwen_pam}
mkdir -p "$OUT"

while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 60; done

for lang in $LANGS; do
    name="ctc_omni-1b_char_pld_${lang}"
    [ -f "$OUT/$name/result.json" ] && { echo "$name: done"; continue; }
    echo "--- $name ${STEPS} steps, start $(date -u +%T)" | tee -a "$OUT/summary.txt"
    FINETUNE_DIR="$OUT" venv/bin/python3 finetune_ctc.py --encoder omni-1b \
        --language "$lang" --units char --max-samples 25000 --max-steps "$STEPS" \
        --batch-size 16 --grad-accum 1 --no-grad-checkpoint --eval-steps 1000 \
        --num-proc 8 --dataloader-workers 8 --resume \
        2>&1 | tr '\r' '\n' | grep -vE 'examples/s|it/s\]$'
    if [ -f "$OUT/$name/result.json" ]; then
        echo "$name: OK $(date -u +%T) $(grep -E 'eval_cer|eval_wer' "$OUT/$name/result.json" | tr -d ' \n')" | tee -a "$OUT/summary.txt"
        rm -rf "$OUT/$name"/checkpoint-*
    else
        echo "$name: FAILED $(date -u +%T)" | tee -a "$OUT/summary.txt"
    fi
done
