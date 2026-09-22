#!/usr/bin/env bash
# whisper-large-v3 on normalised text for the eight languages the bake-off did
# not reach. ceb and pam exist as *-norm continuations; these start from the
# base model, so they get the bake-off's 5000 steps, not the continuation's
# 1500.
#
#   bash scripts/run_whisper_fleet_norm.sh                # all eight
#   LANGS="bcl war" bash scripts/run_whisper_fleet_norm.sh
#
# Same recipe as the bake-off's Whisper arm plus --normalise, so the numbers
# sit beside whisper-large-v3-pld-{ceb,pam}-norm. Own FINETUNE_DIR; the idle
# watchdog stops the VM if this dies.
set -uo pipefail
cd "$(dirname "$0")/.."

LANGS=${LANGS:-bcl eng fil hil ilo pag tsg war}
OUT=${FLEET_DIR:-/mnt/data/whisper_fleet_norm}
SUMMARY="$OUT/summary.txt"
mkdir -p "$OUT"
echo "=== whisper fleet (norm) start $(date -u +%F' '%T) langs='$LANGS'" | tee -a "$SUMMARY"

for lang in $LANGS; do
    name="asr_pld_${lang}_norm"
    if [ -f "$OUT/$name/result.json" ]; then
        echo "$name: done already" | tee -a "$SUMMARY"; continue
    fi
    echo "--- $name start $(date -u +%T)" | tee -a "$SUMMARY"
    FINETUNE_DIR="$OUT" venv/bin/python3 finetune_asr.py --model openai/whisper-large-v3 \
        --dataset pld --language "$lang" --normalise \
        --max-samples 25000 --max-steps 5000 --batch-size 16 --grad-accum 1 \
        --no-grad-checkpoint --attn sdpa --eval-steps 1000 --eval-samples 500 \
        --num-proc 8 --dataloader-workers 8 --resume \
        2>&1 | tr '\r' '\n' | grep -vE 'examples/s|it/s\]$'
    if [ -f "$OUT/$name/result.json" ]; then
        echo "$name: OK $(date -u +%T) $(grep -E 'eval_cer|eval_wer' "$OUT/$name/result.json" | tr -d ' \n')" | tee -a "$SUMMARY"
        rm -rf "$OUT/$name"/checkpoint-*
    else
        echo "$name: FAILED $(date -u +%T)" | tee -a "$SUMMARY"
    fi
done
echo "=== whisper fleet (norm) done $(date -u +%F' '%T)" | tee -a "$SUMMARY"
