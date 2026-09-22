#!/usr/bin/env bash
# Retrain the whisper-small fleet on the frozen speaker- and prompt-disjoint
# splits, so the fleet can be compared with the bake-off models honestly.
#
#   bash scripts/retrain_fleet_frozen.sh              # all ten
#   LANGS="ceb pam" STEPS=2000 bash scripts/retrain_fleet_frozen.sh
#
# Why retrain rather than re-measure (plan item A7, corrected 2026-09-19):
# the published fleet was trained before the splits were frozen, on the hub's
# random split. Measured on ceb, 20.3% of the clips it trained on belong to the
# speakers and prompts the frozen split holds out — 10,042 clips from the 28
# frozen test speakers. Scoring those checkpoints on the frozen test set
# therefore measures speakers the model has heard, which is how whisper-small
# came out at 10.75% CER against whisper-large-v3's 16.38% on the same split:
# not a better model, a contaminated one.
#
# These runs land in their own FINETUNE_DIR so they cannot overwrite the
# bake-off's asr_pld_* results.
set -uo pipefail
cd "$(dirname "$0")/.."

LANGS=${LANGS:-bcl ceb eng fil hil ilo pag tsg war pam}
STEPS=${STEPS:-2000}
SAMPLES=${SAMPLES:-25000}
OUT=${FLEET_DIR:-/mnt/data/fleet_frozen}
SUMMARY="$OUT/summary.txt"

mkdir -p "$OUT"
echo "=== fleet retrain start $(date -u +%F' '%T) langs='$LANGS' steps=$STEPS" >> "$SUMMARY"

for lang in $LANGS; do
    if [ -f "$OUT/asr_pld_${lang}/result.json" ]; then
        echo "$lang: done already, skipping" | tee -a "$SUMMARY"
        continue
    fi
    echo "--- $lang start $(date -u +%T)" | tee -a "$SUMMARY"
    FINETUNE_DIR="$OUT" venv/bin/python3 finetune_asr.py \
        --model openai/whisper-small --dataset pld --language "$lang" \
        --max-samples "$SAMPLES" --max-steps "$STEPS" \
        --batch-size 16 --grad-accum 1 --no-grad-checkpoint \
        --num-proc 8 --dataloader-workers 8 --eval-samples 500 --resume \
        2>&1 | tr '\r' '\n' | grep -vE 'examples/s|it/s\]$'

    if [ -f "$OUT/asr_pld_${lang}/result.json" ]; then
        echo "$lang: OK $(date -u +%T)" | tee -a "$SUMMARY"
    else
        echo "$lang: FAILED $(date -u +%T)" | tee -a "$SUMMARY"
    fi
done

echo "=== fleet retrain done $(date -u +%F' '%T)" >> "$SUMMARY"
for lang in $LANGS; do
    r="$OUT/asr_pld_${lang}/result.json"
    [ -f "$r" ] && python3 -c "
import json; d = json.load(open('$r'))
print(f\"  {'$lang':5s} CER {d.get('cer', 0)*100:6.2f}%  WER {d.get('wer', 0)*100:6.2f}%\")"
done
