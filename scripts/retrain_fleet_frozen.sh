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

# PROFILE=local is the workstation's RTX 3070, where the published fleet was
# trained: its own recipe (batch 8 x accum 2, checkpointing on, 10k clips, 2000
# steps) fits the ~6 GB free and runs ~1 h a language. One dataset process,
# because multiprocess map over decoded audio deadlocks on WSL's 9p mounts.
# PROFILE=gb10 is the 128 GB GB10: the same effective batch as the published
# recipe (16, so the numbers stay comparable), no checkpointing, and the
# spare memory spent on running JOBS languages at once rather than on a
# bigger batch, which would change what is being measured.
# The default is the 96 GB cloud card.
JOBS=1
case "${PROFILE:-cloud}" in
  local)
    HW="--batch-size 8 --grad-accum 2 --num-proc 1 --dataloader-workers 2"
    SAMPLES=${SAMPLES_LOCAL:-10000} ;;
  gb10)
    HW="--batch-size 16 --grad-accum 1 --no-grad-checkpoint --num-proc 4 --dataloader-workers 4"
    SAMPLES=${SAMPLES_GB10:-10000}
    JOBS=${JOBS_GB10:-3} ;;
  *)
    HW="--batch-size 16 --grad-accum 1 --no-grad-checkpoint --num-proc 8 --dataloader-workers 8" ;;
esac
SUMMARY="$OUT/summary.txt"

mkdir -p "$OUT"
echo "=== fleet retrain start $(date -u +%F' '%T) langs='$LANGS' steps=$STEPS jobs=$JOBS" >> "$SUMMARY"

train_one() {
    local lang=$1
    echo "--- $lang start $(date -u +%T)" | tee -a "$SUMMARY"
    FINETUNE_DIR="$OUT" venv/bin/python3 finetune_asr.py \
        --model openai/whisper-small --dataset pld --language "$lang" \
        --max-samples "$SAMPLES" --max-steps "$STEPS" \
        $HW \
        --eval-samples 500 --resume \
        2>&1 | tr '\r' '\n' | grep -vE 'examples/s|it/s\]$'
    if [ -f "$OUT/asr_pld_${lang}/result.json" ]; then
        echo "$lang: OK $(date -u +%T)" | tee -a "$SUMMARY"
    else
        echo "$lang: FAILED $(date -u +%T)" | tee -a "$SUMMARY"
    fi
}

for lang in $LANGS; do
    if [ -f "$OUT/asr_pld_${lang}/result.json" ]; then
        echo "$lang: done already, skipping" | tee -a "$SUMMARY"
        continue
    fi
    if [ "$JOBS" -gt 1 ]; then
        # one log per language: parallel runs would interleave on stdout.
        # A start also waits for JOB_GB (one run's peak) + MIN_FREE_GB of
        # free memory, and the next start waits STAGGER s so it sees this
        # run's memory: on unified memory the GPU's share counts too.
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
        while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt $(( ${MIN_FREE_GB:-16} + ${JOB_GB:-24} )) ]; do
            sleep 30
        done
        train_one "$lang" > "$OUT/asr_pld_${lang}.log" 2>&1 &
        sleep "${STAGGER:-180}"
    else
        train_one "$lang"
    fi
done
wait

echo "=== fleet retrain done $(date -u +%F' '%T)" >> "$SUMMARY"
for lang in $LANGS; do
    r="$OUT/asr_pld_${lang}/result.json"
    [ -f "$r" ] && python3 -c "
import json; d = json.load(open('$r'))
print(f\"  {'$lang':5s} CER {d.get('cer', 0)*100:6.2f}%  WER {d.get('wer', 0)*100:6.2f}%\")"
done
