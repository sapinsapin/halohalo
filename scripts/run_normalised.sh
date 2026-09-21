#!/usr/bin/env bash
# Continue the bake-off's winners on normalised text instead of retraining them.
#
#   bash scripts/run_normalised.sh ceb pam
#
# The audio has not changed and the encoders are where the compute went, so a
# label change does not need a restart: 1500 steps from the finished
# checkpoints, against the 5000 each of them took from the base model. The CTC
# head is rebuilt for the smaller vocabulary with its rows carried over for
# every character that survives normalisation.
#
# Lower learning rates than the from-scratch runs: these weights are already
# where they should be and only the targets moved.
#
# The CTC warm-start is smoke-tested first because it has never run, and this
# script runs unattended. A failed smoke stops the queue, and the idle watchdog
# then stops the VM.
set -uo pipefail
cd "$(dirname "$0")/.."

LANGS=${*:-ceb pam}
STEPS=${STEPS:-1500}
OUT=${NORM_DIR:-/mnt/data/normalised}
SUMMARY="$OUT/summary.txt"
mkdir -p "$OUT"
export FINETUNE_DIR="$OUT"
echo "=== normalised continuation start $(date -u +%F' '%T) langs='$LANGS'" | tee -a "$SUMMARY"

first=${LANGS%% *}
FINETUNE_DIR="$OUT/smoke" venv/bin/python3 finetune_ctc.py --encoder omni-1b \
        --language "$first" --units char --normalise --smoke \
        --init-from "sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_${first}" \
        --num-proc 8 > /tmp/norm_smoke.raw 2>&1
# grep the finished file, not the live pipe: grep -q exits at its first match,
# the writer dies of SIGPIPE, and pipefail reports a passing smoke as failed
tr '\r' '\n' < /tmp/norm_smoke.raw > /tmp/norm_smoke.txt
if ! grep -q "head rows carried over" /tmp/norm_smoke.txt || grep -q Traceback /tmp/norm_smoke.txt; then
    echo "smoke FAILED $(date -u +%T)" | tee -a "$SUMMARY"
    grep -vE '^\s*$' /tmp/norm_smoke.txt | tail -12 | tee -a "$SUMMARY"
    exit 1
fi
grep -E "head rows carried over|vocab:" /tmp/norm_smoke.txt | tee -a "$SUMMARY"
rm -rf "$OUT/smoke"

report() {  # report <run dir name>
    local r="$OUT/$1/result.json"
    if [ -f "$r" ]; then
        echo "$1: OK $(date -u +%T) $(grep -E 'eval_cer|eval_wer' "$r" | tr -d ' \n')" | tee -a "$SUMMARY"
        rm -rf "$OUT/$1"/checkpoint-*
    else
        echo "$1: FAILED $(date -u +%T)" | tee -a "$SUMMARY"
    fi
}

for lang in $LANGS; do
    name="ctc_omni-1b_char_pld_${lang}_norm"
    if [ ! -f "$OUT/$name/result.json" ]; then
        echo "--- $name start $(date -u +%T)" | tee -a "$SUMMARY"
        venv/bin/python3 finetune_ctc.py --encoder omni-1b --language "$lang" \
            --units char --normalise \
            --init-from "sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_${lang}" \
            --max-samples 25000 --max-steps "$STEPS" --lr 3e-5 --warmup 100 \
            --batch-size 16 --grad-accum 1 --no-grad-checkpoint --eval-steps 500 \
            --num-proc 8 --dataloader-workers 8 --resume \
            2>&1 | tr '\r' '\n' | grep -vE 'examples/s|it/s\]$'
        report "$name"
    fi

    name="asr_pld_${lang}_norm"
    if [ ! -f "$OUT/$name/result.json" ]; then
        echo "--- $name start $(date -u +%T)" | tee -a "$SUMMARY"
        venv/bin/python3 finetune_asr.py --model "sapinsapin/whisper-large-v3-pld-${lang}" \
            --dataset pld --language "$lang" --normalise \
            --max-samples 25000 --max-steps "$STEPS" --lr 5e-6 \
            --batch-size 16 --grad-accum 1 --no-grad-checkpoint --attn sdpa \
            --eval-steps 500 --num-proc 8 --dataloader-workers 8 --resume \
            2>&1 | tr '\r' '\n' | grep -vE 'examples/s|it/s\]$'
        report "$name"
    fi
done
echo "=== normalised continuation done $(date -u +%F' '%T)" | tee -a "$SUMMARY"
