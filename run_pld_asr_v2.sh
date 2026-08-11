#!/usr/bin/env bash
# Second-pass ASR training for the languages the first fleet left short.
#
# Pass 1 capped every language at 10k clips / 2000 steps. Four languages were
# still improving at the final eval, and Kapampangan was both the weakest and
# unstable — all of them have far more data available than pass 1 used
# (Bikol 62k, Cebuano 57k, Kapampangan 58k utterances). This pass raises both
# the data cap and the step budget.
#
# Runs write to finetune_runs_v2 so pass-1 models stay intact, and each result
# is pushed only if it beats the published CER (compare_and_push_asr.py).
set -u
cd "$(dirname "$0")"

if [ -f ts-wandb.txt ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < ts-wandb.txt)"
    export WANDB_PROJECT="${WANDB_PROJECT:-halohalo-pld}"
fi

LANGS=(${PLD_ASR2_LANGS:-ceb bcl fil war pam})
STEPS=${PLD_ASR2_STEPS:-5000}
SAMPLES=${PLD_ASR2_SAMPLES:-25000}
RUNDIR=finetune_runs_v2
LOGDIR=finetune_runs
export FINETUNE_DIR="$PWD/$RUNDIR"
mkdir -p "$RUNDIR" "$LOGDIR"
SUMMARY="$LOGDIR/pld_asr_v2_summary.txt"
: > "$SUMMARY"

for lang in "${LANGS[@]}"; do
    if [ -f "$RUNDIR/asr_pld_${lang}/DONE" ]; then
        echo "asr2 ${lang}: already done, skipping" | tee -a "$SUMMARY"
        continue
    fi
    echo "=== ASR2 ${lang} start $(date -u +%H:%M:%S) ===" | tee -a "$SUMMARY"
    if venv/bin/python3 finetune_asr.py --dataset pld --language "$lang" \
        --max-samples "$SAMPLES" --max-steps "$STEPS" \
        > "$LOGDIR/asr2_pld_${lang}.log" 2>&1; then
        touch "$RUNDIR/asr_pld_${lang}/DONE"
        # push only on a real CER improvement over what is already published
        venv/bin/python3 compare_and_push_asr.py "$lang" --run-dir "$RUNDIR" \
            >> "$LOGDIR/asr2_pld_${lang}.log" 2>&1
        grep -hE "^${lang}: (new CER|keeping)" "$LOGDIR/asr2_pld_${lang}.log" \
            | tail -2 | tee -a "$SUMMARY"
        echo "asr2 ${lang}: OK $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
    else
        echo "asr2 ${lang}: FAILED (see $LOGDIR/asr2_pld_${lang}.log)" | tee -a "$SUMMARY"
    fi
    # v2 checkpoints are ~1GB each; keep only the best-model dir per language
    rm -rf "$RUNDIR/asr_pld_${lang}"/checkpoint-* 2>/dev/null || true
done

echo "ASR_V2_DONE $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
