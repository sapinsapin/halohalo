#!/usr/bin/env bash
# Sequential per-language PLD ASR fleet: whisper-small-pld-<lang> for each
# language, pushed to the Hub on completion. Same PUSHED-marker resume
# semantics as run_pld_finetunes.sh. Training curves stream to W&B when
# ts-wandb.txt is present (key stays out of the repo).
set -u
cd "$(dirname "$0")"

if [ -f ts-wandb.txt ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < ts-wandb.txt)"
    export WANDB_PROJECT="${WANDB_PROJECT:-halohalo-pld}"
fi

LANGS=(${PLD_ASR_LANGS:-bcl ceb eng fil hil ilo pag pam tsg war})
STEPS=${PLD_ASR_STEPS:-2000}
SAMPLES=${PLD_ASR_SAMPLES:-10000}
LOGDIR=finetune_runs
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/pld_asr_summary.txt"
: > "$SUMMARY"

for lang in "${LANGS[@]}"; do
    if [ -f "$LOGDIR/asr_pld_${lang}/PUSHED" ]; then
        echo "asr ${lang}: already pushed, skipping" | tee -a "$SUMMARY"
        continue
    fi
    echo "=== ASR ${lang} start $(date -u +%H:%M:%S) ===" | tee -a "$SUMMARY"
    if venv/bin/python3 finetune_asr.py --dataset pld --language "$lang" \
        --max-samples "$SAMPLES" --max-steps "$STEPS" --push \
        > "$LOGDIR/asr_pld_${lang}.log" 2>&1; then
        touch "$LOGDIR/asr_pld_${lang}/PUSHED"
        echo "asr ${lang}: OK $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
    else
        echo "asr ${lang}: FAILED (see $LOGDIR/asr_pld_${lang}.log)" | tee -a "$SUMMARY"
    fi
done

echo "ASR_FLEET_DONE $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
