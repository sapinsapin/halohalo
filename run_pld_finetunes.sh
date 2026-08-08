#!/usr/bin/env bash
# Sequential PLD finetune fleet on the single 8GB GPU:
#   1. speecht5_tts-pld-<lang> for each language (read-speech TTS)
#   2. speecht5_vc-pld — multilingual any-to-any voice conversion
# Each run pushes to the Hub on completion. A failed language is logged and
# skipped so one bad run cannot sink the fleet; rerun the script to retry
# (completed models are cheap to re-verify against the Hub before retraining).
set -u
cd "$(dirname "$0")"

LANGS=(${PLD_TTS_LANGS:-bcl ceb eng fil hil ilo pag pam tsg war})
STEPS=${PLD_TTS_STEPS:-1000}
SAMPLES=${PLD_TTS_SAMPLES:-2000}
LOGDIR=finetune_runs
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/pld_fleet_summary.txt"
: > "$SUMMARY"

for lang in "${LANGS[@]}"; do
    if [ -f "$LOGDIR/tts_pld_${lang}/PUSHED" ]; then
        echo "tts ${lang}: already pushed, skipping" | tee -a "$SUMMARY"
        continue
    fi
    echo "=== TTS ${lang} start $(date -u +%H:%M:%S) ===" | tee -a "$SUMMARY"
    if venv/bin/python3 finetune_tts.py --dataset pld --language "$lang" \
        --max-samples "$SAMPLES" --max-steps "$STEPS" --push \
        > "$LOGDIR/tts_pld_${lang}.log" 2>&1; then
        touch "$LOGDIR/tts_pld_${lang}/PUSHED"
        echo "tts ${lang}: OK $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
    else
        echo "tts ${lang}: FAILED (see $LOGDIR/tts_pld_${lang}.log)" | tee -a "$SUMMARY"
    fi
done

if [ ! -f "$LOGDIR/s2s_pld/PUSHED" ]; then
    echo "=== S2S start $(date -u +%H:%M:%S) ===" | tee -a "$SUMMARY"
    if venv/bin/python3 finetune_s2s.py --push \
        > "$LOGDIR/s2s_pld.log" 2>&1; then
        touch "$LOGDIR/s2s_pld/PUSHED"
        echo "s2s: OK $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
    else
        echo "s2s: FAILED (see $LOGDIR/s2s_pld.log)" | tee -a "$SUMMARY"
    fi
else
    echo "s2s: already pushed, skipping" | tee -a "$SUMMARY"
fi

echo "FLEET_DONE $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
