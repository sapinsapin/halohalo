#!/usr/bin/env bash
# P1 in docs/tts_sota_plan.md: does a codec LM read Philippine text better as
# BPE, as syllables, or spelled out?
#
#   bash scripts/run_tts_ablation.sh ceb
#   ARMS="bpe syllable" STEPS=500 bash scripts/run_tts_ablation.sh ceb pam
#
# Three arms per language, same audio, same steps, differing only in the text
# the model is conditioned on. Resumable and idempotent: an arm whose final/
# exists is skipped, so a preemption costs one arm, not the sweep.
#
# The CTC arm answered the same question for output alphabets (characters beat
# syllables, 17.0 vs 22.8 CER on ceb). That does not settle this one: there the
# units were what the model emitted, here they are what it reads.
set -uo pipefail
cd "$(dirname "$0")/.."

LANGS=${*:-ceb}
ARMS=${ARMS:-bpe syllable char}
STEPS=${STEPS:-2000}
SAMPLES=${SAMPLES:-20000}
RUNS=${FINETUNE_DIR:-/mnt/data/finetune_runs}
SUMMARY="$RUNS/tts_ablation_summary.txt"

mkdir -p "$RUNS"
echo "=== ablation start $(date -u +%F' '%T) langs='$LANGS' arms='$ARMS' steps=$STEPS" >> "$SUMMARY"

for lang in $LANGS; do
    # One prewarm per language rather than one per arm: the SNAC cache is
    # keyed by audio, so all three arms read the same file.
    SAMPLES="$SAMPLES" bash scripts/prewarm_tts.sh "$lang"

    for units in $ARMS; do
        name="orpheus_${units}_pld_${lang}"
        if [ -d "$RUNS/$name/final" ]; then
            echo "$name: already done, skipping" | tee -a "$SUMMARY"
            continue
        fi

        echo "--- $name start $(date -u +%T)" | tee -a "$SUMMARY"
        # --cloud is the 96GB recipe; the profile measured 72.1 GiB peak at
        # batch 8, so this is close to the card's limit already. The 157k-token
        # vocabulary is what costs it, not the LoRA.
        venv/bin/python3 finetune_orpheus.py \
            --cloud --dataset pld --language "$lang" --units "$units" \
            --max-samples "$SAMPLES" --max-steps "$STEPS" --resume \
            2>&1 | tr '\r' '\n' | grep -vE 'examples/s|it/s\]$'

        if [ -d "$RUNS/$name/final" ]; then
            echo "$name: OK $(date -u +%T)" | tee -a "$SUMMARY"
        else
            echo "$name: FAILED $(date -u +%T)" | tee -a "$SUMMARY"
        fi
    done
done

echo "=== ablation done $(date -u +%F' '%T)" >> "$SUMMARY"
echo
echo "Score the arms with:"
for lang in $LANGS; do
    for units in $ARMS; do
        echo "  scripts/tts_eval.py --stage synth --device cuda --model orpheus:$RUNS/orpheus_${units}_pld_${lang}/final"
    done
done
