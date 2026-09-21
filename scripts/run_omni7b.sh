#!/usr/bin/env bash
# Does scale flip R1? The 7B Omnilingual encoder, CTC over characters, on the
# bake-off's two languages and under the bake-off's recipe so the numbers sit
# in the same table as omni-1b (17.04 / 10.21 CER) and whisper-large-v3
# (16.38 / 5.83).
#
#   bash scripts/run_omni7b.sh              # ceb then pam
#
# Recipe A, measured by profile_omni7b.sh on 2026-09-20: fp32 weights with
# bitsandbytes 8-bit Adam, batch 4 x accum 4, peaks at 74.5 GiB of 95. fp32
# weights rather than bf16 because small updates round away in bf16, and they
# fit. Same 5000 steps, effective batch 16 and 25k clips as the bake-off.
#
# --eval-steps 1000, not the default 250: a checkpoint here is ~42 GB with its
# optimiser state, and writing one every 250 steps would spend more time on the
# disk than on the GPU. Finished runs drop their checkpoints for the same
# reason — final/ is the artefact.
set -uo pipefail
cd "$(dirname "$0")/.."

LANGS=${*:-ceb pam}
RUNS=${FINETUNE_DIR:-/mnt/data/finetune_runs}
SUMMARY="$RUNS/omni7b_summary.txt"
echo "=== omni-7b start $(date -u +%F' '%T) langs='$LANGS'" >> "$SUMMARY"

for lang in $LANGS; do
    name="ctc_omni-7b_char_pld_${lang}"
    if [ -f "$RUNS/$name/result.json" ]; then
        echo "$name: done already, skipping" | tee -a "$SUMMARY"; continue
    fi
    echo "--- $name start $(date -u +%T)" | tee -a "$SUMMARY"
    venv/bin/python3 finetune_ctc.py --encoder omni-7b --language "$lang" --units char \
        --max-samples 25000 --max-steps 5000 --batch-size 4 --grad-accum 4 \
        --optim adamw_bnb_8bit --eval-steps 1000 \
        --num-proc 8 --dataloader-workers 8 --resume \
        2>&1 | tr '\r' '\n' | grep -vE 'examples/s|it/s\]$'

    if [ -f "$RUNS/$name/result.json" ]; then
        echo "$name: OK $(date -u +%T) $(python3 -c "import json;d=json.load(open('$RUNS/$name/result.json'));print(f\"CER {d.get('cer',0)*100:.2f} WER {d.get('wer',0)*100:.2f}\")")" | tee -a "$SUMMARY"
        rm -rf "$RUNS/$name"/checkpoint-*
    else
        echo "$name: FAILED $(date -u +%T)" | tee -a "$SUMMARY"
    fi
done
echo "=== omni-7b done $(date -u +%F' '%T)" >> "$SUMMARY"
