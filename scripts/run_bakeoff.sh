#!/usr/bin/env bash
# R1 foundation bake-off + R2 output-unit ablation, one VM, one GPU.
#
#   bash scripts/bootstrap_nebius.sh          # once per VM
#   tmux new -s bakeoff
#   bash scripts/run_bakeoff.sh
#
# Runs every candidate whose stack exists today, on the two languages the plan
# picked: Cebuano (worst BPE fertility, decent text) and Kapampangan (weakest
# published CER, 5.0 chars/word). Each run resumes from its own checkpoint, so
# a preemption costs minutes, not the run. Re-running the script skips
# anything already finished.
#
# Not covered here, because their stacks are not built:
#   - SeamlessM4T-v2 encoder + NLLB decoder (needs its own trainer; CC-BY-NC)
#   - MMS-1b-all adapters (CC-BY-NC)
# Both are research-track-only anyway; everything below is permissive.
set -u
cd "$(dirname "$0")/.."

if [ -f ts-wandb.txt ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < ts-wandb.txt)"
    export WANDB_PROJECT="${WANDB_PROJECT:-halohalo-bakeoff}"
fi

LANGS=(${BAKEOFF_LANGS:-ceb pam})
STEPS=${BAKEOFF_STEPS:-5000}
SAMPLES=${BAKEOFF_SAMPLES:-25000}
# Effective batch is BATCH x ACCUM and must stay 16 across every arm, or the
# bake-off compares optimisation settings instead of models. Spend the card's
# memory on a bigger step instead of on accumulation: on the 96 GB RTX PRO 6000
# the 8x2 default used 13 GB and left the GPU launch-bound.
BATCH=${BAKEOFF_BATCH:-16}
ACCUM=${BAKEOFF_ACCUM:-1}
# 24 vCPUs sit idle while the GPU waits for single-process feature extraction
NUM_PROC=${BAKEOFF_NUM_PROC:-8}
LOADERS=${BAKEOFF_LOADERS:-8}
# gradient checkpointing recomputes activations to save memory we are not short
# of; empty this to turn it back on for a small card
NOCKPT=${BAKEOFF_NOCKPT:---no-grad-checkpoint}
ATTN=${BAKEOFF_ATTN:-sdpa}
# torch.compile pays on Whisper's fixed 30 s mel; the CTC arms feed
# variable-length audio, where it recompiles instead. Off by default there.
COMPILE=${BAKEOFF_COMPILE:-}
W_COMPILE=${BAKEOFF_W_COMPILE:---compile}
# generative eval over the full test set cost ~8 min every 500 steps
W_EVAL_SAMPLES=${BAKEOFF_W_EVAL_SAMPLES:-500}
W_EVAL_STEPS=${BAKEOFF_W_EVAL_STEPS:-1000}
W_GEN_LEN=${BAKEOFF_W_GEN_LEN:-128}
FINETUNE_DIR=${FINETUNE_DIR:-$PWD/finetune_runs}
LOGDIR="$FINETUNE_DIR/logs"
mkdir -p "$LOGDIR"
SUMMARY="$FINETUNE_DIR/bakeoff_summary.txt"

# candidate := "<encoder> <units>"; whisper is handled separately since it is
# a seq2seq trainer, not a CTC one.
# Cloud profile by default. For the local 8 GB card:
#   BAKEOFF_CTC="omni-300m:char omni-300m:syllable" \
#   BAKEOFF_WHISPER=openai/whisper-small FINETUNE_DIR=$PWD/finetune_runs_frozen \
#   bash scripts/run_bakeoff.sh
if [ -n "${BAKEOFF_CTC:-}" ]; then
    CTC_CANDIDATES=()
    for c in $BAKEOFF_CTC; do CTC_CANDIDATES+=("${c%%:*} ${c##*:}"); done
else
    CTC_CANDIDATES=(
        "omni-1b char"
        "omni-1b syllable"
        "w2v-bert char"
        "w2v-bert syllable"
    )
fi
WHISPER=${BAKEOFF_WHISPER:-openai/whisper-large-v3}
# same effective batch of 16 either way. large-v3 at 2x8 was the 8 GB-card
# recipe: on 96 GB it fits the full 16 in one step, with no accumulation
case "$WHISPER" in
    *large*) W_BATCH=${BAKEOFF_W_BATCH:-16}; W_ACCUM=${BAKEOFF_W_ACCUM:-1} ;;
    *)       W_BATCH=8; W_ACCUM=2 ;;
esac
export FINETUNE_DIR

# Keep the datasets cache beside the runs, not on the system disk: locally the
# default lives inside WSL's ext4.vhdx on a nearly full C:, and on a cloud VM
# it would fill the boot disk rather than the persistent one. TMPDIR stays
# where it is: DataLoader workers put Unix sockets there, and the /mnt/d
# (drvfs) mount rejects them with "Errno 95 Operation not supported".
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$FINETUNE_DIR/.cache/datasets}
mkdir -p "$HF_DATASETS_CACHE"

echo "=== bake-off start $(date -u +%F' '%H:%M:%S) ===" | tee -a "$SUMMARY"
echo "languages: ${LANGS[*]} | steps: $STEPS | samples: $SAMPLES" | tee -a "$SUMMARY"

for lang in "${LANGS[@]}"; do
    # the frozen speaker+prompt-disjoint spec must exist, or every number
    # below is an in-domain number and the bake-off proves nothing
    if [ ! -f "${PLD_SPLIT_DIR:-splits}/pld_${lang}.json" ]; then
        echo "!! no frozen split for $lang — run: python -m halolib.splits --languages $lang" | tee -a "$SUMMARY"
        continue
    fi

    for cand in "${CTC_CANDIDATES[@]}"; do
        set -- $cand
        enc=$1; units=$2
        run="ctc_${enc}_${units}_pld_${lang}"
        if [ -f "$FINETUNE_DIR/$run/result.json" ]; then
            echo "$run: already done, skipping" | tee -a "$SUMMARY"
            continue
        fi
        echo "--- $run start $(date -u +%H:%M:%S) ---" | tee -a "$SUMMARY"
        if venv/bin/python3 finetune_ctc.py \
                --encoder "$enc" --units "$units" \
                --dataset pld --language "$lang" \
                --max-samples "$SAMPLES" --max-steps "$STEPS" \
                --batch-size "$BATCH" --grad-accum "$ACCUM" --resume \
                --num-proc "$NUM_PROC" --dataloader-workers "$LOADERS" $NOCKPT \
                --attn "$ATTN" $COMPILE \
                >> "$LOGDIR/$run.log" 2>&1; then
            echo "$run: OK $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
        else
            echo "$run: FAILED (see $LOGDIR/$run.log)" | tee -a "$SUMMARY"
        fi
    done

    # the Whisper arm: the bar every CTC candidate has to clear, same split
    wrun="asr_pld_${lang}"
    wname=$(basename "$WHISPER")
    if [ ! -f "$FINETUNE_DIR/$wrun/result.json" ]; then
        echo "--- $wname $lang start $(date -u +%H:%M:%S) ---" | tee -a "$SUMMARY"
        if venv/bin/python3 finetune_asr.py \
                --model "$WHISPER" \
                --dataset pld --language "$lang" \
                --max-samples "$SAMPLES" --max-steps "$STEPS" \
                --batch-size "$W_BATCH" --grad-accum "$W_ACCUM" --resume \
                --num-proc "$NUM_PROC" --dataloader-workers "$LOADERS" $NOCKPT \
                --attn "$ATTN" $W_COMPILE \
                --eval-samples "$W_EVAL_SAMPLES" --eval-steps "$W_EVAL_STEPS" \
                --gen-max-length "$W_GEN_LEN" \
                >> "$LOGDIR/${wname}_$lang.log" 2>&1; then
            echo "$wname $lang: OK $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
        else
            echo "$wname $lang: FAILED (see $LOGDIR/${wname}_$lang.log)" | tee -a "$SUMMARY"
        fi
    else
        echo "$wname $lang: already done, skipping" | tee -a "$SUMMARY"
    fi
done

echo "=== bake-off done $(date -u +%F' '%H:%M:%S) ===" | tee -a "$SUMMARY"
venv/bin/python3 scripts/bakeoff_report.py | tee -a "$SUMMARY"
