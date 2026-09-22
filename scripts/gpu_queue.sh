#!/usr/bin/env bash
# Keep the cloud card busy once the current jobs finish.
#
#   bash scripts/gpu_queue.sh            # waits for named sessions, then works
#
# Everything here is decision-independent: none of it presumes which text
# frontend P1 picks, so it can run before that result lands without baking an
# unmeasured choice into anything.
#
#   1. Qwen3-TTS codec encoding for whatever export_qwen_tts.py has written.
#      Upstream's prepare_data.py, in its own venv (see scripts/setup_qwen_venv.sh)
#      because qwen-tts pins a dependency set we do not want anywhere near the
#      training venv.
#   2. SNAC caches for the eight languages P2 has not touched. The cache is
#      keyed by audio, so every frontend and every arm reuses it: doing this now
#      means the fleet run starts at step 0 whichever frontend wins.
set -uo pipefail
cd "$(dirname "$0")/.."

WORK=${PLD_WORK_DIR:-/mnt/data/pld_shards}
QVENV=${QVENV:-venv_qwen}
REST=${REST:-bcl eng fil hil ilo pag tsg war}
SAMPLES=${SAMPLES:-20000}

for s in tts_pam tts_score2; do
    while tmux has-session -t "$s" 2>/dev/null; do
        sleep 60
    done
    echo "=== $s finished"
done

echo "=== 1. Qwen codec encoding"
if [ -x "$QVENV/bin/python3" ]; then
    for jsonl in "$WORK"/qwen_tts/*/*.jsonl; do
        case "$jsonl" in *_coded.jsonl) continue;; esac
        [ -e "$jsonl" ] || continue
        coded="${jsonl%.jsonl}_coded.jsonl"
        if [ -s "$coded" ]; then
            echo "  $(basename "$coded") exists, skipping"
            continue
        fi
        echo "  encoding $(basename "$jsonl")"
        "$QVENV/bin/python3" third_party/qwen3_tts_finetuning/prepare_data.py \
            --input_jsonl "$jsonl" --output_jsonl "$coded" \
            --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
            --device cuda:0 || echo "  !! failed on $(basename "$jsonl")"
    done
else
    echo "  no $QVENV — run scripts/setup_qwen_venv.sh first; skipping"
fi

echo "=== 2. SNAC caches for the remaining languages"
for lang in $REST; do
    echo "--- $lang"
    # --max-samples matters: the cache is keyed by corpus size so a 2k
    # cache cannot masquerade as a full one, which means a prewarm at the
    # trainer's 2000 default is one the fleet will never read. Match it.
    venv/bin/python3 finetune_orpheus.py --cache-only --cloud \
        --dataset pld --language "$lang" --max-samples "$SAMPLES" \
        2>&1 | grep -vE 'examples/s|it/s'
done

echo "=== queue done $(date -u +%F' '%T)"
