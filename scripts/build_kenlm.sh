#!/usr/bin/env bash
# Build word n-gram LMs for D1 of docs/asr_decoder_plan.md. CPU only.
#
#   bash scripts/build_kenlm.sh ceb pam          # -> $PLD_WORK_DIR/lm/<lang>.arpa
#
# Runs on the cloud VM because lmplz needs cmake and Boost to compile, and the
# workstation has neither and a C: drive with no room to spare. The output is a
# few MB and is copied down for decoding on the 3070.
#
# The text is the frozen **train** split's transcripts and nothing else. That
# split is prompt-disjoint from test by construction, so this LM cannot have
# read a test sentence. External text (the FineWeb-2 ingests) would make a
# better LM and is deliberately left for a second pass, because PLD prompts are
# read sentences that may exist on the web: it needs an n-gram overlap check
# against the test prompts first, or WER turns into a memory test.
set -euo pipefail
cd "$(dirname "$0")/.."

LANGS=${*:-ceb pam}
ORDER=${ORDER:-4}
WORK=${PLD_WORK_DIR:-/mnt/data/pld_shards}
KENLM=${KENLM:-/mnt/data/kenlm}
mkdir -p "$WORK/lm"

if [ ! -x "$KENLM/build/bin/lmplz" ]; then
    sudo apt-get install -y -qq cmake libboost-program-options-dev \
        libboost-system-dev libboost-thread-dev libboost-test-dev \
        libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
    [ -d "$KENLM" ] || git clone -q --depth 1 https://github.com/kpu/kenlm "$KENLM"
    cmake -S "$KENLM" -B "$KENLM/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build "$KENLM/build" -j 8 --target lmplz > /dev/null
fi

for lang in $LANGS; do
    txt="$WORK/lm/${lang}_train.txt"
    venv/bin/python3 - "$lang" "$txt" <<'PY'
import os, sys
sys.path.insert(0, ".")
from halolib.finetune import load_speech_dataset
lang, out = sys.argv[1], sys.argv[2]
ds = load_speech_dataset("pld", task="asr", language=lang, max_samples=None,
                         token=os.environ.get("HF_TOKEN"))
# same normalisation the scorer applies: lowercase, single spaces
lines = sorted({" ".join(t.lower().split()) for t in ds["train"]["text"]})
open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"{lang}: {len(lines)} distinct train sentences")
PY
    # --discount_fallback: a corpus this small can leave an order with too few
    # distinct counts for Kneser-Ney's discounts to be estimated
    "$KENLM/build/bin/lmplz" -o "$ORDER" --discount_fallback \
        < "$txt" > "$WORK/lm/${lang}.arpa" 2> "$WORK/lm/${lang}.lmplz.log"
    echo "$lang: $(du -h "$WORK/lm/${lang}.arpa" | cut -f1) $WORK/lm/${lang}.arpa"
done
