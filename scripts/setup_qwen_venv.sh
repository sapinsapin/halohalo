#!/usr/bin/env bash
# A separate venv for Qwen3-TTS.
#
#   bash scripts/setup_qwen_venv.sh
#
# Upstream asks for a fresh environment, and they are right to: `qwen-tts`
# pulls its own pinned dependency set, and the training venv on this VM holds a
# working transformers 5.17 + torch 2.11+cu128 pair that took a bootstrap to
# get right (Blackwell has no kernels in the cu126 build). Installing qwen-tts
# over it risks the whole training stack mid-run, for a package only two
# scripts need.
#
# Idempotent: re-running upgrades in place.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3.12}
QVENV=${QVENV:-venv_qwen}

[ -d "$QVENV" ] || "$PY" -m venv "$QVENV"
"$QVENV/bin/pip" install -q --upgrade pip wheel
# No flash-attn: it wants a long compile and sdpa is enough for codec encoding,
# which is what this venv is for. The SFT can ask for it later if it pays off.
"$QVENV/bin/pip" install -q --upgrade qwen-tts

"$QVENV/bin/python3" - <<'PY'
import torch
from qwen_tts import Qwen3TTSTokenizer          # noqa: F401
print("qwen-tts ok | torch", torch.__version__, "| cuda", torch.cuda.is_available())
PY
