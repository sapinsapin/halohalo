#!/usr/bin/env bash
# Bring a fresh Nebius VM to the point where any job in docs/pld_sota_track.md
# §7 can run. Idempotent: safe to re-run after a preemption.
#
#   ssh nebius-vm
#   git clone git@github.com:sapinsapin/halohalo.git && cd halohalo
#   bash scripts/bootstrap_nebius.sh
#
# Secrets are NOT in this script and never in the repo. Before running, copy
# these two files up by hand (scp from the workstation):
#     .env            HF_TOKEN=... plus the paths below
#     ts-wandb.txt    the W&B key, one line
#
# Assumes: Ubuntu 22.04/24.04, one NVIDIA GPU, a persistent disk mounted at
# $DATA_ROOT that survives the VM (Nebius: attach before first boot).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-/mnt/data}
PY=${PY:-python3.12}
# cu128 is the oldest build with Blackwell (RTX PRO 6000, B200) kernels
TORCH_CUDA=${TORCH_CUDA:-cu128}
TORCH_VER=${TORCH_VER:-2.11.0}

say() { printf '\n=== %s ===\n' "$*"; }

say "system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg git git-lfs tmux htop "$PY-venv" "$PY-dev" build-essential
git lfs install --skip-repo || true

say "persistent disk at $DATA_ROOT"
if [ ! -d "$DATA_ROOT" ]; then
    echo "!! $DATA_ROOT does not exist. Attach and mount the persistent disk first."
    echo "   Everything below will land on the ephemeral root disk and die with the VM."
    exit 1
fi
sudo mkdir -p "$DATA_ROOT"/{hf_cache,finetune_runs,pld_shards}
sudo chown -R "$USER" "$DATA_ROOT"
df -h "$DATA_ROOT" | tail -1

say "python env"
[ -d venv ] || "$PY" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q --upgrade pip wheel
pip install -q -r requirements.txt
# training stack not in requirements.txt (that file covers the data pipeline)
pip install -q transformers accelerate peft bitsandbytes wandb speechbrain \
    sentencepiece evaluate
# TTS: the SNAC codec for the Orpheus arm. Missing it cost a cloud run on
# 2026-09-19, found only when the trainer reached its first import.
pip install -q snac
# torch and torchaudio must be a version-matched pair from the same CUDA index.
# Pinned last: silero-vad caps torchaudio<2.10 and would otherwise downgrade the
# pair on a fresh VM (but not on a re-run), so two VMs ended up on different
# torch. silero-vad is only used by the livestream pipeline, never in training.
pip install -q "torch==${TORCH_VER}+${TORCH_CUDA}" "torchaudio==${TORCH_VER}+${TORCH_CUDA}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__)
PY

say "secrets"
missing=0
[ -f .env ] || { echo "!! .env missing — scp it up"; missing=1; }
[ -f ts-wandb.txt ] || { echo "!! ts-wandb.txt missing — W&B logging will be off"; }
grep -q '^HF_TOKEN=' .env 2>/dev/null || { echo "!! HF_TOKEN not set in .env"; missing=1; }
[ "$missing" -eq 0 ] || exit 1
# point caches at the persistent disk regardless of what .env says: drop any
# workstation paths first (sort -u keeps the first of a key, not the last)
sed -i -E '/^(HF_HOME|HF_XET_CACHE|FINETUNE_DIR|PLD_WORK_DIR|PLD_DIR|PLD_SOURCE)=/d' .env
{
  echo "HF_HOME=$DATA_ROOT/hf_cache"
  echo "FINETUNE_DIR=$DATA_ROOT/finetune_runs"
  echo "PLD_WORK_DIR=$DATA_ROOT/pld_shards"
  echo "PLD_SOURCE=hub"        # the raw corpus is not on this machine
} >> .env
sort -u -t= -k1,1 .env -o .env
export HF_HOME="$DATA_ROOT/hf_cache"
# huggingface-cli is gone from current huggingface_hub; `hf` reads HF_TOKEN from
# the environment, so there is no need to write the token to ~/.cache
HF_TOKEN="$(grep '^HF_TOKEN=' .env | cut -d= -f2-)" venv/bin/hf auth whoami

say "data"
# PLD is ~25 GB and private; FSC ~7 GB. Pull once onto the persistent disk so
# preemption does not re-download. Skips whatever is already cached.
venv/bin/python3 - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv(".env")
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
for repo in ("sapinsapin/pld", "sapinsapin/filipinospeechcorpus"):
    print("fetching", repo, flush=True)
    snapshot_download(repo, repo_type="dataset", token=tok,
                      max_workers=8, allow_patterns=["*.parquet", "*.md"])
print("done")
PY
du -sh "$DATA_ROOT/hf_cache" || true

say "ready"
# scripts/nebius/run.sh skips the bootstrap when this matches the script's hash
sha256sum "$0" | cut -d' ' -f1 > "$DATA_ROOT/.bootstrapped"
cat <<'EOF'
Start a job inside tmux so a dropped ssh does not kill it:

    tmux new -s r1
    source venv/bin/activate
    python finetune_ctc.py --encoder omni-1b --language ceb --units char --push

Preemption: every trainer checkpoints every 250 steps into $FINETUNE_DIR.
Re-running the same command resumes from the last checkpoint.
EOF
