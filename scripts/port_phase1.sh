#!/usr/bin/env bash
# Porting pipeline, Phase 1: everything the workstation can convert and check.
# CPU only (CUDA hidden, niced), so it runs beside the local GPU queue.
#
# Launch from Windows so a wsl.exe client lives as long as the job — a job
# detached inside WSL dies when the last client exits:
#   Start-Process -WindowStyle Hidden wsl.exe -ArgumentList 'bash -c "cd /mnt/d/halohalo && bash scripts/port_phase1.sh >> finetune_runs/port/phase1.log 2>&1"'
#
#   MODELS=demo  (default) one model per family, fully validated
#   MODELS=phase every Phase 1 model; siblings get a SMOKE-clip check
#   MODELS=sapinsapin/whisper-small-pld-pam,...  a list
#
# Resumable: finished builds and validations are skipped on a rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES= PIP_CACHE_DIR=/mnt/d/pip_cache TMPDIR=/mnt/d/tmp \
       npm_config_cache=/mnt/d/npm_cache HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
       PORT_THREADS=${PORT_THREADS:-4}
mkdir -p "$TMPDIR"
MODELS=${MODELS:-demo}
N=${N:-100}; SMOKE=${SMOKE:-20}
PY=python3                          # the orchestrator is stdlib-only
say() { echo "$(date '+%F %T') $*"; }

say "=== porting phase 1 (models: $MODELS)"
[ -f finetune_runs/port/evalpack/ceb.npz ] || venv/bin/python3 -m porting.evalpack --languages ceb pam --n "$N"
for t in onnx openvino executorch mlx coreml; do
  v=$(case $t in onnx) echo venv_port_onnx;; openvino) echo venv_port_ov;; executorch) echo venv_port_et;;
                    mlx) echo venv_port_mlx;; coreml) echo venv_port_coreml;; esac)
  [ -f "$v/.halohalo_ok" ] || { say "installing $t"; bash porting/setup_venvs.sh "$t"; }
done
[ -x /mnt/d/halohalo/third_party/whisper.cpp/build/bin/whisper-cli ] || bash porting/setup_venvs.sh ggml
[ -d porting/web/node_modules/@huggingface/transformers ] || bash porting/setup_venvs.sh web

# Builds on BUILD_CPUS only (default four cores): ONNX Runtime and torch size
# their thread pools from the affinity mask, so this caps load and heat. The
# workstation has been throwing CPU machine-check exceptions under sustained
# all-core load.
say "--- build";    nice -n 10 taskset -c "${BUILD_CPUS:-0-3}" $PY -m porting build --phase 1 --models "$MODELS"
say "--- validate"; nice -n 10 $PY -m porting validate --phase 1 --models "$MODELS" --n "$N" --smoke "$SMOKE"
say "--- report";   $PY -m porting report
say "=== done"
