#!/usr/bin/env bash
# One venv per porting toolchain, all on D:.
#
#   bash porting/setup_venvs.sh onnx          # ONNX export, ORT quantisation, OpenVINO
#   bash porting/setup_venvs.sh executorch    # ExecuTorch (XNNPACK; CoreML/MPS need macOS)
#   bash porting/setup_venvs.sh mlx           # MLX (Linux CPU wheel; Metal needs macOS)
#   bash porting/setup_venvs.sh coreml        # coremltools (converts on Linux, runs on macOS)
#
# Why separate venvs: each toolchain pins its own torch or transformers
# (ExecuTorch releases pin an exact torch; optimum-onnx a transformers range),
# and the training venv must never be disturbed — a queue may be running in it.
# Why D:: WSL's own disk lives on a nearly full C:, so venvs and the pip cache
# written under ~ would grow C:. A full C: has crashed WSL before.
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/mnt/d/pip_cache} TMPDIR=${TMPDIR:-/mnt/d/tmp}
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
PY=${PY:-python3.12}
CPU_TORCH="--index-url https://download.pytorch.org/whl/cpu"

mk() { [ -x "$1/bin/python3" ] || "$PY" -m venv "$1"; "$1/bin/python3" -m pip install -q --upgrade pip wheel; }

case "${1:?toolchain: onnx|executorch|mlx|coreml}" in
  onnx)
    V=venv_port_onnx; mk $V
    # CPU torch is enough to export; the artefacts are what run on the targets
    $V/bin/pip install -q torch $CPU_TORCH
    $V/bin/pip install -q "optimum-onnx[onnxruntime]" onnx onnxscript onnxruntime \
        soundfile jiwer huggingface_hub safetensors openvino
    ;;
  executorch)
    V=venv_port_et; mk $V
    $V/bin/pip install -q executorch   # pulls the torch it was built against
    $V/bin/pip install -q transformers soundfile jiwer huggingface_hub
    ;;
  mlx)
    V=venv_port_mlx; mk $V
    $V/bin/pip install -q "mlx[cpu]" mlx-whisper soundfile jiwer huggingface_hub safetensors
    $V/bin/pip install -q torch $CPU_TORCH   # only to read the PyTorch weights
    ;;
  coreml)
    V=venv_port_coreml; mk $V
    $V/bin/pip install -q coremltools torch $CPU_TORCH
    $V/bin/pip install -q transformers huggingface_hub
    ;;
  *) echo "unknown toolchain $1"; exit 2 ;;
esac
"$V/bin/python3" -m pip freeze | grep -iE "^(torch|transformers|onnx|onnxruntime|optimum|optimum-onnx|executorch|mlx|coremltools|openvino)==" || true
echo "ok: $V"
