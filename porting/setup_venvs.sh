#!/usr/bin/env bash
# One environment per porting toolchain, all on D:.
#
#   bash porting/setup_venvs.sh onnx        # ONNX export + ORT quantisation (Arm CPU, web, NPU QDQ, AMD)
#   bash porting/setup_venvs.sh openvino    # optimum-intel + NNCF (Intel CPU/iGPU/NPU)
#   bash porting/setup_venvs.sh executorch  # optimum-executorch, XNNPACK (Android/iOS CPU)
#   bash porting/setup_venvs.sh mlx         # MLX Linux CPU wheel (Metal needs macOS)
#   bash porting/setup_venvs.sh coreml      # coremltools (converts on Linux, runs on macOS)
#   bash porting/setup_venvs.sh ggml        # clone + build whisper.cpp (needs the onnx venv for cmake)
#   bash porting/setup_venvs.sh web         # Transformers.js + onnxruntime-node for Node validation
#   bash porting/setup_venvs.sh all
#
# Why separate venvs: each toolchain pins its own torch or transformers
# (ExecuTorch pins an exact torch; optimum-onnx and optimum-intel each a
# transformers range), and the training venv must never be disturbed — a
# queue may be running in it. Torch is always the CPU build: exporting needs
# no GPU, and the CUDA wheels are 3 GB each.
# Why D:: WSL's own disk lives on a nearly full C:, so venvs, pip and npm
# caches written under ~ would grow C:. Install one toolchain at a time: D: is
# reached through a single 9P pipe, and parallel installs stall each other.
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/mnt/d/pip_cache} TMPDIR=${TMPDIR:-/mnt/d/tmp}
export npm_config_cache=${npm_config_cache:-/mnt/d/npm_cache}
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$npm_config_cache"
PY=${PY:-python3.12}
CPU="https://download.pytorch.org/whl/cpu"
TOOLS=${PORT_TOOLS:-/mnt/d/halohalo/third_party}
COMMON="soundfile jiwer huggingface_hub safetensors numpy"

mk() { [ -x "$1/bin/python3" ] || "$PY" -m venv "$1"; "$1/bin/python3" -m pip install -q --upgrade pip wheel; }
torch_cpu() { "$1/bin/pip" install -q torch --index-url "$CPU"; }
show() { "$1/bin/python3" -m pip freeze | grep -iE "^(torch|transformers|onnx|onnxruntime|optimum|optimum-onnx|optimum-intel|optimum-executorch|executorch|mlx|mlx-whisper|coremltools|openvino|nncf|snac)==" || true; }

one() {
  case "$1" in
    onnx)
      V=venv_port_onnx; mk $V; torch_cpu $V
      # accelerate lets the exporter de-duplicate tied weights (Whisper's output head)
      $V/bin/pip install -q "optimum-onnx[onnxruntime]" onnx onnxscript onnxruntime snac cmake accelerate \
          $COMMON --extra-index-url "$CPU"
      show $V ;;
    openvino)
      V=venv_port_ov; mk $V; torch_cpu $V
      $V/bin/pip install -q "optimum-intel[openvino]" nncf $COMMON --extra-index-url "$CPU"
      show $V ;;
    executorch)
      V=venv_port_et; mk $V; torch_cpu $V
      # --extra-index-url keeps any torch ExecuTorch pins on the CPU build
      $V/bin/pip install -q optimum-executorch $COMMON --extra-index-url "$CPU"
      show $V ;;
    mlx)
      V=venv_port_mlx; mk $V; torch_cpu $V         # torch only to read the checkpoint
      $V/bin/pip install -q "mlx[cpu]" mlx-whisper $COMMON --extra-index-url "$CPU"
      show $V ;;
    coreml)
      V=venv_port_coreml; mk $V; torch_cpu $V
      # openai-whisper: whisper.cpp's Core ML encoder converter loads OpenAI's model class
      $V/bin/pip install -q coremltools transformers openai-whisper snac $COMMON --extra-index-url "$CPU"
      show $V ;;
    ggml)
      mkdir -p "$TOOLS"
      [ -d "$TOOLS/whisper.cpp" ] || git clone -q --depth 1 https://github.com/ggml-org/whisper.cpp "$TOOLS/whisper.cpp"
      [ -d "$TOOLS/whisper" ] || git clone -q --depth 1 https://github.com/openai/whisper "$TOOLS/whisper"
      CM=venv_port_onnx/bin/cmake
      # CPU build, AVX2 on this machine; the same source builds NEON/KleidiAI on Arm
      $CM -S "$TOOLS/whisper.cpp" -B "$TOOLS/whisper.cpp/build" -DCMAKE_BUILD_TYPE=Release \
          -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF > /dev/null
      $CM --build "$TOOLS/whisper.cpp/build" -j "$(nproc)" --config Release > /dev/null
      ls "$TOOLS/whisper.cpp/build/bin" ;;
    web)
      (cd porting/web && npm install --no-audit --no-fund --loglevel=error)
      for p in @huggingface/transformers onnxruntime-node onnxruntime-web; do
        [ -f "porting/web/node_modules/$p/package.json" ] && \
          grep -m1 '"version"' "porting/web/node_modules/$p/package.json" | sed "s|^ *|$p |"
      done ;;
    *) echo "unknown toolchain $1"; exit 2 ;;
  esac
  # a venv that exists is not a venv that installed: a crash mid-install
  # leaves python3 in place, so callers check this marker instead
  [ -n "${V:-}" ] && touch "$V/.halohalo_ok"
  echo "ok: $1"
}

if [ "${1:?toolchain: onnx|openvino|executorch|mlx|coreml|ggml|web|all}" = all ]; then
  for t in onnx openvino executorch mlx coreml ggml web; do one "$t"; done
else
  one "$1"
fi
