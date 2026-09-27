#!/usr/bin/env bash
# The Apple targets, run on a Mac: MLX on Metal, Core ML on the Neural Engine
# and GPU, whisper.cpp with Metal and with its Core ML encoder, SNAC in Core
# ML, and Orpheus through llama.cpp on Metal (optionally MLX too).
#
# Unpack the bundle made by scripts/port_bundle_mac.sh, then from its folder:
#
#   bash scripts/port_mac.sh setup        # ~10-15 min: venv, whisper.cpp + llama.cpp, .mlmodelc
#   bash scripts/port_mac.sh validate     # ~1 h; downloads PyTorch weights for the checks (public repos)
#   bash scripts/port_mac.sh orpheus-mlx  # optional: merge Orpheus here, MLX 4-bit, end-to-end check (16 GB+ RAM)
#   bash scripts/port_mac.sh report       # docs/porting_report.md, Mac rows marked "measured on the mac"
#   bash scripts/port_mac.sh pack         # mac-results.tar: the results to copy back
#
# Needs Apple silicon, the Xcode command line tools (xcode-select --install),
# and Homebrew's Python and cmake (brew install python@3.12 cmake).
# Every step skips what is done and keeps going past a failure; the log is
# finetune_runs/port/mac.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export FINETUNE_DIR="$PWD/finetune_runs" PORT_TOOLS="$PWD/third_party" PYTHONPATH="$PWD" \
       PORT_HOST=mac PORT_THREADS=${PORT_THREADS:-8} TOKENIZERS_PARALLELISM=false
PY=${PY:-python3.12}
V=venv_mac
LOG=finetune_runs/port/mac.log
W=sapinsapin/whisper-small-pld-ceb
C=sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm
O=sapinsapin/orpheus-3b-0.1-pretrained-char-pld-ceb
G=finetune_runs/port/artefacts/whisper-small-pld-ceb
mkdir -p finetune_runs/port
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
run() { say "$*"; "$@" >> "$LOG" 2>&1 && tail -1 "$LOG" | cut -c1-160 || say "  failed (see $LOG)"; }

setup() {
  [ "$(uname -s)" = Darwin ] || { echo "this is the Mac half of the pipeline"; exit 1; }
  [ "$(uname -m)" = arm64 ] || say "warning: not Apple silicon; MLX and the Neural Engine need it"
  command -v cmake > /dev/null || { echo "brew install cmake"; exit 1; }
  command -v "$PY" > /dev/null || { echo "brew install python@3.12 (or PY=python3.x)"; exit 1; }
  [ -x $V/bin/python3 ] || "$PY" -m venv $V
  $V/bin/pip install -q --upgrade pip
  $V/bin/pip install -q mlx mlx-whisper mlx-lm coremltools torch transformers peft accelerate snac \
      soundfile scipy jiwer huggingface_hub safetensors numpy python-dotenv datasets
  mkdir -p third_party
  # whisper.cpp: Metal is on by default on macOS; Core ML encoder support on,
  # with fallback so models without a compiled encoder still run on Metal
  [ -d third_party/whisper.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/whisper.cpp third_party/whisper.cpp
  cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DWHISPER_COREML=1 -DWHISPER_COREML_ALLOW_FALLBACK=1 -DWHISPER_BUILD_TESTS=OFF >> "$LOG"
  cmake --build third_party/whisper.cpp/build -j --config Release >> "$LOG"
  [ -d third_party/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
  cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >> "$LOG"
  cmake --build third_party/llama.cpp/build -j --config Release --target llama-server llama-quantize >> "$LOG"
  # whisper.cpp's Core ML mode loads <model>-encoder.mlmodelc beside <model>.bin
  if [ ! -d $G/ggml/ggml-model-f16-coreml-encoder.mlmodelc ]; then
    xcrun coremlc compile $G/coreml/encoder.mlpackage $G/coreml/ >> "$LOG"
    ln -sf ggml-model-f16.bin $G/ggml/ggml-model-f16-coreml.bin
    cp -R $G/coreml/encoder.mlmodelc $G/ggml/ggml-model-f16-coreml-encoder.mlmodelc
  fi
  say "setup done: $($V/bin/python3 -c 'import mlx.core as mx, coremltools as ct, torch; print("mlx", mx.__version__, "| coremltools", ct.__version__, "| torch", torch.__version__, "| metal", mx.metal.is_available())')"
}

validate() {
  local P="$V/bin/python3 -m porting.validate"
  # MLX on Metal: the full evalpack, now at real speed
  for v in fp16 q8 q4; do run $P $W --runtime mlx --variant $v; done
  # Core ML: Neural Engine where it fits, against CPU only
  for v in all cpu; do run $P $W --runtime coreml --variant $v; done
  for v in all cpu; do run $P $C --runtime coreml --variant $v; done
  # whisper.cpp: Metal, and Metal + Core ML encoder
  for v in f16 q5_0 f16-coreml; do run $P $W --runtime whispercpp --variant $v; done
  # SNAC decoder in Core ML
  for v in coreml coreml-cpu; do run $V/bin/python3 -m porting.validate_snac hubertsiuzdak/snac_24khz --variant $v --n 20; done
  # Orpheus Cebuano, 4-bit GGUF on llama.cpp with Metal, judged by MMS-1b-all
  run $V/bin/python3 -m porting.validate_orpheus $O --runtime llamacpp --variant q4_k_m --n 10
}

orpheus_mlx() {
  local R=finetune_runs/port/artefacts/orpheus-3b-0.1-pretrained-char-pld-ceb
  [ -f $R/merged/config.json ] || run $V/bin/python3 -m porting.export_orpheus $O --steps merge
  [ -f $R/mlx/q4/config.json ] || run $V/bin/python3 -m porting.export_orpheus $O --steps mlx
  run $V/bin/python3 -m porting.validate_orpheus $O --runtime mlx --variant q4 --n 10
}

report() { run $V/bin/python3 -m porting report; say "open docs/porting_report.md"; }

pack() {
  tar -cf mac-results.tar $(cd . && ls finetune_runs/port/results/*/*@mac*.json) "$LOG"
  say "mac-results.tar: copy it back and untar it in the repo root on the workstation, then: python3 -m porting report"
}

case "${1:-}" in
  setup) setup ;;
  validate) validate ;;
  orpheus-mlx) orpheus_mlx ;;
  report) report ;;
  pack) pack ;;
  all) setup; validate; report; pack ;;
  *) sed -n 2,18p "$0"; exit 2 ;;
esac
