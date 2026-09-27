#!/usr/bin/env bash
# The Apple half of the porting pipeline, run on a Mac (Apple silicon).
# Spec: docs/prd_mac_porting.md.
#
# From a clone of this repo, after untarring the evaluation pack
# (halohalo-mac-eval.tar, made on the workstation by
# `bash scripts/port_bundle_mac.sh --eval-only`) in the repo root:
#
#   bash scripts/port_mac.sh setup       # venvs, whisper.cpp (Metal + Core ML), llama.cpp (Metal), openai/whisper assets
#   bash scripts/port_mac.sh build       # export every Apple artefact from the public HF checkpoints
#   bash scripts/port_mac.sh reference   # PyTorch transcripts, only if the eval pack did not bring them
#   bash scripts/port_mac.sh validate    # MLX, Core ML, whisper.cpp, SNAC, Orpheus on llama.cpp
#   bash scripts/port_mac.sh executorch  # ExecuTorch with the Core ML backend (macOS-only export) + checks
#   bash scripts/port_mac.sh orpheus-mlx # Orpheus merged here, MLX 4-bit, end-to-end check (16 GB+ RAM)
#   bash scripts/port_mac.sh report      # docs/porting_report.md, Mac rows marked "measured on the mac"
#   bash scripts/port_mac.sh pack        # mac-results.tar with the result files (they hold test transcripts: keep private)
#   bash scripts/port_mac.sh all         # setup build validate executorch report pack
#
# A prebuilt bundle (halohalo-mac.tar, from `bash scripts/port_bundle_mac.sh`)
# also works: `build` then only fills in what is missing.
#
# Needs the Xcode command line tools (xcode-select --install) and Homebrew's
# Python and cmake (brew install python@3.12 cmake). Every step skips what is
# done and keeps going past a failure; the log is finetune_runs/port/mac.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export FINETUNE_DIR="$PWD/finetune_runs" PORT_TOOLS="$PWD/third_party" PYTHONPATH="$PWD" \
       PORT_HOST=mac PORT_THREADS=${PORT_THREADS:-8} TOKENIZERS_PARALLELISM=false
PY=${PY:-python3.12}
V=venv_mac            # MLX, Core ML, torch, transformers
VE=venv_mac_et        # ExecuTorch pins its own torch
LOG=finetune_runs/port/mac.log
W=sapinsapin/whisper-small-pld-ceb
C=sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm
O=sapinsapin/orpheus-3b-0.1-pretrained-char-pld-ceb
S=hubertsiuzdak/snac_24khz
A=finetune_runs/port/artefacts
G=$A/whisper-small-pld-ceb
mkdir -p finetune_runs/port
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
run() { say "$*"; "$@" >> "$LOG" 2>&1 && tail -1 "$LOG" | cut -c1-160 || say "  failed (see $LOG)"; }
need_eval() {
  [ -f finetune_runs/port/evalpack/ceb.npz ] && return 0
  echo "missing finetune_runs/port/evalpack/ceb.npz: untar halohalo-mac-eval.tar in the repo root first"
  echo "(made on the workstation with: bash scripts/port_bundle_mac.sh --eval-only)"
  exit 1
}

setup() {
  [ "$(uname -s)" = Darwin ] || { echo "this is the Mac half of the pipeline"; exit 1; }
  [ "$(uname -m)" = arm64 ] || say "warning: not Apple silicon; MLX and the Neural Engine need it"
  command -v cmake > /dev/null || { echo "brew install cmake"; exit 1; }
  command -v "$PY" > /dev/null || { echo "brew install python@3.12 (or PY=python3.x)"; exit 1; }
  [ -x $V/bin/python3 ] || "$PY" -m venv $V
  $V/bin/pip install -q --upgrade pip
  $V/bin/pip install -q mlx mlx-whisper mlx-lm coremltools torch transformers peft accelerate snac \
      soundfile scipy jiwer huggingface_hub safetensors numpy python-dotenv datasets sentencepiece protobuf
  mkdir -p third_party
  # whisper.cpp: Metal is on by default on macOS; the Core ML encoder with
  # fallback, so models without a compiled encoder still run on Metal
  [ -d third_party/whisper.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/whisper.cpp third_party/whisper.cpp
  cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DWHISPER_COREML=1 -DWHISPER_COREML_ALLOW_FALLBACK=1 -DWHISPER_BUILD_TESTS=OFF >> "$LOG"
  cmake --build third_party/whisper.cpp/build -j --config Release >> "$LOG"
  # OpenAI's repo: the mel filters and tokenizer whisper.cpp's converter reads
  [ -d third_party/whisper ] || git clone -q --depth 1 https://github.com/openai/whisper third_party/whisper
  [ -d third_party/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
  cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >> "$LOG"
  cmake --build third_party/llama.cpp/build -j --config Release --target llama-server llama-quantize >> "$LOG"
  say "setup done: $($V/bin/python3 -c 'import mlx.core as mx, coremltools as ct, torch; print("mlx", mx.__version__, "| coremltools", ct.__version__, "| torch", torch.__version__, "| metal", mx.metal.is_available())')"
}

build() {
  local P=$V/bin/python3
  [ -f $G/mlx/q4/config.json ] || run $P -m porting.export_mlx $W
  [ -d $G/coreml/encoder.mlpackage ] || run $P -m porting.export_coreml $W
  [ -f $G/ggml/ggml-model-q5_0.bin ] || run $P -m porting.export_ggml $W
  [ -d $A/omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm/coreml/model.mlpackage ] || run $P -m porting.export_coreml $C
  [ -d $A/snac_24khz/coreml/decoder.mlpackage ] || run $P -m porting.export_snac $S --target coreml
  # Orpheus: LoRA merged into the base, then GGUF for llama.cpp (needs ~16 GB RAM, ~20 GB disk)
  local R=$A/orpheus-3b-0.1-pretrained-char-pld-ceb
  [ -f $R/gguf/model-q4_k_m.gguf ] || run $P -m porting.export_orpheus $O --steps merge gguf
  # whisper.cpp's Core ML mode loads <model>-encoder.mlmodelc beside <model>.bin
  if [ -d $G/coreml/encoder.mlpackage ] && [ ! -d $G/ggml/ggml-model-f16-coreml-encoder.mlmodelc ]; then
    run xcrun coremlc compile $G/coreml/encoder.mlpackage $G/coreml/
    ln -sf ggml-model-f16.bin $G/ggml/ggml-model-f16-coreml.bin
    cp -R $G/coreml/encoder.mlmodelc $G/ggml/ggml-model-f16-coreml-encoder.mlmodelc
  fi
}

reference() {
  # PyTorch transcripts every port is compared with; the eval pack normally
  # brings the workstation's, and these are filed without the @mac tag
  need_eval
  for m in $W $C; do
    n=$(basename $m)
    [ -f finetune_runs/port/results/$n/torch-fp32-ceb.json ] || \
      run env PORT_HOST= $V/bin/python3 -m porting.validate $m --runtime torch
  done
}

validate() {
  need_eval
  local P="$V/bin/python3 -m porting.validate"
  # MLX on Metal: the full evalpack, at real speed
  for v in fp16 q8 q4; do run $P $W --runtime mlx --variant $v; done
  # Core ML: Neural Engine where it fits, against the CPU alone
  for v in all cpu; do run $P $W --runtime coreml --variant $v; done
  for v in all cpu; do run $P $C --runtime coreml --variant $v; done
  # whisper.cpp: Metal, and Metal + the Core ML encoder
  for v in f16 q5_0 f16-coreml; do run $P $W --runtime whispercpp --variant $v; done
  # SNAC decoder in Core ML
  for v in coreml coreml-cpu; do run $V/bin/python3 -m porting.validate_snac $S --variant $v --n 20; done
  # Orpheus Cebuano, 4-bit GGUF on llama.cpp with Metal, judged by MMS-1b-all
  run $V/bin/python3 -m porting.validate_orpheus $O --runtime llamacpp --variant q4_k_m --n 10
}

executorch() {
  # the mac-executorch target: ExecuTorch's Core ML partitioner exports only on macOS
  need_eval
  [ -x $VE/bin/python3 ] || "$PY" -m venv $VE
  $VE/bin/pip install -q --upgrade pip
  $VE/bin/pip install -q optimum-executorch coremltools snac soundfile jiwer huggingface_hub numpy
  for m in $W $C; do
    n=$(basename $m)
    [ -d $A/$n/executorch/coreml ] || run $VE/bin/python3 -m porting.export_executorch $m --backend coreml
    run $VE/bin/python3 -m porting.validate $m --runtime executorch --variant coreml --n 20
  done
}

orpheus_mlx() {
  local R=$A/orpheus-3b-0.1-pretrained-char-pld-ceb
  [ -f $R/merged/config.json ] || run $V/bin/python3 -m porting.export_orpheus $O --steps merge
  [ -f $R/mlx/q4/config.json ] || run $V/bin/python3 -m porting.export_orpheus $O --steps mlx
  run $V/bin/python3 -m porting.validate_orpheus $O --runtime mlx --variant q4 --n 10
}

report() { run $V/bin/python3 -m porting report; say "open docs/porting_report.md"; }

pack() {
  tar -cf mac-results.tar $(ls finetune_runs/port/results/*/*@mac*.json) "$LOG"
  say "mac-results.tar: untar it in the repo root on the workstation, then: python3 -m porting report"
}

case "${1:-}" in
  setup) setup ;;
  build) build ;;
  reference) reference ;;
  validate) validate ;;
  executorch) executorch ;;
  orpheus-mlx) orpheus_mlx ;;
  report) report ;;
  pack) pack ;;
  all) setup; build; reference; validate; executorch; report; pack ;;
  *) sed -n 2,27p "$0"; exit 2 ;;
esac
