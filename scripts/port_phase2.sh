#!/usr/bin/env bash
# Porting pipeline, Phase 2: the RTX PRO 6000 VM (96 GB, Blackwell sm_120).
# Everything the workstation cannot convert or check:
#   - Orpheus-3B x 9 languages -> GGUF, MLX, ONNX GenAI, OpenVINO, ExecuTorch,
#     WebLLM, MediaPipe; end-to-end TTS check per runtime with the MMS judge
#   - omniASR 7B CTC -> ONNX / OpenVINO; whisper-large-v3 NPU QDQ encoders
#
# One chained run, so the card never waits on a person (the idle watchdog
# stops the VM 30 min after the last job). Resumable: every step skips what
# exists. Run in tmux on the VM:
#   tmux new -d -s port 'bash scripts/port_phase2.sh 2>&1 | tee -a /mnt/data/finetune_runs/port/phase2.log'
#
#   LANGS="ceb pam"  a subset of Orpheus languages (default: all nine)
#   STEPS="..."      Orpheus toolchains (default: all)
#   N=10             sentences per language for the end-to-end check
#   DRY=1            the workstation dry run: CPU only, llama.cpp only
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export FINETUNE_DIR=${FINETUNE_DIR:-/mnt/data/finetune_runs}
export PORT_TOOLS=${PORT_TOOLS:-/mnt/data/third_party}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/mnt/data/pip_cache} TMPDIR=${TMPDIR:-/mnt/data/tmp}
mkdir -p "$PORT_TOOLS" "$TMPDIR" "$FINETUNE_DIR/port"
LANGS=${LANGS:-"bcl ceb eng fil hil ilo pam tsg war"}
STEPS=${STEPS:-"merge gguf onnx mlx openvino executorch webllm mediapipe"}
N=${N:-10}
DRY=${DRY:-0}
CU="https://download.pytorch.org/whl/cu128"
# the workstation has no system cmake; the ONNX toolchain venv carries one
command -v cmake > /dev/null || export PATH="$PATH:$PWD/venv_port_onnx/bin"
say() { echo "$(date '+%F %T') $*"; }

# ------------------------------------------------------------ environments
# One per toolchain: ExecuTorch, MLC and ai-edge-torch each pin their own torch.
venv() {   # venv <dir> <pip args...>
  local d=$1; shift
  [ -x "$d/bin/python3" ] && return
  python3 -m venv "$d" && "$d/bin/pip" install -q --upgrade pip wheel && "$d/bin/pip" install -q "$@"
}
LLMV=venv_llm
if [ "$DRY" = 1 ]; then
  LLMV=venv                          # the workstation's training venv has torch, peft and snac already
  export CUDA_VISIBLE_DEVICES=       # the 3070 belongs to the local queue; the dry run is CPU only
fi
setup() {
  [ -x "$LLMV/bin/python3" ] || { venv venv_llm torch --index-url "$CU"; venv_llm/bin/pip install -q transformers peft snac \
        accelerate safetensors sentencepiece gguf onnxruntime-genai soundfile jiwer; }
  if [ ! -x "$PORT_TOOLS/llama.cpp/build/bin/llama-server" ]; then
    [ -d "$PORT_TOOLS/llama.cpp" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$PORT_TOOLS/llama.cpp"
    local CUDA=ON; [ "$DRY" = 1 ] && CUDA=OFF
    cmake -S "$PORT_TOOLS/llama.cpp" -B "$PORT_TOOLS/llama.cpp/build" -DGGML_CUDA=$CUDA \
          -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build "$PORT_TOOLS/llama.cpp/build" -j "$(nproc)" --target llama-server llama-quantize > /dev/null
    [ "$DRY" = 1 ] || venv_llm/bin/pip install -q -r "$PORT_TOOLS/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" || true
  fi
  [ "$DRY" = 1 ] && return
  venv venv_llm_mlx "mlx[cpu]" mlx-lm
  venv venv_llm_ov "optimum-intel[openvino]" nncf --extra-index-url https://download.pytorch.org/whl/cpu
  venv venv_llm_et optimum-executorch --extra-index-url https://download.pytorch.org/whl/cpu
  venv venv_llm_mlc --pre -f https://mlc.ai/wheels mlc-llm-nightly-cu128 mlc-ai-nightly-cu128
  venv venv_llm_aiedge ai-edge-torch mediapipe transformers --extra-index-url https://download.pytorch.org/whl/cpu
}

# step -> the venv whose python runs it (PATH also gets that venv, for optimum-cli / mlc_llm)
py_for() {
  case $1 in
    merge|gguf|onnx) echo "$LLMV" ;;
    mlx) echo venv_llm_mlx ;;
    openvino) echo venv_llm_ov ;;
    executorch) echo venv_llm_et ;;
    webllm) echo venv_llm_mlc ;;
    mediapipe) echo venv_llm_aiedge ;;
  esac
}

say "=== porting phase 2 (langs: $LANGS; steps: $STEPS; dry: $DRY)"
setup

# ------------------------------------------------------------ Orpheus
for l in $LANGS; do
  repo=sapinsapin/orpheus-3b-0.1-pretrained-char-pld-$l
  for s in $STEPS; do
    [ "$DRY" = 1 ] && [[ "$s" != merge && "$s" != gguf ]] && continue
    v=$(py_for "$s")
    say "orpheus $l: $s ($v)"
    PATH="$PWD/$v/bin:$PATH" PYTHONPATH="$PWD" "$v/bin/python3" -u -m porting.export_orpheus "$repo" --steps "$s" \
      || say "  $s failed (logged in build_orpheus.json); continuing"
  done
  # end to end: the reference, then each runtime this VM can execute
  say "orpheus $l: end-to-end checks"
  if [ "$DRY" = 1 ]; then
    RUNS="llamacpp:q4_k_m"
  else
    RUNS="torch:bf16 llamacpp:q8_0 llamacpp:q4_k_m onnx:cpu openvino:int4"
  fi
  for rv in $RUNS; do
    rt=${rv%%:*}; var=${rv##*:}
    out="$FINETUNE_DIR/port/results/orpheus-3b-0.1-pretrained-char-pld-$l/orpheus-$rt-$var.json"
    [ -f "$out" ] && continue
    PYTHONPATH="$PWD" venv/bin/python3 -m porting.validate_orpheus "$repo" --runtime "$rt" --variant "$var" --n "$N" \
      || say "  $rt/$var failed; continuing"
  done
done

[ "$DRY" = 1 ] && { say "=== dry run done"; exit 0; }

# ------------------------------------------------------------ ASR that waited for the VM
say "asr: phase 2 builds and checks"
python3 -m porting build --phase 2 --models phase
python3 -m porting validate --phase 2 --models phase --smoke 20
python3 -m porting report
say "=== phase 2 done"
