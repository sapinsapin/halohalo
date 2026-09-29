#!/usr/bin/env bash
# Porting pipeline, Phase 2: the models the workstation cannot hold.
# Runs on the RTX PRO 6000 VM (96 GB, sm_120) or on the GB10 (128 GB unified,
# aarch64, sm_121: PORT_HOST=gb10, see docs/prd_gb10.md):
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
#   STEPS="..."      Orpheus toolchains (default: all; on the GB10 the ones with aarch64 wheels)
#   N=10             sentences per language for the end-to-end check
#   JOBS=1           languages side by side, each only while MIN_FREE_GB + NEED_GB is free
#   ORPHEUS_PARALLEL / ORPHEUS_BATCH   sentences decoded at once per check (llama.cpp slots / torch batch)
#   TORCH_INDEX, CUDA_ARCH             the wheel index and GPU arch (cu128 / 120 on the VM)
#   ASR=0            Orpheus only (gb10.sh runs the ASR half in its CPU lane)
#   DRY=1            the workstation dry run: CPU only, llama.cpp only
set -uo pipefail
cd "$(dirname "$0")/.."
# .env fills in what the caller did not set: gb10.sh exports its own paths,
# which a copied workstation .env (FINETUNE_DIR=/mnt/d/...) must not undo
_caller=$(export -p)
[ -f .env ] && { set -a; . ./.env; set +a; }
eval "$_caller" 2> /dev/null
ASR=${ASR:-1}
export FINETUNE_DIR=${FINETUNE_DIR:-/mnt/data/finetune_runs}
export PORT_TOOLS=${PORT_TOOLS:-/mnt/data/third_party}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/mnt/data/pip_cache} TMPDIR=${TMPDIR:-/mnt/data/tmp}
mkdir -p "$PORT_TOOLS" "$TMPDIR" "$FINETUNE_DIR/port"
HOST=${PORT_HOST:-}
LANGS=${LANGS:-"bcl ceb eng fil hil ilo pam tsg war"}
if [ "$HOST" = gb10 ]; then
  # MLX is the Mac's (scripts/port_mac.sh); MLC publishes no aarch64 CUDA
  # wheels. MediaPipe is best effort: ai-edge-torch's aarch64 wheels come and go.
  STEPS=${STEPS:-"merge gguf onnx openvino executorch mediapipe"}
  TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
  CUDA_ARCH=${CUDA_ARCH:-121}
  JOBS=${JOBS:-2}
  export ORPHEUS_PARALLEL=${ORPHEUS_PARALLEL:-4} ORPHEUS_BATCH=${ORPHEUS_BATCH:-4}
else
  STEPS=${STEPS:-"merge gguf onnx mlx openvino executorch webllm mediapipe"}
  TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
  CUDA_ARCH=${CUDA_ARCH:-120}
  JOBS=${JOBS:-1}
fi
N=${N:-10}
DRY=${DRY:-0}
MIN_FREE_GB=${MIN_FREE_GB:-16}     # never started into: the OOM headroom
NEED_GB=${NEED_GB:-40}             # the largest Orpheus export step's peak (ExecuTorch, NNCF int4)
# the workstation has no system cmake; the ONNX toolchain venv carries one
command -v cmake > /dev/null || export PATH="$PATH:$PWD/venv_port_onnx/bin"
say() { echo "$(date '+%F %T') $*"; }
free_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
wait_mem() { while [ "$(free_gb)" -lt $((MIN_FREE_GB + $1)) ]; do sleep 20; done; }

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
  # four cores for this shell and everything it starts (nproc, torch and ORT
  # follow the mask): the workstation throws machine checks under all-core load
  taskset -cp "${BUILD_CPUS:-0-3}" $$ > /dev/null
fi
setup() {
  if [ ! -x "$LLMV/bin/python3" ]; then
    venv venv_llm torch --index-url "$TORCH_INDEX"
    venv_llm/bin/pip install -q transformers peft snac accelerate safetensors sentencepiece gguf soundfile jiwer \
        scipy python-dotenv
    # the ONNX GenAI builder; aarch64 wheels lag x86 (the onnx step then fails and says so)
    venv_llm/bin/pip install -q onnxruntime-genai || say "onnxruntime-genai: no wheel for $(uname -m)"
  fi
  if [ ! -x "$PORT_TOOLS/llama.cpp/build/bin/llama-server" ]; then
    [ -d "$PORT_TOOLS/llama.cpp" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$PORT_TOOLS/llama.cpp"
    local CUDA=ON; [ "$DRY" = 1 ] && CUDA=OFF
    cmake -S "$PORT_TOOLS/llama.cpp" -B "$PORT_TOOLS/llama.cpp/build" -DGGML_CUDA=$CUDA \
          -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build "$PORT_TOOLS/llama.cpp/build" -j "$(nproc)" --target llama-server llama-quantize > /dev/null
    [ "$DRY" = 1 ] || venv_llm/bin/pip install -q -r "$PORT_TOOLS/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" || true
  fi
  [ "$DRY" = 1 ] && return
  for s in $STEPS; do
    case $s in
      mlx) venv venv_llm_mlx "mlx[cpu]" mlx-lm ;;
      # + what the end-to-end check needs besides the runtime: SNAC, the judge's resampler, CER
      openvino) venv venv_llm_ov "optimum-intel[openvino]" nncf snac scipy soundfile jiwer python-dotenv \
                  --extra-index-url https://download.pytorch.org/whl/cpu ;;
      executorch) venv venv_llm_et optimum-executorch --extra-index-url https://download.pytorch.org/whl/cpu ;;
      webllm) venv venv_llm_mlc --pre -f https://mlc.ai/wheels mlc-llm-nightly-cu128 mlc-ai-nightly-cu128 ;;
      mediapipe) venv venv_llm_aiedge ai-edge-torch mediapipe transformers --extra-index-url https://download.pytorch.org/whl/cpu \
                   || say "ai-edge-torch/mediapipe: no wheels for $(uname -m); the mediapipe step will fail and say so" ;;
    esac
  done
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

say "=== porting phase 2 (host: ${HOST:-vm}; langs: $LANGS; steps: $STEPS; jobs: $JOBS; dry: $DRY)"
setup

# ------------------------------------------------------------ Orpheus
orpheus_lang() {
  local l=$1 repo=sapinsapin/orpheus-3b-0.1-pretrained-char-pld-$1 s v rt var out tag
  for s in $STEPS; do
    [ "$DRY" = 1 ] && [[ "$s" != merge && "$s" != gguf ]] && continue
    v=$(py_for "$s")
    [ -x "$v/bin/python3" ] || { say "orpheus $l: $s skipped (no $v)"; continue; }
    wait_mem "$NEED_GB"
    say "orpheus $l: $s ($v)"
    PATH="$PWD/$v/bin:$PATH" PYTHONPATH="$PWD" "$v/bin/python3" -u -m porting.export_orpheus "$repo" --steps "$s" \
      || say "  $s failed (logged in build_orpheus.json); continuing"
  done
  # end to end: the reference, then each runtime this machine can execute
  say "orpheus $l: end-to-end checks"
  if [ "$DRY" = 1 ]; then
    RUNS="llamacpp:q4_k_m"
  else
    RUNS="torch:bf16 llamacpp:q8_0 llamacpp:q4_k_m onnx:cpu openvino:int4"
  fi
  for rv in $RUNS; do
    rt=${rv%%:*}; var=${rv##*:}
    # a port measured on another machine is tagged <runtime>@host; the torch
    # reference is the same everywhere and is never tagged
    tag=$HOST; [ "$rt" = torch ] && tag=
    out="$FINETUNE_DIR/port/results/orpheus-3b-0.1-pretrained-char-pld-$l/orpheus-$rt${tag:+@$tag}-$var.json"
    [ -f "$out" ] && continue
    # each check runs where its runtime is installed: the main venv has
    # neither onnxruntime-genai nor optimum-intel, and must not get them
    case $rt in onnx) v=$LLMV ;; openvino) v=venv_llm_ov ;; *) v=venv ;; esac
    [ -x "$v/bin/python3" ] || { say "  $rt/$var skipped (no $v)"; continue; }
    wait_mem 12
    PORT_HOST=$tag PYTHONPATH="$PWD" "$v/bin/python3" -m porting.validate_orpheus "$repo" --runtime "$rt" --variant "$var" --n "$N" \
      || say "  $rt/$var failed; continuing"
  done
}

for l in $LANGS; do
  if [ "$JOBS" -gt 1 ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    orpheus_lang "$l" > "$FINETUNE_DIR/port/phase2_orpheus_$l.log" 2>&1 &
    sleep 90      # let the new job's memory show up before the next gate reads it
  else
    orpheus_lang "$l"
  fi
done
wait

[ "$DRY" = 1 ] && { say "=== dry run done"; exit 0; }
[ "$ASR" = 0 ] && { say "=== orpheus done (ASR=0: the ASR half runs elsewhere)"; exit 0; }

# ------------------------------------------------------------ ASR that waited for the big machine
say "asr: phase 2 builds and checks"
PY=python3; [ -x venv/bin/python3 ] && PY=venv/bin/python3
$PY -m porting build --phase 2 --models phase --jobs "${ASR_JOBS:-1}"
$PY -m porting validate --phase 2 --models phase --smoke 20 --jobs "${ASR_JOBS:-1}"
$PY -m porting report
say "=== phase 2 done"
