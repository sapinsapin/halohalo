#!/usr/bin/env bash
# The GB10 runner (docs/prd_gb10.md). A DGX Spark-class box: 20 Arm cores
# (10 Cortex-X925 + 10 Cortex-A725), a Blackwell GPU (sm_121, CUDA 13) and
# 128 GB of LPDDR5x that the CPU and GPU share. Everything here is sized to
# that memory: a job starts only while its expected peak plus MIN_FREE_GB
# would still fit, and spare memory goes to running jobs side by side.
#
#   bash scripts/gb10.sh setup      # checks, venvs, whisper.cpp (KleidiAI + generic); llama.cpp comes with orpheus
#   bash scripts/gb10.sh evalpack   # 100 frozen-test + 64 calibration clips for all ten languages
#   bash scripts/gb10.sh arm        # the headline Arm CPU numbers: demo models, quiet machine, 4 X925 cores
#   bash scripts/gb10.sh probe      # whisper-small at the L5 recipe, 1/2/3 at once -> JOBS for the fleet retrain
#   bash scripts/gb10.sh cpu-lane   # Phase 1 models on the Arm targets, NPU calibration >500M, the 7B CTC model
#   bash scripts/gb10.sh gpu-lane   # L6 (w2v-BERT, three LRs), L5 (whisper-small fleet), then Orpheus x 9
#   bash scripts/gb10.sh arm-tts    # Orpheus on 4 X925 cores (llama.cpp q4_k_m), quiet machine
#   bash scripts/gb10.sh report     # docs/porting_report.md with the @gb10 rows
#   bash scripts/gb10.sh pack       # finetune_runs/gb10/halohalo-gb10-results.tar, for the workstation
#   bash scripts/gb10.sh status     # memory, GPU, running jobs, the summary's tail
#   bash scripts/gb10.sh all        # setup evalpack arm probe, both lanes side by side, arm-tts report pack
#
# Run inside tmux so a dropped ssh does not end it:
#   tmux new -d -s gb10 'bash scripts/gb10.sh all'
# Every step skips finished work and continues past failures: rerunning any
# step, or `all`, is always safe.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }        # HF_TOKEN, if the user wrote one (`hf auth login` also works)
# this machine's paths, whatever a copied .env says
export PORT_HOST=gb10
export FINETUNE_DIR=${GB10_FINETUNE_DIR:-$PWD/finetune_runs}
export PORT_TOOLS=${GB10_PORT_TOOLS:-$PWD/third_party}
export TMPDIR=$FINETUNE_DIR/tmp PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOME/.cache/pip} npm_config_cache=$HOME/.npm
case "${HF_HOME:-}" in ""|/mnt/*) export HF_HOME=$HOME/.cache/huggingface ;; esac
export PLD_SOURCE=hub WANDB_MODE=${WANDB_MODE:-disabled} PYTHONPATH=$PWD
export TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130} CUDA_ARCH=${CUDA_ARCH:-121}
export MIN_FREE_GB=${MIN_FREE_GB:-16} PORT_MIN_FREE_GB=${PORT_MIN_FREE_GB:-16}
export PORT_STAGGER_S=${PORT_STAGGER_S:-60}
export NPU_CALIB_MAX_M=${NPU_CALIB_MAX_M:-2000}     # the 1B CTC models and whisper-large-v3; not the 7B
[ -d /usr/local/cuda/bin ] && export PATH=$PATH:/usr/local/cuda/bin
G=$FINETUNE_DIR/gb10
Q=$FINETUNE_DIR/gb10_queue
mkdir -p "$G" "$Q" "$TMPDIR"
SUMMARY=$G/summary.txt
PY=venv/bin/python3
LANGS10="bcl ceb eng fil hil ilo pag pam tsg war"
# what this machine can check: Arm CPU (ONNX, ggml, ExecuTorch), the NPU
# graphs' numerics, OpenVINO's Arm CPU plugin. Not Apple (the Mac's), not the
# browser (the workstation's Chrome).
GB10_TARGETS=arm-cpu-onnx,arm-cpu-ggml,arm-mobile-executorch,npu-qnn,npu-ryzenai,npu-openvino
# the fleet recipe on this machine (scripts/retrain_fleet_frozen.sh, PROFILE=gb10)
L5_HW="--batch-size 16 --grad-accum 1 --no-grad-checkpoint --num-proc 4 --dataloader-workers 4"

log() { echo "$(date '+%F %T') $*" | tee -a "$SUMMARY"; }
avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
total_gb() { awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo; }
wait_mem() { while [ "$(avail_gb)" -lt $((MIN_FREE_GB + $1)) ]; do sleep 30; done; }
lane() { echo "$2" > "$G/.lane_$1"; log "[$1] $2"; }

big_cores() {   # the four fastest cores (the X925s: a phone's big-core class), e.g. "10,11,12,13"
  local c f
  for c in /sys/devices/system/cpu/cpu[0-9]*; do
    f=$(cat "$c/cpu_capacity" 2> /dev/null || cat "$c/cpufreq/cpuinfo_max_freq" 2> /dev/null || echo 0)
    echo "$f ${c##*cpu}"
  done | sort -k1,1nr -k2,2n | head -4 | awk '{print $2}' | sort -n | paste -sd, -
}

need_splits() {
  local l miss=""
  for l in $LANGS10; do [ -f "splits/pld_$l.json" ] || miss="$miss $l"; done
  [ -z "$miss" ] && return 0
  log "splits/pld_*.json missing for:$miss. Untar halohalo-gb10-inputs.tar in the repo root (ask the user for it)."
  log "Do NOT continue without them: the loader would fall back to the random split and every number would change meaning."
  return 1
}

# ------------------------------------------------------------------ setup
setup() {
  lane main setup
  [ "$(uname -m)" = aarch64 ] || { log "not aarch64 ($(uname -m)): this runner is for the GB10"; return 1; }
  command -v nvidia-smi > /dev/null || { log "no nvidia-smi: is the NVIDIA driver installed?"; return 1; }
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee -a "$SUMMARY"
  log "memory $(total_gb) GB, $(nproc) cores, big cores $(big_cores)"
  local disk; disk=$(df -BG --output=avail "$FINETUNE_DIR" | tail -1 | tr -dc 0-9)
  log "disk: ${disk} GB free under $FINETUNE_DIR"
  # Orpheus x 9 is ~30 GB a language of artefacts, the 7B CTC ports ~100 GB,
  # the PLD download ~25 GB: below this, point GB10_FINETUNE_DIR at a bigger disk
  [ "$disk" -ge "${MIN_DISK_GB:-700}" ] || { log "less than ${MIN_DISK_GB:-700} GB free: stopping (MIN_DISK_GB overrides)"; return 1; }
  need_splits || return 1
  local c miss=""
  for c in git cmake ffmpeg tmux gcc g++ python3.12 nvcc; do command -v "$c" > /dev/null || miss="$miss $c"; done
  if [ -n "$miss" ]; then
    log "missing:$miss. The user installs them (sudo): sudo apt-get install -y git cmake ffmpeg tmux build-essential python3.12-venv python3.12-dev (nvcc: the CUDA toolkit, usually /usr/local/cuda/bin)"
    return 1
  fi
  if [ ! -f venv/.halohalo_ok ]; then
    log "main venv (torch from $TORCH_INDEX; the workstation's transformers/peft/datasets pins)"
    [ -x venv/bin/python3 ] || python3.12 -m venv venv
    venv/bin/pip install -q --upgrade pip wheel
    venv/bin/pip install -q "torch==${TORCH_VER:-2.11.0}" "torchaudio==${TORCH_VER:-2.11.0}" --index-url "$TORCH_INDEX" \
      || venv/bin/pip install -q torch torchaudio --index-url "$TORCH_INDEX" || { log "no CUDA torch for aarch64 from $TORCH_INDEX"; return 1; }
    venv/bin/pip install -q "transformers==5.14.1" "peft==0.20.0" "accelerate==1.14.0" "datasets==2.21.0" \
        huggingface_hub python-dotenv soundfile numpy scipy librosa jiwer onnxruntime snac sentencepiece \
        evaluate speechbrain defusedxml || { log "main venv install failed"; return 1; }
    venv/bin/pip install -q fasttext-wheel || log "fasttext-wheel did not install: the LID check will be skipped"
    venv/bin/python3 - << 'PY' | tee -a "$SUMMARY" || return 1
import torch, transformers
assert torch.cuda.is_available(), "torch sees no GPU"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      "sm_%d%d" % torch.cuda.get_device_capability(0), "| matmul ok", float((x @ x).float().abs().mean()) > 0,
      "| transformers", transformers.__version__)
PY
    touch venv/.halohalo_ok
  fi
  venv/bin/hf auth whoami > /dev/null 2>&1 \
    || { log "Hugging Face: not logged in. The user runs: venv/bin/hf auth login (PLD is private). Never handle the token yourself."; return 1; }
  local t v
  for t in onnx:venv_port_onnx openvino:venv_port_ov executorch:venv_port_et; do
    v=${t#*:}; t=${t%%:*}
    [ -f "$v/.halohalo_ok" ] && continue
    log "porting venv: $t"
    PY=python3.12 bash porting/setup_venvs.sh "$t" >> "$G/setup.log" 2>&1 || log "  $t failed (see $G/setup.log); its targets will be skipped"
  done
  # whisper.cpp twice: with Arm's KleidiAI micro-kernels (the default build,
  # what an Arm phone app would ship) and without, so their effect is measured
  if [ ! -x "$PORT_TOOLS/whisper.cpp/build/bin/whisper-cli" ]; then
    log "whisper.cpp (KleidiAI)"
    WHISPERCPP_CMAKE="-DGGML_CPU_KLEIDIAI=ON" bash porting/setup_venvs.sh ggml >> "$G/setup.log" 2>&1 || log "  failed (see $G/setup.log)"
  fi
  if [ ! -x "$PORT_TOOLS/whisper.cpp/build-generic/bin/whisper-cli" ]; then
    log "whisper.cpp (generic NEON)"
    WHISPERCPP_BUILD=build-generic bash porting/setup_venvs.sh ggml >> "$G/setup.log" 2>&1 || log "  failed (see $G/setup.log)"
  fi
  log "setup done"
}

evalpack() {
  lane main evalpack
  need_splits || return 1
  $PY -m porting.evalpack --languages $LANGS10 --n 100 --calib 64 2>&1 | grep -vE "it/s\]|examples/s" | tee -a "$G/evalpack.log"
}

# ------------------------------------------------------------------ Arm, headline
arm() {
  lane main "arm: demo builds"
  $PY -m porting build --models demo --targets "$GB10_TARGETS" --jobs 3
  local cores; cores=$(big_cores)
  lane main "arm: demo checks on cores $cores, 4 threads, one at a time"
  # nothing else runs now: on unified memory a busy GPU or a build on the
  # other cores takes bandwidth the check's RTF would then not show
  PORT_CPUS=$cores PORT_THREADS=4 PORT_STAGGER_S=0 $PY -m porting validate --models demo --targets "$GB10_TARGETS" --jobs 1
  lane main "arm: whisper.cpp without KleidiAI (filed as @gb10-generic)"
  PORT_HOST=gb10-generic WHISPERCPP_BUILD=build-generic PORT_CPUS=$cores PORT_THREADS=4 PORT_STAGGER_S=0 \
    $PY -m porting validate --models demo --targets arm-cpu-ggml --jobs 1
}

arm_tts() {
  local repo=sapinsapin/orpheus-3b-0.1-pretrained-char-pld-ceb name=orpheus-3b-0.1-pretrained-char-pld-ceb cores
  [ -f "$FINETUNE_DIR/port/results/$name/orpheus-llamacpp@gb10-cpu-q4_k_m.json" ] && return
  [ -f "$FINETUNE_DIR/port/artefacts/$name/gguf/model-q4_k_m.gguf" ] || { log "arm-tts: no q4_k_m GGUF yet (gpu-lane builds it)"; return; }
  cores=$(big_cores)
  lane main "arm-tts: Orpheus ceb, llama.cpp q4_k_m on cores $cores"
  CUDA_VISIBLE_DEVICES= PORT_HOST=gb10-cpu PORT_THREADS=4 ORPHEUS_PARALLEL=1 \
    taskset -c "$cores" $PY -m porting.validate_orpheus "$repo" --runtime llamacpp --variant q4_k_m --n 3 \
    2>&1 | tail -3 | tee -a "$SUMMARY"
}

# ------------------------------------------------------------------ probe
# tqdm's elapsed time at a training step, in seconds
elapsed_at() {   # LOG STEP TOTAL
  tr '\r' '\n' < "$1" 2> /dev/null | grep -oE "\| $2/$3 \[[0-9:]+" | tail -1 | grep -oE '[0-9:]+$' \
    | awk -F: '{ if (NF == 3) print $1*3600 + $2*60 + $3; else print $1*60 + $2 }'
}

probe() {
  [ -f "$G/probe.env" ] && { log "probe: done already ($(tr '\n' ' ' < "$G/probe.env"))"; return; }
  lane gpu probe
  mkdir -p "$G/probe"
  local FROM=20 TO=90 TOTAL=100 k j t a b s rate base low peak per_job=0 best=1 best_rate=0 reached alive p
  for k in 1 2 3; do
    base=$(( $(total_gb) - $(avail_gb) )); low=$(avail_gb)
    local pids=()
    for j in $(seq 1 "$k"); do
      rm -rf "$G/probe/k$k-$j"
      # its own process group, so stopping it also stops its dataloader workers
      FINETUNE_DIR=$G/probe/k$k-$j setsid $PY finetune_asr.py --model openai/whisper-small --dataset pld --language ceb \
          --max-samples 2000 --max-steps $TOTAL $L5_HW --eval-samples 50 --eval-steps 100000 \
          > "$G/probe/k$k-$j.log" 2>&1 &
      pids+=($!)
    done
    for ((t = 0; t < 3600; t += 10)); do
      sleep 10
      a=$(avail_gb); [ "$a" -lt "$low" ] && low=$a
      reached=0; alive=0
      for j in $(seq 1 "$k"); do [ -n "$(elapsed_at "$G/probe/k$k-$j.log" $TO $TOTAL)" ] && reached=$((reached + 1)); done
      for p in "${pids[@]}"; do kill -0 "$p" 2> /dev/null && alive=$((alive + 1)); done
      if [ "$reached" -eq "$k" ] || [ "$alive" -eq 0 ]; then break; fi
    done
    for p in "${pids[@]}"; do kill -- "-$p" 2> /dev/null; done      # the probe needs step times, not a model
    wait "${pids[@]}" 2> /dev/null; sleep 20                         # let the memory come back before the next round
    peak=$(( $(total_gb) - low - base ))
    rate=0; s=""
    for j in $(seq 1 "$k"); do
      a=$(elapsed_at "$G/probe/k$k-$j.log" $FROM $TOTAL); b=$(elapsed_at "$G/probe/k$k-$j.log" $TO $TOTAL)
      if [ -z "$a" ] || [ -z "$b" ] || [ "$b" -le "$a" ]; then rate=""; break; fi
      s="$s $(awk "BEGIN{printf \"%.2f\", ($b - $a) / ($TO - $FROM)}")"
      rate=$(awk "BEGIN{print $rate + ($TO - $FROM) / ($b - $a)}")
    done
    if [ -z "$rate" ]; then
      log "probe: $k at once did not reach step $TO (see $G/probe/k$k-*.log)"
      [ "$k" -eq 1 ] && return 1
      break
    fi
    log "probe: $k at once: s/step$s | $(printf %.2f "$rate") steps/s together | ${peak} GB at peak"
    [ "$k" -eq 1 ] && per_job=$peak
    # another job must buy 15% more throughput to be worth its memory
    if awk "BEGIN{exit !($rate >= 1.15 * $best_rate)}"; then best=$k; best_rate=$rate; else break; fi
  done
  printf 'JOBS_GB10=%s\nJOB_GB=%s\n' "$best" "$((per_job + 4))" > "$G/probe.env"
  log "probe: fleet retrain runs $best at once, $((per_job + 4)) GB each (written to $G/probe.env)"
}

# ------------------------------------------------------------------ lanes
npu_models() {   # >500M ASR models whose A16W8 calibration fits here
  $PY - << 'PY'
import os
from porting.registry import MODELS
lim = float(os.environ["NPU_CALIB_MAX_M"])
print(",".join(m.repo for m in MODELS if m.family in ("whisper", "wav2vec2-ctc") and 500 < m.params_m <= lim))
PY
}

cpu_lane() {
  local j=${CPU_JOBS:-4} npu
  lane cpu "Phase 1 builds: every model, Arm targets"
  $PY -m porting build --phase 1 --models phase --targets "$GB10_TARGETS" --jobs "$j"
  lane cpu "Phase 1 checks: siblings at 20 clips"
  $PY -m porting validate --phase 1 --models phase --targets "$GB10_TARGETS" --smoke 20 --jobs "$j"
  npu=$(npu_models)
  lane cpu "NPU A16W8 calibration above 500M params"
  $PY -m porting build --phase 2 --models "$npu" --targets npu-qnn,npu-ryzenai --jobs "${NPU_JOBS:-3}"
  $PY -m porting validate --phase 2 --models "$npu" --targets npu-qnn,npu-ryzenai --smoke 20 --jobs "${NPU_JOBS:-3}"
  lane cpu "Phase 2: the 7B CTC model (its conversion waits for about 90 GB free; no NPU graph at 7B)"
  $PY -m porting build --phase 2 --models phase --targets arm-cpu-onnx,arm-cpu-ggml,arm-mobile-executorch,npu-openvino --jobs 2
  $PY -m porting validate --phase 2 --models phase --targets arm-cpu-onnx,arm-cpu-ggml,arm-mobile-executorch,npu-openvino --smoke 20 --jobs 2
  lane cpu done
}

l6() {   # w2v-BERT: does the loss move at all? Three LRs side by side, the bake-off's 1e-4 as the control
  local lr out n=0
  for lr in 1e-5 3e-5 1e-4; do
    out=$Q/w2vbert_lr$lr
    compgen -G "$out/*/result.json" > /dev/null && { log "L6 lr $lr: done already"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "${L6_JOBS:-3}" ]; do wait -n; done
    wait_mem "${L6_GB:-40}"
    # the bake-off recipe's effective batch (16), none of the 6 GB card's compromises:
    # fp32 weights, fused AdamW, no gradient checkpointing
    FINETUNE_DIR=$out $PY finetune_ctc.py --encoder w2v-bert --language ceb --units char \
        --max-samples 4000 --max-steps 500 --lr "$lr" --warmup 50 --batch-size 16 --grad-accum 1 \
        --no-grad-checkpoint --eval-steps 250 --eval-samples 300 --num-proc 4 --dataloader-workers 4 \
        > "$Q/l6_lr$lr.log" 2>&1 &
    n=$((n + 1)); sleep 120
  done
  wait
  for lr in 1e-5 3e-5 1e-4; do
    local r; r=$(ls "$Q"/w2vbert_lr$lr/*/result.json 2> /dev/null | head -1)
    if [ -n "$r" ]; then log "L6 lr $lr: $(grep -E '"eval_(cer|wer|loss)"' "$r" | tr -d ' \n')"
    else log "L6 lr $lr: FAILED (see $Q/l6_lr$lr.log)"; fi
  done
}

l5() {
  [ -f "$G/probe.env" ] && . "$G/probe.env"
  log "L5: fleet retrain, ${JOBS_GB10:-2} at once, ${JOB_GB:-24} GB each"
  PROFILE=gb10 FLEET_DIR=$FINETUNE_DIR/fleet_frozen JOBS_GB10=${JOBS_GB10:-2} JOB_GB=${JOB_GB:-24} \
    bash scripts/retrain_fleet_frozen.sh
  tail -12 "$FINETUNE_DIR/fleet_frozen/summary.txt" | tee -a "$SUMMARY"
}

orpheus() {
  # nine languages, two at once; four sentences decoded at once per check
  ASR=0 JOBS=${ORPHEUS_JOBS:-2} bash scripts/port_phase2.sh 2>&1 | tee -a "$G/orpheus.log" | grep -E "===|failed" | tee -a "$SUMMARY"
}

gpu_lane() {
  lane gpu "L6: w2v-BERT at three learning rates"; l6
  lane gpu "L5: whisper-small fleet on the frozen splits"; l5
  lane gpu "Orpheus x 9: exports and end-to-end checks"; orpheus
  lane gpu done
}

# ------------------------------------------------------------------ watching
monitor() {   # every 30 s: memory, GPU, and what each lane is doing
  [ -f "$G/monitor.tsv" ] || printf 'time\tused_gb\tavail_gb\tgpu_util\tgpu_watts\tcpu_lane\tgpu_lane\n' > "$G/monitor.tsv"
  local u
  while :; do
    u=$(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' ' | tr , '\t')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" $(( $(total_gb) - $(avail_gb) )) "$(avail_gb)" "${u:-NA	NA}" \
      "$(cat "$G/.lane_cpu" 2> /dev/null)" "$(cat "$G/.lane_gpu" 2> /dev/null)" >> "$G/monitor.tsv"
    sleep 30
  done
}

watchers_start() {
  monitor & echo $! > "$G/monitor.pid"
  # the last resort: end the biggest porting job (resumable) before the
  # kernel's OOM killer picks something, possibly a training run
  MIN_KB=${GUARD_KB:-8000000} MEM_GUARD_LOG=$G/mem_guard.log bash scripts/mem_guard.sh & echo $! > "$G/mem_guard.pid"
}
watchers_stop() {
  local f
  for f in "$G/monitor.pid" "$G/mem_guard.pid"; do [ -f "$f" ] && kill "$(cat "$f")" 2> /dev/null; rm -f "$f"; done
}

status() {
  free -g | head -2
  nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu --format=csv 2> /dev/null
  ps -eo pid,etime,rss,args --sort=-rss | grep -E "[f]inetune_|[p]orting\.|[l]lama-server" | cut -c1-160 | head -12
  echo "lanes: cpu=$(cat "$G/.lane_cpu" 2> /dev/null) gpu=$(cat "$G/.lane_gpu" 2> /dev/null)"
  [ -f "$G/monitor.tsv" ] && awk -F'\t' 'NR > 1 { if ($2 > m) m = $2; if ($4 ~ /^[0-9]+$/) { s += $4; n++ } }
      END { printf "monitor: peak used %d GB; mean GPU util %.0f%%\n", m, n ? s / n : 0 }' "$G/monitor.tsv"
  tail -15 "$SUMMARY" 2> /dev/null
}

report() { lane main report; $PY -m porting report; }

pack() {   # results for the workstation: small files only, no weights
  local out=$G/halohalo-gb10-results.tar list
  list=$(mktemp)
  {
    find finetune_runs/port/results finetune_runs/port/logs -type f 2> /dev/null
    find finetune_runs/port/artefacts -maxdepth 3 -name 'build*.json' 2> /dev/null
    find finetune_runs/port/evalpack -maxdepth 1 -type f 2> /dev/null
    find finetune_runs/fleet_frozen -maxdepth 2 \( -name 'result.json' -o -name '*.log' -o -name summary.txt \) 2> /dev/null
    find finetune_runs/gb10_queue -maxdepth 3 \( -name 'result.json' -o -name '*.log' \) 2> /dev/null
    find finetune_runs/gb10 -maxdepth 2 -type f ! -name '*.tar' ! -name '*.pid' 2> /dev/null
  } | sort -u > "$list"
  tar -cf "$out" -T "$list" && rm -f "$list"
  log "pack: $out ($(du -h "$out" | cut -f1)). PLD clips and transcripts inside: copy by hand, never commit or upload."
}

all() {
  setup || return 1
  evalpack || return 1
  arm
  probe
  lane cpu waiting; lane gpu waiting
  watchers_start
  trap watchers_stop EXIT
  cpu_lane > "$G/cpu_lane.log" 2>&1 &
  gpu_lane > "$G/gpu_lane.log" 2>&1 &
  wait
  watchers_stop
  arm_tts
  report
  pack
  log "=== all done"
}

case "${1:-}" in
  setup) setup ;; evalpack) evalpack ;; arm) arm ;; arm-tts) arm_tts ;; probe) probe ;;
  cpu-lane) cpu_lane ;; gpu-lane) gpu_lane ;; l6) l6 ;; l5) l5 ;; orpheus) orpheus ;;
  report) report ;; pack) pack ;; status) status ;; all) all ;;
  *) sed -n '2,23p' "$0"; exit 2 ;;
esac
