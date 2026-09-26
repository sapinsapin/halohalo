#!/usr/bin/env bash
# The workstation GPU queue: everything from docs/status_2026-09-25.md that fits
# the RTX 3070 (8 GB, ~6 GB free with the desktop), run in order, unattended.
#
#   setsid nohup bash scripts/local_queue.sh > finetune_runs/local_queue/queue.log 2>&1 &
#   tail -f finetune_runs/local_queue/summary.txt
#
# Resumable: each step leaves done_<step> and is skipped on a rerun; a failed
# step is logged and the queue moves on. Order puts short, decision-making work
# first and the ~12 h fleet retrain last.
#
# Disk: C: is nearly full and WSL's own filesystem lives on it, so every cache,
# temp dir and venv here is on D:. (The WSL crashes once blamed on a full C:
# were CPU machine-check exceptions: see the kernel-panic logs in
# %LOCALAPPDATA%\Temp\wsl-crashes. Hence resumable steps and a supervisor.)
set -uo pipefail
cd "$(dirname "$0")/.."

set -a; . ./.env; set +a
export PLD_SOURCE=hub
export PIP_CACHE_DIR=/mnt/d/pip_cache TMPDIR=/mnt/d/tmp
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0          # the xet cache is on the WSL disk
export WANDB_MODE=disabled                      # results land in result.json anyway
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

R=/mnt/d/halohalo/finetune_runs
Q=$R/local_queue
SUMMARY=$Q/summary.txt
mkdir -p "$Q"
PY=venv/bin/python3
F='bogus|Warning|warn|it/s\]|examples/s'

log() { echo "$(date '+%F %T') $*" | tee -a "$SUMMARY"; }

wait_for_gpu() {   # other sessions have trained on this card
    while :; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [ "${free:-0}" -ge 5000 ] && return
        log "  GPU busy (${free} MiB free); waiting"
        sleep 300
    done
}

step() {           # step <name> <description> <function>
    local name=$1 desc=$2 fn=$3
    if [ -f "$Q/done_$name" ]; then log "== $name: done already"; return; fi
    wait_for_gpu
    log "== $name start: $desc"
    if $fn >> "$Q/$name.log" 2>&1; then
        touch "$Q/done_$name"; log "== $name OK"
    else
        log "== $name FAILED (see $Q/$name.log)"
    fi
}

# ---------------------------------------------------------------- L1
# MMS-TTS Cebuano scored 17.8 % CER on 2026-09-15 and 42.9 % on 2026-09-20; the
# judge and the sentences both changed. Same sentences, old judge: if it comes
# back near 17.8 the judge moved it, if near 42.9 the sentences did. Runs on a
# copy so the canonical results.json is never touched.
l1() {
    local W=$Q/l1_judge_check
    mkdir -p "$W/tts_eval/out/mms" "$W/tts_eval/ref"
    $PY - "$R/tts_eval/manifest.json" "$W/tts_eval/manifest.json" <<'PYX'
import json, sys
rows = [r for r in json.load(open(sys.argv[1])) if r["lang"] == "ceb"]
json.dump(rows, open(sys.argv[2], "w"), ensure_ascii=False)
print(len(rows), "ceb rows")
PYX
    ln -sfn "$R/tts_eval/ref/ceb" "$W/tts_eval/ref/ceb"
    ln -sfn "$R/tts_eval/out/mms/ceb" "$W/tts_eval/out/mms/ceb"
    FINETUNE_DIR=$W TTS_JUDGE=whisper-small $PY scripts/tts_eval.py --stage score --device cuda --model mms 2>&1 | grep -vE "$F"
    $PY - "$W/tts_eval/results.json" "$R/tts_eval/results.json" <<'PYX' | tee -a "$SUMMARY"
import json, sys
new, old = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
n, o = new["mms"]["ceb"]["cer"], old["mms"]["ceb"]["cer"]
f = new["reference"]["ceb"]["cer"]
print(f"  L1: MMS-TTS ceb = {n*100:.1f} % CER with whisper-small-pld-ceb (human floor {f*100:.1f}); "
      f"{o*100:.1f} % with whisper-large-v3 on the same sentences; 17.8 % on 2026-09-15's sentences.")
print("  L1 verdict: " + ("the judge moved it" if abs(n - 0.178) < abs(n - o) else "the sentences moved it"))
PYX
}

# ---------------------------------------------------------------- L2
# The independent judge: Meta's MMS-1b-all re-transcribes every saved TTS output.
# Writes results_mms_judge.json beside the canonical results.json.
ORPH=""
for l in bcl ceb eng fil hil ilo pag pam tsg war; do ORPH="$ORPH --model orpheus:$R/orpheus_char_pld_$l/final"; done
l2() {
    TTS_JUDGE=mms-1b-all TTS_RESULTS=results_mms_judge.json $PY scripts/tts_eval.py \
        --stage score --device cuda --model mms --model speecht5 --model qwen3tts_base $ORPH 2>&1 | grep -vE "$F"
    TTS_JUDGE=mms-1b-all TTS_RESULTS=results_mms_judge.json $PY scripts/tts_eval.py \
        --stage table 2>&1 | grep -E "^\||cer|spk" | tee -a "$SUMMARY"
}

# ---------------------------------------------------------------- L3
# MMS-1b-all zero-shot ASR on the frozen test splits: the baseline row.
l3() {
    $PY scripts/eval_mms_zeroshot.py 2>&1 | grep -vE "$F"
    tail -14 "$Q/l3.log" | grep -E "^\|" | tee -a "$SUMMARY"
}

# ---------------------------------------------------------------- L7
# Qwen3-TTS base, zero-shot, all ten languages (ceb and pam already exist and
# are skipped). Its own venv, as on the VM, created on D:.
l7() {
    [ -x venv_qwen/bin/python3 ] || bash scripts/setup_qwen_venv.sh
    FINETUNE_DIR=$R venv_qwen/bin/python3 scripts/synth_qwen_tts.py \
        --language bcl ceb eng fil hil ilo pag pam tsg war 2>&1 | grep -vE "$F|pad_token|SoX|path variables"
    TTS_JUDGE=mms-1b-all TTS_RESULTS=results_mms_judge.json $PY scripts/tts_eval.py \
        --stage score --device cuda --model qwen3tts_base 2>&1 | grep -vE "$F"
    $PY scripts/tts_eval.py --stage score --device cuda --model qwen3tts_base 2>&1 | grep -vE "$F"
    $PY - "$R/tts_eval" <<'PYX' | tee -a "$SUMMARY"
import json, sys
d = sys.argv[1]
for name, lab in (("results_mms_judge.json", "MMS judge"), ("results.json", "our judges")):
    r = json.load(open(f"{d}/{name}")).get("qwen3tts_base", {})
    print(f"  L7 qwen3tts_base ({lab}): " + "  ".join(
        f"{l} {v['cer']*100:.1f}%/{(v.get('spk_sim') or 0):.2f}" for l, v in sorted(r.items())))
PYX
}

# ---------------------------------------------------------------- L6
# Why did w2v-BERT learn nothing in any bake-off arm? Two short runs at lower
# learning rates than the 1e-4 it shared with the omni encoders. Not the
# bake-off recipe — bf16 weights and 8-bit Adam at batch 2 are what fits 6 GB —
# so the question is only "does the loss move at all", not "how good is it".
l6() {
    for lr in 1e-5 3e-5; do
        FINETUNE_DIR=$Q/w2vbert_lr$lr $PY finetune_ctc.py --encoder w2v-bert --language ceb \
            --units char --max-samples 4000 --max-steps 500 --lr $lr --warmup 50 \
            --batch-size 2 --grad-accum 8 --bf16-weights --optim adamw_bnb_8bit \
            --eval-steps 250 --eval-samples 300 --num-proc 1 --dataloader-workers 2 \
            2>&1 | grep -vE "$F" | grep -E "loss|eval_cer|Error|Traceback|vocab"
        r=$(ls $Q/w2vbert_lr$lr/*/result.json 2>/dev/null | head -1)
        [ -n "$r" ] && log "  L6 w2v-bert lr $lr: $(grep -E 'eval_cer|eval_wer|eval_loss' "$r" | tr -d ' \n')"
    done
}

# ---------------------------------------------------------------- L5
# The published whisper-small fleet on the frozen splits, its own recipe:
# honest numbers for the dataset card and the Space's "fast baseline".
l5() {
    PROFILE=local FLEET_DIR=$R/fleet_frozen bash scripts/retrain_fleet_frozen.sh
    tail -12 "$R/fleet_frozen/summary.txt" | tee -a "$SUMMARY"
}

log "=== local queue start"
step l1 "MMS-TTS Cebuano: judge or sentences?" l1
step l2 "independent TTS judge (MMS-1b-all) over every saved TTS output" l2
step l3 "MMS-1b-all zero-shot ASR baseline, frozen splits" l3
step l7 "Qwen3-TTS base zero-shot, ten languages" l7
step l6 "w2v-BERT collapse diagnosis" l6
step l5 "whisper-small fleet retrain on frozen splits (~12 h)" l5
log "=== local queue done"
