#!/usr/bin/env bash
# Start a job on the VM inside tmux, logging to /mnt/data/logs/<session>.log.
# Runs the bootstrap first whenever it has not completed for the current
# version of scripts/bootstrap_nebius.sh (fresh VM, or the script changed).
#
#   bash scripts/nebius/run.sh smoke                 # 20-clip CTC run, ~minutes
#   bash scripts/nebius/run.sh bakeoff               # R1/R2 bake-off, resumes
#   BAKEOFF_LANGS="ceb" bash scripts/nebius/run.sh bakeoff
#   bash scripts/nebius/run.sh <session> '<command run in ~/halohalo>'
#
# Re-running after a preemption resumes: every trainer picks up its last
# checkpoint, and run_bakeoff.sh skips finished runs.
source "$(dirname "$0")/common.sh"

session="${1:?usage: run.sh smoke|bakeoff|<session> [command]}"
case "$session" in
  smoke)   cmd='FINETUNE_DIR=/mnt/data/smoke venv/bin/python3 finetune_ctc.py --encoder omni-300m --language ceb --units char --smoke' ;;
  bakeoff) cmd='bash scripts/run_bakeoff.sh' ;;
  *)       cmd="${2:?give a command for custom session $session}" ;;
esac

# pass bake-off knobs through if set on the workstation
passthru=""
for v in BAKEOFF_LANGS BAKEOFF_STEPS BAKEOFF_SAMPLES BAKEOFF_BATCH BAKEOFF_ACCUM BAKEOFF_CTC BAKEOFF_WHISPER; do
    [ -n "${!v:-}" ] && passthru+="export $v=$(printf %q "${!v}"); "
done

vm_ssh 'bash -s' <<REMOTE
set -euo pipefail
cd ~/halohalo
if [ ! -f .env ] || ! grep -q '^HF_TOKEN=' .env; then
    echo "!! secrets missing on the VM: run  bash scripts/nebius/push_secrets.sh  yourself"; exit 3
fi
if tmux has-session -t '$session' 2>/dev/null; then
    echo "!! tmux session '$session' is already running; see status.sh"; exit 4
fi
mkdir -p $DATA_ROOT/logs
want=\$(sha256sum scripts/bootstrap_nebius.sh | cut -d' ' -f1)
have=\$(cat $DATA_ROOT/.bootstrapped 2>/dev/null || true)
boot=""
[ "\$want" = "\$have" ] || boot="bash scripts/bootstrap_nebius.sh && "
# the paths bootstrap writes to .env, exported for scripts that read the environment
env="export HF_HOME=$DATA_ROOT/hf_cache FINETUNE_DIR=$DATA_ROOT/finetune_runs PLD_WORK_DIR=$DATA_ROOT/pld_shards HF_XET_CACHE=$DATA_ROOT/hf_cache/xet PLD_SOURCE=hub; $passthru"
tmux new -d -s '$session' "\$env { \$boot $cmd ; } 2>&1 | tee -a $DATA_ROOT/logs/$session.log"
echo "started '$session'\${boot:+ (bootstrap first)}; log: $DATA_ROOT/logs/$session.log"
REMOTE
