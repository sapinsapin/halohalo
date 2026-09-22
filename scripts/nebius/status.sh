#!/usr/bin/env bash
# One-screen view: VM state, GPU, disk, running tmux jobs, log tails.
source "$(dirname "$0")/common.sh"

state="$(vm_state || true)"
ip="$(vm_ip || true)"
echo "$VM_NAME: ${state:-state unknown (CLI signed out)} $ip"
# with no CLI, SSH still answers the only question that matters: is it working?
[ "$state" = RUNNING ] || [ -n "$ip" ] || exit 0

vm_ssh 'bash -s' <<'REMOTE'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/gpu: /'
df -h /mnt/data | tail -1 | awk '{print "disk: " $3 " used, " $4 " free"}'
echo "jobs: $(tmux ls -F '#S' 2>/dev/null | paste -sd' ' || true)"
for log in /mnt/data/logs/*.log; do
    [ -f "$log" ] || continue
    echo "--- $(basename "$log") ($(date -r "$log" -u +%H:%M) UTC)"
    # progress bars rewrite one line with \r; keep only the latest state
    tail -c 4000 "$log" | tr '\r' '\n' | grep -v '^\s*$' | tail -4
done
run_log=$(ls -t /mnt/data/finetune_runs/logs/*.log 2>/dev/null | head -1)
if [ -n "$run_log" ]; then
    echo "--- latest run: $(basename "$run_log")"
    # the newest step counter and the last eval line
    tail -c 20000 "$run_log" | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+ ?(s/it|it/s)\]' | tail -1
    tail -c 200000 "$run_log" | tr '\r' '\n' | grep -E "'eval_(cer|loss)'" | tail -1
fi
summary=/mnt/data/finetune_runs/bakeoff_summary.txt
if [ -f "$summary" ]; then echo "--- bakeoff_summary.txt"; tail -8 "$summary"; fi
REMOTE
