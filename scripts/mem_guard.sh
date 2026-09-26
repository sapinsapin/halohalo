#!/usr/bin/env bash
# Kill the biggest porting job before the kernel's OOM killer does.
#
# In WSL an OOM kill anywhere fails systemd's init.scope, which holds every
# wsl.exe process of every session; the scope then SIGKILLs each new session
# after 90 s until `wsl --shutdown` (seen 2026-09-26). Ending one export early
# costs one resumable step instead of every job on the machine.
#
#   Start-Process -WindowStyle Hidden wsl.exe -ArgumentList 'bash /mnt/d/halohalo/scripts/mem_guard.sh'
MIN_KB=${MIN_KB:-3000000}          # act below ~3 GB available
LOG=/mnt/d/halohalo/finetune_runs/port/mem_guard.log
while :; do
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$avail" -lt "$MIN_KB" ]; then
    victim=$(ps -eo pid,rss,cmd --sort=-rss | grep -E "[p]orting\.(export|validate)" | head -1)
    if [ -n "$victim" ]; then
      pid=$(echo "$victim" | awk '{print $1}')
      echo "$(date '+%F %T') MemAvailable ${avail} kB < ${MIN_KB}: terminating $(echo "$victim" | cut -c1-150)" >> "$LOG"
      kill "$pid"
      sleep 20
    fi
  fi
  sleep 5
done
