#!/usr/bin/env bash
# Power the VM off when it has been idle too long. Runs on the VM, from cron.
#
#   bash scripts/idle_watchdog.sh --install      # every 5 minutes, as root's cron
#   bash scripts/idle_watchdog.sh --status
#   touch /mnt/data/KEEP_AWAKE                   # hold it up deliberately
#
# Why this exists: on 2026-09-20 two RTX PRO 6000 VMs sat idle for about ten
# hours, roughly $20, because the session driving them was waiting on a person
# and nothing on the machines knew they were no longer needed. An agent that
# only runs when woken cannot be the thing that stops the meter. The machine
# has to.
#
# "Idle" means all three, for IDLE_MINUTES in a row: no tmux session (every job
# here runs in one), GPU memory under 1 GiB, and no SSH login. A shutdown from
# inside is a stop, not a delete: the disk and everything on it survive, and
# scripts/nebius/up.sh starts it again.
set -uo pipefail

IDLE_MINUTES=${IDLE_MINUTES:-30}
STATE=/var/tmp/idle_watchdog.count
KEEP=/mnt/data/KEEP_AWAKE
SELF="$(readlink -f "$0")"

case "${1:-}" in
  --install)
    line="*/5 * * * * IDLE_MINUTES=$IDLE_MINUTES bash $SELF >> /var/log/idle_watchdog.log 2>&1"
    ( sudo crontab -l 2>/dev/null | grep -vF idle_watchdog.sh; echo "$line" ) | sudo crontab -
    echo "installed: powers off after $IDLE_MINUTES idle minutes"; exit 0 ;;
  --status)
    echo "idle checks in a row: $(cat $STATE 2>/dev/null || echo 0) (x5 min; limit $IDLE_MINUTES min)"
    [ -e "$KEEP" ] && echo "KEEP_AWAKE is set"; exit 0 ;;
esac

busy=""
[ -e "$KEEP" ] && busy="KEEP_AWAKE"
# tmux sessions belong to the login user, not to root's cron
for u in $(ls /home); do
    sudo -u "$u" tmux ls >/dev/null 2>&1 && busy="tmux($u)"
done
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
[ "${mem:-0}" -gt 1024 ] && busy="gpu ${mem}MiB"
[ -n "$(who)" ] && busy="ssh login"

if [ -n "$busy" ]; then
    echo 0 > "$STATE"
    exit 0
fi

n=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$STATE"
echo "$(date -u +%F' '%T) idle check $n"
if [ $(( n * 5 )) -ge "$IDLE_MINUTES" ]; then
    echo "$(date -u +%F' '%T) idle for $(( n * 5 )) min: powering off"
    echo 0 > "$STATE"
    shutdown -h now
fi
