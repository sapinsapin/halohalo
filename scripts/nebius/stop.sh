#!/usr/bin/env bash
# Stop the VM: GPU billing ends, the disk (and everything on /mnt/data) is kept
# and still billed. up.sh starts it again; run.sh then resumes jobs.
#
# Deleting the VM (and its disk) is deliberately not scripted. Claude Code's
# Nebius access runs in safe mode, so do that yourself:
#   nebius compute instance delete --id <id printed below>
source "$(dirname "$0")/common.sh"

id="$(vm_id)"
[ -n "$id" ] || { echo "$VM_NAME does not exist"; exit 0; }
if [ "$(vm_state)" = RUNNING ]; then
    vm_ssh 'tmux ls 2>/dev/null' && echo "!! the jobs above will be interrupted (they resume from checkpoints)" || true
fi
nebius compute instance stop --id "$id" >/dev/null
echo "stopped $VM_NAME ($id)"
