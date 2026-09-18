#!/usr/bin/env bash
# Copy .env (HF_TOKEN) and ts-wandb.txt to the VM, owner-only. Run this
# yourself: Claude Code is not allowed to move credentials to another machine.
# Only needed once per new VM; the files survive stop/start and preemption.
source "$(dirname "$0")/common.sh"
cd "$REPO"
for f in .env ts-wandb.txt; do [ -f "$f" ] || { echo "!! $REPO/$f missing"; exit 1; }; done
tar -cf - .env ts-wandb.txt | vm_ssh 'mkdir -p ~/halohalo && umask 077 && tar -xf - -C ~/halohalo && chmod 600 ~/halohalo/.env ~/halohalo/ts-wandb.txt'
vm_ssh 'cd ~/halohalo && stat -c "%A %s %n" .env ts-wandb.txt'
