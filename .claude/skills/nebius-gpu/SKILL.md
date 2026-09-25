---
name: nebius-gpu
description: Bring up, resume, monitor or stop the halohalo research GPU VM on Nebius (RTX PRO 6000, uk-south2) and run training jobs on it. Use when asked to start or restart cloud jobs, run the bake-off, check on a cloud run, recover from a preemption, or shut the VM down.
---

# Nebius GPU VM for halohalo

Everything is scripted in `scripts/nebius/`. Settings (project, VM name, GPU,
disk, SSH key) live in `scripts/nebius/vm.env`. Don't rediscover them with the
Nebius MCP unless a script fails.

The VM is **`halohalo-rtx6000-b`**, one RTX PRO 6000 (96 GB) with a **1 TB**
boot disk, preemptible. The first VM (`halohalo-rtx6000`, 500 GB) was deleted
on 2026-09-21 after its disk filled; see "Disk" below for why it was replaced
rather than grown.

## How to call the scripts

They run in WSL, where the Nebius CLI, SSH key and GitHub access live. From
Claude Code's Bash tool on Windows, prefix `MSYS_NO_PATHCONV=1` so Git Bash
does not rewrite the path:

    MSYS_NO_PATHCONV=1 wsl bash /mnt/d/halohalo/scripts/nebius/status.sh

Don't inline `$VAR`s in `wsl bash -c "..."`: Git Bash expands them on the
Windows side. Put anything non-trivial in a script file. Long heredocs passed
through the Bash tool lose backslashes (`\n` arrives as a newline): write
Python patches as files and run them, and build escapes with `chr(92)`.

## The flow

| Situation | Steps |
|---|---|
| Check on things | `status.sh`, or direct SSH if the CLI is signed out (below) |
| VM was preempted or stopped | `up.sh`, then `run.sh <session> '<cmd>'` (runs resume from checkpoints) |
| Code changed locally | `push.sh`, then restart the job's session |
| Brand-new VM | `VM_NAME=... up.sh --create`, `push.sh`, **user runs** `push_secrets.sh`, `run.sh` bootstrap, then **`bash scripts/idle_watchdog.sh --install`** |
| Done for now | `stop.sh` (GPU billing stops, disk kept) |

`run.sh <session> '<command>'` runs any command in `~/halohalo` in tmux, with
`HF_HOME`, `HF_XET_CACHE`, `FINETUNE_DIR`, `PLD_WORK_DIR` and `PLD_SOURCE=hub`
exported. Logs go to `/mnt/data/logs/<session>.log`. It runs
`scripts/bootstrap_nebius.sh` first whenever that script has changed.

## Rules

- **Money.** `up.sh` on a stopped VM starts billing (~$1.08/h). Confirm with
  the user first unless they have just asked for exactly that.
- **Never leave the meter to a poll.** This session only acts when a person
  writes or a task notification arrives, and background waiters time out at
  10 minutes. Two VMs once idled ~10 hours (~$20) that way. So:
  - the VM runs `scripts/idle_watchdog.sh` from root's cron: it powers off
    after 30 minutes with no tmux session, <1 GiB GPU memory and no SSH login.
    `touch /mnt/data/KEEP_AWAKE` holds it up deliberately;
  - chain dependent work into **one** tmux session or queue script, so the GPU
    moves to the next job without anyone being awake;
  - `stop.sh` explicitly when a queue is done rather than wait for the timer.
- **Secrets.** Never copy `.env` or `ts-wandb.txt` yourself. Ask the user to run
  `VM_NAME=... bash scripts/nebius/push_secrets.sh` in WSL. Never print them.
- **Deletion.** The Nebius MCP is in safe mode, and `update`/`delete` are
  blocked in it. Give the user the command; don't script it.
- **Killing processes.** Kill by PID or `tmux kill-session -t <name>`, never
  `pkill -f <pattern>`: patterns have matched the waiter's own command line.

## When the CLI is signed out

The federation login expires about daily. Symptoms: `status.sh` prints
"state unknown (CLI signed out)", `vm_state` returns empty, or a command hangs
trying to open a browser. WSL's Windows interop is often missing
(`fork/exec /mnt/c/windows/system32/cmd.exe: exec format error`), so the
browser cannot open by itself. The user runs, and leaves running:

    wsl nebius iam whoami --no-browser

and opens the printed link in a Windows browser. Meanwhile SSH still works:
the scripts accept `VM_IP=x.x.x.x`, cache the last IP in
`~/.cache/halohalo-nebius-ip`, and a direct call is

    ssh -i ~/.ssh/ts_carrot -o UserKnownHostsFile=~/.ssh/known_hosts_nebius tim@<ip>

Wrapping the CLI in `timeout` fails with "No such file or directory": it is
`~/.nebius/bin/nebius`, on PATH only after `common.sh` is sourced. A quick
"is it up" check that needs no CLI: `echo > /dev/tcp/<ip>/22`.

## Disk

- **A managed boot disk cannot be grown.** `nebius compute disk update` refuses
  ("disk is managed by the instance"); `instance update --patch
  --boot-disk-managed-disk-size-gibibytes` succeeds and records the new size in
  the spec, but the disk never changes — not after a stop, not after a start.
  The way to more disk is a new VM (`VM_NAME=... VM_DISK_GIB=1000 up.sh
  --create`) and an `rsync` of `/mnt/data` over the internal network (~140 MB/s
  VM to VM, with a throwaway key generated on the source VM).
- `/mnt/data` is a directory on the root disk, not a separate volume.
- What fills it: `checkpoint-*` inside finished runs (35 GB per whisper-large
  run, 22 GB per omni-1B, 42 GB per omni-7B). Once `final/` exists they are
  only resume state; delete them. The run scripts now do so themselves.

## Traps already hit (fixed in the scripts; keep them fixed)

- Preemptible VMs reject the default recovery policy: `--recovery-policy fail`.
- Blackwell GPUs need a PyTorch built for CUDA 12.8 or newer; the cu126 build has no kernels for them.
  `silero-vad` (livestream only) drags torch down, so bootstrap pins torch last.
- `huggingface-cli` no longer works; use `hf`, which reads `HF_TOKEN` from the env.
- The workstation `.env` points `HF_XET_CACHE` (and other paths) at
  `/home/carrot/...`. Bootstrap rewrites them, but secrets pushed *after*
  bootstrap undo that; `run.sh` exports the VM paths itself. Any ad-hoc command
  that uploads to the Hub must `export HF_XET_CACHE=/mnt/data/hf_cache/xet`,
  or the upload dies on a permission error.
- `sort -u -t= -k1,1` keeps the *first* duplicate key, so bootstrap deletes
  workstation paths from `.env` before appending VM paths.
- `git ls-files -co` includes the untracked 16 GB `FilipinoSpeechCorpus/` and
  old run folders; `push.sh` excludes them. `splits/` is gitignored and pushed
  explicitly. Without it the bake-off refuses to run.
- The public IP is not static and can change after a stop/start.
- `up.sh --create` could not create a VM under a new name: `vm_state` returned
  non-zero for "absent" and `set -e` exited before the create branch. Fixed.
- Logs are appended across runs. Wait on the **tmux session ending** or on an
  output file, never on `grep`ping the log — stale tracebacks from an earlier
  run have ended waits three times.
- `cmd | grep -q x` under `pipefail` reports failure when `grep` exits at its
  first match and the writer dies of SIGPIPE. Grep a finished file.
- Hub uploads of several multi-GB models in one batch can die on a timeout
  part-way. Push one repo at a time with a retry, then verify each with
  `curl -o /dev/null -w "%{http_code}" https://huggingface.co/api/models/<id>`.
- `qwen-tts` lives in its own `venv_qwen` (it pins a different torch); never
  install it into the training venv.
