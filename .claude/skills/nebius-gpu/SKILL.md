---
name: nebius-gpu
description: Bring up, resume, monitor or stop the halohalo research GPU VM on Nebius (RTX PRO 6000, uk-south2) and run training jobs on it. Use when asked to start or restart cloud jobs, run the bake-off, check on a cloud run, recover from a preemption, or shut the VM down.
---

# Nebius GPU VM for halohalo

Everything is scripted in `scripts/nebius/`. Settings (project, VM name, GPU,
disk, SSH key) live in `scripts/nebius/vm.env`. Don't rediscover them with the
Nebius MCP unless a script fails.

## How to call the scripts

They run in WSL, where the Nebius CLI, SSH key and GitHub access live. From
Claude Code's Bash tool on Windows, prefix `MSYS_NO_PATHCONV=1` so Git Bash
does not rewrite the path:

    MSYS_NO_PATHCONV=1 wsl bash /mnt/d/halohalo/scripts/nebius/status.sh

Don't inline `$VAR`s in `wsl bash -c "..."`: Git Bash expands them on the
Windows side. Put anything non-trivial in a script file.

## The flow

| Situation | Steps |
|---|---|
| Check on things | `status.sh` |
| VM was preempted or stopped | `up.sh`, then `run.sh bakeoff` (resumes from checkpoints, skips finished runs) |
| Code changed locally | `push.sh`, then restart the job's session |
| Brand-new VM | `up.sh --create`, `push.sh`, **user runs** `push_secrets.sh`, `run.sh smoke`, `run.sh bakeoff` |
| Done for now | `stop.sh` (GPU billing stops, disk kept) |

`run.sh` runs `scripts/bootstrap_nebius.sh` automatically when it has not
completed for the current version of that script. On a new VM that includes
downloading ~30 GB of PLD+FSC, about 5 minutes on Nebius.

`run.sh <session> '<command>'` runs any command in `~/halohalo` in tmux, with
`HF_HOME`, `FINETUNE_DIR`, `PLD_WORK_DIR` and `PLD_SOURCE=hub` exported. Logs
go to `/mnt/data/logs/<session>.log`; results to `/mnt/data/finetune_runs`.

## Rules

- **Money.** `up.sh --create` and `up.sh` on a stopped VM start billing
  (preemptible RTX PRO 6000 was ~$1.08/h in Sept 2026). Confirm with the user
  first unless they have just asked for exactly that.
- **Secrets.** Never copy `.env` or `ts-wandb.txt` yourself; the permission
  classifier blocks it, rightly. Ask the user to run
  `bash scripts/nebius/push_secrets.sh` in WSL. Never print their contents.
- **Deletion.** The Nebius MCP is in safe mode. Give the user
  `nebius compute instance delete --id <id>` to run; don't script it.
- **Killing processes.** Kill by PID, never `pkill -f <pattern>`: patterns
  have matched the WSL session itself before.

## Traps already hit (fixed in the scripts; keep them fixed)

- Preemptible VMs reject the default recovery policy: `--recovery-policy fail`.
- Blackwell GPUs need a PyTorch built for CUDA 12.8 or newer; the cu126 build has no kernels for them.
  `silero-vad` (livestream only) drags torch down, so bootstrap pins torch last.
- `huggingface-cli` no longer works; use `hf`, which reads `HF_TOKEN` from the env.
- `sort -u -t= -k1,1` keeps the *first* duplicate key, so bootstrap deletes
  workstation paths from `.env` before appending VM paths.
- `git ls-files -co` includes the untracked 16 GB `FilipinoSpeechCorpus/` and
  old run folders; `push.sh` excludes them. `splits/` is gitignored and pushed
  explicitly. Without it the bake-off refuses to run.
- The public IP is not static and can change after a stop/start. The scripts
  look it up each time; host keys go to `~/.ssh/known_hosts_nebius`.
