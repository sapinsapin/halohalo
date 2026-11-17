# halohalo handoff (2026-11-17)

Open work in this repo, written so someone on another machine can pick it up. This repo is public: tasks that
involve partners, licensing or infrastructure access are tracked privately by the maintainers and are not
listed here.

## State
- Working branch: `chika-curation-workflow`, in sync with origin and well ahead of `main`.
- Tracking issue for curation: #8. Machine-readable state: `curation/STATUS.json`.
- Contributor MCP notes: `docs/contributor_mcp_handoff.md`.

## Pending tasks

### Chika curation
- [ ] **Take Chika curation to 250 retained hours** each for Filipino/Taglish, Hiligaynon, Cebuano, Bikol and
  Ilocano. Progress is in `curation/STATUS.json` (last updated 2026-11-15; check it against the workers before
  trusting it).
- [ ] **Plan the move of the curation supervisor.** The supervisor and its workers run on one workstation and
  the raw audio is local to it. Moving machines means moving both, or restarting from the published chunks.
- [ ] **Resume the paused transcription batches only with the maintainer's go-ahead.** The bulk batch stopped
  after finalize timeouts, and the Hiligaynon batch was stopped on 2026-11-16.
- [ ] **Keep unreviewed submissions out of every training split.** The dataset stays private until the
  maintainers release it.

### Porting to Apple hardware (`porting/`)
- [ ] **Validate the Mac builds against PyTorch.** Blocked on the evaluation archive from the training
  workstation. After that: regenerate `docs/porting_report.md` and package the results.
- [ ] **Revisit whisper-small on ExecuTorch Core ML** when the upstream conversion issue is fixed
  (coremltools 9.0 with torch 2.14.1).
- [ ] **Reconcile the `bantaywika` branch** with origin; a local copy has diverged in both directions.

### Pull requests
- [ ] **Merge, finish or close PRs #5, #6 and #7.** #5 is open; #6 and #7 have been drafts since October.

## Finished recently
- Orpheus MLX and the 1B CTC model on ExecuTorch Core ML were built.
- Porting fixes `708ac74` and `0ee4030` are on `chika-curation-workflow`.
- The first Filipino and Hiligaynon transcription batches were completed and checked against their gate sets.
