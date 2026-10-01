# PRD: Phase 2 porting, real Arm numbers and the stalled GPU queue, on the GB10

| | |
|---|---|
| Status | Ready to execute |
| Written | 2026-09-29, on the Windows workstation (RTX 3070) |
| Executed by | a Claude Code session on the GB10 (DGX Spark class) |
| Code | `scripts/gb10.sh` (the runner), `scripts/port_phase2.sh`, `porting/`, `scripts/retrain_fleet_frozen.sh`, branch `bantaywika` |
| Parent docs | [porting_pipeline.md](porting_pipeline.md) (design), [porting_report.md](porting_report.md) (results so far), [prd_mac_porting.md](prd_mac_porting.md) (the Apple half), [status_2026-09-25.md](status_2026-09-25.md) (the queue items L1–L9, C1–C8) |

## 1. Why this exists

halohalo's speech models are being ported to every kind of hardware: Arm
CPUs, NPUs, Apple silicon, the browser and AMD. The pipeline in `porting/`
exports each model with the toolchain that reaches a target, then checks the
result against the original PyTorch model on the same held-out clips.

Phase 1 ran on the workstation (RTX 3070, 31 GB RAM). Phase 2 is the set of
models that do not fit there: Orpheus-3B in nine languages, the 7B CTC model,
and NPU calibration for everything above 500M parameters. It was scripted for
an RTX PRO 6000 cloud VM that is waiting on credit. The GB10 has 128 GB of
memory shared by CPU and GPU, enough for all of it.

It is also an Arm machine. So far every "Arm CPU" row in the report was
numerics checked on x86. The GB10's Cortex-X925 is the prime core of 2025
flagship phones (MediaTek's Dimensity 9400 has one). Four of them at four
threads is an upper bound for a phone CPU, not a match, but these are real
Arm timings with Arm kernels, including KleidiAI.

Third, the workstation's GPU queue stalled.

- **L3 and L7 finished** on 2026-09-27: the MMS zero-shot ASR baseline and
  Qwen3-TTS zero-shot. Do not repeat them.
- **L6 failed** before step 0. transformers has no SDPA attention for
  `Wav2Vec2BertForCTC`. `finetune_ctc.py` now switches w2v-bert to eager
  attention.
- **L5 hung for 2.5 days** at step 0 on its first language. DataLoader
  workers pass tensors through a Unix socket in `$TMPDIR`, and `/mnt/d/tmp`
  sits behind WSL's 9P bridge, which refuses sockets (`OSError: [Errno 95]`).

The GB10 takes over L6 and L5; its runs are the ones of record.

## 2. Goals

1. **Arm, measured.** The demo models on the Arm targets, run alone on four
   X925 cores: accuracy, parity with PyTorch, size and real-time factor
   (RTF). whisper.cpp is timed with and without KleidiAI. Orpheus TTS is
   timed on the same cores.
2. **Phase 1 everywhere.** Every Phase 1 model gets a 20-clip parity check on
   the Arm targets, not just the three demo models.
3. **Phase 2.** Calibrate the A16W8 NPU graphs for the 13 ASR models between
   500M and 2B parameters. Port the 7B CTC model. Run Orpheus × 9 through
   GGUF, ONNX GenAI, OpenVINO and ExecuTorch (MediaPipe best effort), with an
   end-to-end check per runtime scored by an independent judge.
4. **The queue.** L6 diagnoses the w2v-BERT collapse; L5 retrains the
   whisper-small fleet on the frozen splits.
5. **Use the memory.** Wherever there is room, run bigger batches or more jobs
   at once, by the policy in section 5, and measure whether it paid off.
6. **Report and return.** Regenerate `docs/porting_report.md` with the
   `@gb10` rows, and pack the results for the workstation.

### Non-goals

- Publishing anything to Hugging Face or elsewhere: models, artefacts or
  results. The user decides.
- Apple targets (the Mac's PRD) and browser rows (the workstation's Chrome).
- Changing a recipe that has to stay comparable. L5 keeps the published
  fleet's effective batch of 16.
- The tier-3 research runs (section 6.3), until the user says go.

## 3. The machine and what it implies

| | GB10 |
|---|---|
| CPU | 20 Arm cores: 10 Cortex-X925 + 10 Cortex-A725 (aarch64) |
| GPU | Blackwell, compute capability 12.1 (sm_121), CUDA 13 |
| Memory | 128 GB LPDDR5x, **unified**: the GPU allocates from the same pool; ~273 GB/s |
| OS | DGX OS (Ubuntu 24.04) |

- **One pool.** A GPU job's memory is system memory. `nvidia-smi` may report
  memory as N/A, so the scripts measure with `/proc/meminfo` (MemAvailable)
  and `free -g`.
- **Bandwidth-bound.** 273 GB/s is about a seventh of the RTX PRO 6000's, so
  single-stream LLM decoding (Orpheus) will be slow. Spare memory helps with
  throughput (parallel streams, batches, more jobs), not single-stream
  latency.
- **aarch64 wheels.** CUDA torch comes from `download.pytorch.org/whl/cu130`,
  the scripts' `TORCH_INDEX` default. Known gaps:
  - MLC/WebLLM ships no aarch64 CUDA wheels: skipped.
  - ai-edge-torch/MediaPipe and onnxruntime-genai: best effort, with any
    failure recorded.
  - bitsandbytes and flash-attn: not used, because the GB10 recipes need
    neither 8-bit optimisers nor flash attention.

## 4. Inputs

1. **This repo**, branch `bantaywika`.
2. **`halohalo-gb10-inputs.tar`** (47 MB). The user copies it from the
   workstation (`D:\halohalo\finetune_runs\port\gb10\`) and untars it in the
   repo root. It holds what a clone cannot rebuild:
   - `splits/pld_*.json`, the frozen speaker- and prompt-disjoint split specs
     for all ten languages;
   - the ceb and pam evaluation packs;
   - every result file so far, including the PyTorch reference transcripts;
   - the TTS sentence manifest.

   **It contains PLD corpus text and audio (CC-BY-NC, research only): never
   commit it, upload it or paste its contents anywhere.**

   If the split specs are missing, stop and ask for the pack. The data loader
   does not fail without them: it falls back to the published random split,
   whose test speakers overlap training. L5 would then train on test
   speakers, which is exactly the contamination it exists to remove. `gb10.sh`
   refuses to run without them.
3. **Hugging Face access.** PLD is a private dataset. After `setup` creates the
   venv, the **user** runs `venv/bin/hf auth login`, or writes a `.env` with
   `HF_TOKEN=` themselves. The session never asks for, reads, prints or copies
   the token. Do not copy the workstation's `.env`: its paths are Windows
   mounts. `gb10.sh` overrides them anyway.
4. **Disk:** about 700 GB free under `finetune_runs/`. That covers Orpheus
   artefacts (~30 GB a language), the 7B ports (~100 GB) and the PLD
   download (~25 GB). `setup` checks. To use another disk, point
   `GB10_FINETUNE_DIR` at it.
5. **System packages:** `git cmake ffmpeg tmux build-essential python3.12-venv
   python3.12-dev` and the CUDA toolkit (`nvcc`). `setup` lists whatever is
   missing. The user installs them with sudo; the session does not run sudo.

## 5. Memory and parallelism policy

The workstation's recipes are full of compromises for a 6 GB card. On the
GB10 they cost speed, and some cost accuracy, for nothing. The rules:

- **M1. Spend spare memory in this order:**
  1. **Drop the memory compromises.** Use fp32 master weights, not
     `--bf16-weights`. Use fused AdamW, not `adamw_bnb_8bit`. Turn gradient
     checkpointing off (`--no-grad-checkpoint`). Do not load trained models
     4-bit or 8-bit (`--no-quant` for Orpheus).
  2. **Raise the per-device batch and cut gradient accumulation, at the same
     effective batch.** Faster, same experiment. L5: 16 × 1 instead of 8 × 2.
     L6: 16 × 1 instead of 2 × 8.
  3. **Run independent jobs side by side** (languages, learning rates,
     models), each started only while its expected peak plus the reserve
     (M2) still fits.
  4. **Raise the effective batch only for new experiments.** Scale the
     learning rate (linear, or square root for Adam; say which), and write
     both into the run's result and section 9. Never do this for reruns that
     must match a published recipe (L5, the L6 control) or for bake-off
     comparisons.
  5. **Batch inference.** Orpheus decodes four sentences at once: llama.cpp
     server slots (`ORPHEUS_PARALLEL`) and torch generate batches
     (`ORPHEUS_BATCH`).
- **M2. Keep a reserve.** No job starts if it would leave less than 16 GB
  free (`MIN_FREE_GB`, `PORT_MIN_FREE_GB`). The porting orchestrator gates
  each start on the model's expected peak (`porting/__main__.py: est_gb`):
  conversion about 3× the fp32 weights, static int8 calibration about 7×, a
  check about 1.5×. Starts are serialised and staggered (`PORT_STAGGER_S`),
  so each gate sees the previous job's memory. `mem_guard.sh` is the last
  resort: below 8 GB free it ends the largest *porting* job, which can
  resume. It never ends a training run.
- **M3. Measure, don't assume.** A second job on one GPU helps only if the
  first leaves the GPU idle.
  - `gb10.sh probe` runs whisper-small at the L5 recipe with 1, 2 and 3
    copies at once, and measures steady-state time per step. The job count
    goes up only while each extra job adds at least 15% throughput. The
    result goes to `finetune_runs/gb10/probe.env`, which L5 reads.
  - `monitor.tsv` logs memory, GPU utilisation and power every 30 s,
    alongside what each lane is doing.
  - After the first hour of each lane, read it:
    - If used memory peaks under ~70 GB and GPU utilisation averages under
      70%, raise that lane's job count by one at the next restart. Every
      step resumes, so restarting is safe.
    - If MemAvailable dipped under 16 GB, or `mem_guard` fired, lower it.
- **M4. Speed numbers come from quiet runs.** Parallel runs are for
  correctness: CER and parity. Their RTF and tokens/s are recorded but are not
  headline numbers, because on unified memory any other job takes bandwidth.
  The headline Arm numbers (`arm`, `arm-tts`) run alone, pinned to four X925
  cores, at four threads.
- **M5. Write it down.** Every change to a job count, batch or learning rate
  goes into section 9, with the monitor numbers that justified it.

### Knobs (environment variables)

| knob | default | controls |
|---|---|---|
| `MIN_FREE_GB`, `PORT_MIN_FREE_GB` | 16 | the reserve no job may take (M2) |
| `CPU_JOBS` | 4 | Phase 1 models ported and checked side by side |
| `NPU_JOBS` | 3 | NPU calibrations side by side |
| `PORT_STAGGER_S` | 60 | seconds between porting starts |
| `JOBS_GB10`, `JOB_GB` | from `probe` | fleet languages at once, and each one's memory |
| `L6_JOBS`, `L6_GB` | 3, 40 | w2v-BERT runs at once, and each one's memory |
| `ORPHEUS_JOBS`, `NEED_GB` | 2, 40 | Orpheus languages at once; memory an export step waits for |
| `ORPHEUS_PARALLEL`, `ORPHEUS_BATCH` | 4, 4 | sentences decoded at once per check |
| `NPU_CALIB_MAX_M` | 2000 | largest model (M params) given NPU calibration; the 7B is left out |
| `GUARD_KB` | 8000000 | `mem_guard` threshold (kB available) |

### Expected peaks (estimates; replace them with measured ones in section 9)

| job | expected peak |
|---|---|
| whisper-small training, batch 16, no checkpointing | 15–25 GB (`probe` measures it) |
| w2v-BERT 600M CTC training, batch 16, fp32, no checkpointing | 30–40 GB |
| Orpheus export step (ExecuTorch, NNCF int4 the largest) | up to ~40 GB; merge ~15 GB |
| Orpheus check: torch bf16 batch 4, or llama.cpp with 4 slots, plus the MMS judge | 10–15 GB |
| ONNX export: whisper-small / 1B CTC / whisper-large-v3 / 7B CTC | ~3 / ~11 / ~17 / ~78 GB |
| A16W8 calibration: 1B CTC (measured) / whisper-large-v3 | 25 GB / ~40 GB |
| 7B CTC check on CPU (fp32) | ~40 GB |

## 6. The work

### 6.1 Tier 1: porting

| step | what | acceptance |
|---|---|---|
| `arm` | The demo models (whisper-small ceb, omniASR 1B ceb-norm, SNAC) built and checked on `arm-cpu-onnx`, `arm-cpu-ggml`, `arm-mobile-executorch`, `npu-qnn`/`npu-ryzenai` (numerics) and `npu-openvino`. Checks run one at a time, pinned to four X925 cores; whisper.cpp runs again without KleidiAI (`@gb10-generic`). | ORT fp32 CER within 0.5 points of PyTorch on the same clips; int8 within 1 point. whisper.cpp within 0.5 points of the workstation's whisper.cpp row. RTF recorded for every row. The KleidiAI speed-up is stated as a ratio. |
| `cpu-lane`, part 1 | Every Phase 1 model on the same targets, 20 clips each | Same thresholds; on 20 clips, flag gaps over 1 point rather than failing them |
| `cpu-lane`, part 2 | A16W8 NPU graphs for whisper-large-v3 × 9 and omniASR 1B × 4 | qdq16 within 1 point of PyTorch on 20 clips; report either way |
| `cpu-lane`, part 3 | The 7B CTC model: ONNX fp32/int8, OpenVINO, ExecuTorch | Each one exports and checks, or its failure is written down. "Does not fit in 128 GB" is a result. |
| `gpu-lane`, last | Orpheus × 9: merge, GGUF, ONNX GenAI, OpenVINO, ExecuTorch, MediaPipe. Checks: torch bf16 (the reference), llama.cpp q8_0 and q4_k_m on the GPU, ONNX GenAI int4 and OpenVINO int4 on the CPU; 10 sentences each | No empty outputs. Judge CER recorded; flag a port more than 3 points worse than torch on the same sentences (10 sampled sentences make smaller gaps noise). tokens/s recorded; real time needs ~86 per stream. |
| `arm-tts` | Orpheus ceb, llama.cpp q4_k_m, CPU only, four X925 cores, 3 sentences | tokens/s recorded (`@gb10-cpu`); state how far from real time it is |

### 6.2 Tier 2: the queue

| item | run | acceptance |
|---|---|---|
| **L6** w2v-BERT collapse | ceb, char units, 4000 clips, 500 steps, learning rates 1e-5, 3e-5 and 1e-4 (the bake-off's, as the control), side by side. Batch 16 × 1, fp32, fused AdamW, no checkpointing. | Each run writes `result.json`. Section 9 answers: does the loss move at the lower learning rates, and does the control collapse as the bake-off arms did? |
| **L5** whisper-small fleet on the frozen splits | 10 languages, 2000 steps, 10k clips, effective batch 16 (the published recipe), `JOBS_GB10` at once | A `result.json` per language with `"split": "frozen-disjoint"` (anything else means the splits were missing: stop). Summary table beside the published CERs. |
| L3, L7 | **done on the workstation** 2026-09-27 | nothing to do |

### 6.3 Tier 3: research runs, only when the user says go

Start these once tiers 1 and 2 are done. For each, write a short plan in
section 9 and ask the user first. They follow M1: none of the old memory
compromises.

- **C3** w2v-BERT, four bake-off arms. Only if L6 shows the loss moves.
- **L4 → C2** Qwen3-TTS finetune. First debug on 0.6B: run upstream's SFT
  unmodified, then swap in our collate, then our save. Then re-finetune
  1.7B. Build `venv_qwen` with `TORCH_INDEX` set; `setup_qwen_venv.sh`
  installs CUDA torch from it first.
- **C8** One multilingual Orpheus adapter with language tags. This needs a
  small change to the ablation runner. Use the `--cloud` preset of
  `finetune_orpheus.py` (no quantisation, no checkpointing, fused optimiser)
  at the per-language adapters' effective batch.
- **C7** Fish S2 Pro. No code exists yet; a smoke test at most.
- **G4** Serve Gemma 4 for NYO: `scripts/custom-models/gemma4-gb10.sh` in
  the `aineolab/llm-gateway-hub` repo (`setup`, `serve`, `smoke`, `tunnel`,
  `nyo`). llama.cpp with a 4-bit GGUF (26B-A4B, ~16 GB), on :8081, reusing
  this PRD's llama.cpp build; it refuses to start unless the 16 GB reserve
  (M2) survives. Record tokens/s and peak memory in section 9. The tunnel
  URL goes to the NYO owner, never into a file here.

## 7. How to run it

```bash
git clone git@github.com:sapinsapin/halohalo.git && cd halohalo && git checkout bantaywika
tar xf ~/halohalo-gb10-inputs.tar                  # wherever the user put it
bash scripts/gb10.sh setup                         # stops and says what is missing (packages, HF login)
tmux new -d -s gb10 'bash scripts/gb10.sh all'     # everything below, in order
bash scripts/gb10.sh status                        # any time: memory, GPU, jobs, progress
```

What `all` does, and roughly how long each part takes:

1. `setup`: about 30 min.
2. `evalpack`: about 1 h. It downloads PLD, about 25 GB, which the training
   runs then reuse.
3. `arm`: about 2 h.
4. `probe`: about 30 min.
5. Then two lanes side by side:
   - **CPU lane:** Phase 1 everywhere, then NPU calibration, then the 7B.
   - **GPU lane:** L6, then L5, then Orpheus × 9.

   The lanes take about a day together.
6. `arm-tts`, `report`, `pack`.

Each part can also be run by name:

```bash
bash scripts/gb10.sh cpu-lane       # or gpu-lane, l6, l5, orpheus, arm, probe, arm-tts, report, pack
```

Logs are in `finetune_runs/gb10/`:
- `summary.txt`
- `cpu_lane.log`, `gpu_lane.log`, `orpheus.log`
- `monitor.tsv`, `mem_guard.log`, `probe/`

Per-job logs are in `finetune_runs/port/logs/`, `finetune_runs/port/phase2_orpheus_<lang>.log`,
`finetune_runs/fleet_frozen/` and `finetune_runs/gb10_queue/`. Every step
skips finished work and continues past failures, so rerunning any step, or
`all`, is safe.

**Results back to the workstation:** `pack` writes
`finetune_runs/gb10/halohalo-gb10-results.tar`, which holds result JSON,
logs, build records, the ten evaluation packs and the monitor data. It has no
weights. The user copies it by hand and untars it in `D:\halohalo`. Then
`python -m porting report` there shows every host's rows together.

## 8. Known pitfalls

- **sm_121.** If nvcc rejects `CMAKE_CUDA_ARCHITECTURES=121`, set
  `CUDA_ARCH=121a` or `CUDA_ARCH=native`. If torch warns that sm_121 is
  unsupported, `setup`'s matmul test fails. Try a newer cu130 wheel, or
  NVIDIA's PyTorch container, before anything else.
- **Wheels missing on aarch64** (onnxruntime-genai, ai-edge-torch/MediaPipe,
  some ExecuTorch or OpenVINO builds). The step fails and says so. Record the
  failure, don't fight it for more than a few minutes, and move on.
- **ExecuTorch's Python runtime is single-threaded** for large models. The
  orchestrator caps those checks at 20 clips.
- **The 7B conversion is the tightest fit.** The gate holds it until about
  90 GB is free (73 GB expected, plus the 16 GB reserve). If `mem_guard`
  still ends it, that is the result: record it. The 7B gets no NPU graph:
  its calibration would need about 180 GB.
- **Tausug.** MMS-1b-all has no Tausug adapter. The Orpheus check falls back
  to our own whisper-small judge and records `"judge_independent": false`.
  Say so wherever the number is used.
- **`nvidia-smi` memory is N/A** on unified memory. Trust MemAvailable. Page
  cache counts as available, so a large cache is not pressure.
- **TMPDIR must be a local disk.** It is `finetune_runs/tmp` here. DataLoader
  workers put Unix sockets in it, and a network or 9P mount breaks them
  silently: that is what hung the workstation's L5.
- **Contaminated numbers.** A fleet `result.json` whose `split` is not
  `frozen-disjoint` trained on test speakers. Discard it, fix the splits and
  rerun.

## 9. Findings (fill in while running)

<!-- One line per check that failed, missed its threshold, or surprised you,
     and per change to a job count / batch / learning rate (with the monitor
     numbers behind it). Numbers belong in the report; the why belongs here. -->

## 10. Done means

1. `all` has run to the end, or each step has run separately. Tier-1 and
   tier-2 failures are recorded in section 9.
2. `docs/porting_report.md` is regenerated and shows the `@gb10`,
   `@gb10-generic` and `@gb10-cpu` rows.
3. Section 9 records:
   - the probe result;
   - the peak memory and GPU use of each lane;
   - every change to job counts or batches;
   - the KleidiAI ratio;
   - the L6 answer.
4. The L5 summary table (ten languages, frozen-disjoint) is in section 9,
   beside the published fleet's CERs.
5. Code fixes are committed with messages that say what broke and why.
6. Everything is pushed to `bantaywika`, or to a branch off it if the user
   prefers. Never push to `main` without asking.
7. The user gets `halohalo-gb10-results.tar`. It holds test transcripts, so
   copy it by hand and never commit it.

## 11. Guardrails for the session

- **Artefacts and checkpoints stay on the GB10** until the user says
  otherwise: Orpheus exports, 7B ports, the fleet's `final/` directories and
  the L6 runs.

- **Commits.** Commit code, docs and the regenerated report. Never commit
  `finetune_runs/`, `third_party/`, `venv*/`, `splits/`, the tars, audio or
  transcripts. Check `git status` before each commit.
- **Secrets.** The Hugging Face login is the user's. Never print, copy or
  create tokens. W&B stays off (`WANDB_MODE=disabled`).
- **No publishing.** No `--push`, no `push_to_hub`, no uploads of models,
  artefacts or results.
- **Processes.** Stop only processes you started, by PID or process group.
  Never `pkill -f`: a pattern can match your own shell.
- **System.** No sudo and no system changes (power modes, swap, drivers). If
  something needs them, tell the user the command.
- **Git.** Commits end with the `Co-Authored-By` line the session's
  instructions give. Use the same git author as this branch's history
  (`git log -1 --format='%an <%ae>'`) unless the user says otherwise.
