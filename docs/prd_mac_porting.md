# PRD: the porting pipeline's Apple targets, run on a Mac

| | |
|---|---|
| Status | Ready to execute |
| Written | 2026-09-27, on the Windows workstation (RTX 3070) |
| Executed by | a Claude Code session on the user's Mac |
| Code | `porting/`, `scripts/port_mac.sh`, branch `bantaywika` |
| Parent docs | [porting_pipeline.md](porting_pipeline.md) (design), [porting_report.md](porting_report.md) (results so far) |

## 1. Why this exists

halohalo's speech models are being ported to every kind of hardware: Arm
CPUs, NPUs, Apple silicon, the browser and AMD. The pipeline in `porting/`
exports each model with the toolchain that reaches a target, then checks the
result against the original PyTorch model on the same held-out clips.

The workstation built the Apple artefacts but could not really check them.
Core ML predicts only on macOS. ExecuTorch's Core ML backend exports only on
macOS. MLX ran there on its x86 CPU backend at a real-time factor of 50, which
proves the weights are right and says nothing about a Mac. This PRD moves the
Apple targets to a Mac, where they can be built natively, checked properly
and timed on the hardware they are for.

## 2. Goals

1. **Build** every Apple artefact on the Mac from the public Hugging Face
   checkpoints, with the same exporters the workstation uses.
2. **Check** each one against the PyTorch reference: accuracy (CER/WER
   against the human transcript) and parity (CER against PyTorch's own
   transcript), plus size and real-time factor on Apple silicon.
3. **Measure** what only a Mac can: Neural Engine against CPU for Core ML,
   Metal speed for MLX, whisper.cpp and llama.cpp, and whether Orpheus TTS
   reaches real time on a Mac.
4. **Report** in the shared report, with Mac rows next to the workstation's,
   and fix whatever breaks in the code on the way.

### Non-goals

- Publishing anything to Hugging Face or elsewhere.
- Phase 2 models beyond Orpheus Cebuano: whisper-large-v3, the 7B CTC model and
  NPU calibration wait for the RTX PRO 6000.
- iOS or on-device app packaging. The .pte and .mlpackage files are what an
  app would ship; building the app is later work.
- Retraining or changing any model.

## 3. What is in scope

Three demonstration models, one per family, plus the TTS pair:

| model (Hugging Face, all public) | family | params |
|---|---|---|
| `sapinsapin/whisper-small-pld-ceb` | Whisper (encoder-decoder ASR) | 242M |
| `sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm` | wav2vec2 CTC ASR | 963M |
| `hubertsiuzdak/snac_24khz` | SNAC audio decoder (for Orpheus) | 20M |
| `sapinsapin/orpheus-3b-0.1-pretrained-char-pld-ceb` | Orpheus TTS: LoRA on `unsloth/orpheus-3b-0.1-pretrained` | 3.3B |

Against these Apple targets:

| target (registry id) | runtime | artefact | built by |
|---|---|---|---|
| mac-mlx | mlx-whisper, mlx-lm on Metal | MLX safetensors, fp16 / 8-bit / 4-bit | `porting.export_mlx`, `export_orpheus --steps mlx` |
| npu-ane | Core ML, Neural Engine | `.mlpackage` fp16 | `porting.export_coreml`, `export_snac --target coreml` |
| mac-executorch | ExecuTorch, Core ML backend | `.pte` | `porting.export_executorch --backend coreml` |
| arm-cpu-ggml on Apple | whisper.cpp and llama.cpp on Metal | ggml / GGUF | `porting.export_ggml`, `export_orpheus --steps merge gguf` |
| web-webgpu on Safari (stretch) | onnxruntime-web / Transformers.js | ONNX fp16 / q4f16 | `porting.export_onnx` |

## 4. Inputs

1. **This repo**, branch `bantaywika`.
2. **The evaluation pack**, `halohalo-mac-eval.tar` (12 MB). It holds what a
   clone cannot rebuild: the 100 frozen-test Cebuano clips
   (`finetune_runs/port/evalpack/ceb.npz`), the workstation's result files
   including the PyTorch reference transcripts, and the TTS sentence manifest.
   The frozen split's definitions live in `splits/`, kept out of git because
   they contain corpus text. The user copies the file from the workstation
   (`D:\halohalo\finetune_runs\port\mac\halohalo-mac-eval.tar`) and untars it
   in the repo root. **It contains PLD test audio and transcripts (CC-BY-NC,
   research only): never commit it, upload it or paste its text anywhere.**
3. **Optional**: `halohalo-mac.tar` (5.8 GB), the same pack plus the
   artefacts the workstation already built. With it, `build` only fills gaps.

If the evaluation pack is missing, stop and ask the user for it. Do not try
to rebuild it from the PLD dataset: without the frozen split files, the test
set would silently differ and every comparison would be wrong.

## 5. Requirements

### Environment

- R1. Apple silicon (M1 or later), macOS 14 or newer, 16 GB RAM or more
  (Orpheus merge needs about 14 GB), about 40 GB free disk.
- R2. Xcode command line tools (`xcrun coremlc`, clang), Homebrew
  `python@3.12` and `cmake`, git. Node 20+ only for the Safari stretch goal.
- R3. Two venvs, never mixed: `venv_mac` (MLX, coremltools, torch,
  transformers) and `venv_mac_et` (ExecuTorch pins its own torch).
- R4. Network access to Hugging Face (public models, no token needed) and
  GitHub (whisper.cpp, llama.cpp, openai/whisper).

### Building

- R5. `bash scripts/port_mac.sh setup` then `build` produces, under
  `finetune_runs/port/artefacts/`:
  - `whisper-small-pld-ceb/mlx/{fp16,q8,q4}/`, `coreml/encoder.mlpackage`,
    `ggml/ggml-model-{f16,q8_0,q5_0}.bin`, and the compiled
    `ggml/ggml-model-f16-coreml-encoder.mlmodelc`
  - `omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm/coreml/model.mlpackage`
  - `snac_24khz/coreml/decoder.mlpackage`
  - `orpheus-3b-0.1-pretrained-char-pld-ceb/merged/` and `gguf/model-{f16,q8_0,q4_k_m}.gguf`
- R6. `executorch` produces `executorch/coreml/*.pte` for whisper-small and the
  CTC model. This target has never been exported: whatever happens is a
  result.

### Checking

- R7. Every check runs through `porting.validate` (ASR), `porting.validate_snac`
  or `porting.validate_orpheus`, with `PORT_HOST=mac` (the script sets it), so
  results are filed as `<runtime>@mac` beside the workstation's.
- R8. ASR checks use all 100 clips, except ExecuTorch on the 1B model
  (20, it is slow). Compare with PyTorch on the same clips; the report does
  this.
- R9. Core ML runs twice per model: compute units `all` (Neural Engine where
  it fits) and `cpu`, so the Neural Engine's effect on speed and accuracy is
  visible.
- R10. Orpheus: 10 sentences through llama.cpp q4_k_m on Metal, audio decoded
  by SNAC and transcribed by `facebook/mms-1b-all` (downloads ~4 GB once).
  Report tokens per second. Real time needs about 86 tokens/s: 7 codec
  tokens per 12 Hz frame.

### Acceptance

| check | pass when |
|---|---|
| MLX fp16, q8 | CER within 0.5 points of PyTorch on the same clips; identical transcripts on 95%+ |
| MLX q4 | CER within 1 point; report it either way |
| Core ML whisper encoder (`all`, `cpu`) | CER within 0.5 points; RTF recorded for both |
| Core ML CTC model (`all`, `cpu`) | CER within 0.5 points. Parity may sit below 100%: the fixed 10 s windows are zero-padded, and wav2vec2 takes no attention mask; the workstation saw 65% identical from that alone |
| whisper.cpp f16, q5_0, f16-coreml | runs on Metal; CER within 0.5 points of the workstation's whisper.cpp row (its decoding differs from transformers', see the report) |
| SNAC Core ML | "at floor" in `snac-coreml@mac.json` |
| Orpheus llama.cpp Metal | no empty outputs; judge CER recorded (workstation CPU: 4.2% on 3 sentences); tokens/s recorded |
| ExecuTorch Core ML | exports, or fails with the reason written down |
| Every run | a result JSON, or its failure recorded in section 8 of this PRD |

Anything that misses its threshold is not a reason to stop. Write down what
happened and why, fix it if the cause is in our code, and move on.

## 6. How to run it

```bash
git clone git@github.com:sapinsapin/halohalo.git && cd halohalo && git checkout bantaywika
tar xf ~/Downloads/halohalo-mac-eval.tar          # wherever the user put it
bash scripts/port_mac.sh setup                    # ~15 min
bash scripts/port_mac.sh build                    # ~45 min; Orpheus merge + GGUF is most of it
bash scripts/port_mac.sh reference                # no-op when the eval pack brought the references
bash scripts/port_mac.sh validate                 # ~1 h
bash scripts/port_mac.sh executorch               # ~30 min, may fail: see R6
bash scripts/port_mac.sh orpheus-mlx              # optional, ~20 min
bash scripts/port_mac.sh report                   # regenerates docs/porting_report.md
bash scripts/port_mac.sh pack                     # mac-results.tar for the workstation
```

Everything logs to `finetune_runs/port/mac.log`; each step skips finished work
and continues past failures, so rerunning a step is always safe.

### Stretch: Safari WebGPU

The browser target, checked in Safari 26 instead of Chrome. It needs the
ONNX web artefacts, which `build` does not make:

```bash
python3.12 -m venv venv_mac_onnx && venv_mac_onnx/bin/pip install "optimum-onnx[onnxruntime]" onnx onnxscript accelerate
venv_mac_onnx/bin/python3 -m porting.export_onnx sapinsapin/whisper-small-pld-ceb --skip npu
venv_mac_onnx/bin/python3 -m porting.export_onnx sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm --skip npu
(cd porting/web && npm install)
FINETUNE_DIR=$PWD/finetune_runs python3 porting/web/serve.py      # then open http://localhost:8765/web/ in Safari
```

Run whisper-small fp16 and the CTC model q4f16 on WebGPU. The page files each
result through the server. For Safari rows, rename them afterwards to
`webgpu@safari-…`, or change the POST payload's `device`.

## 7. Known pitfalls

- **coremltools and torch.** coremltools 9 is tested against torch 2.7.
  The workstation converted with torch 2.14 after one fix: SNAC's TorchScript
  activation broke the converter, so `export_snac` swaps in a plain one for
  fixed-shape targets. If a conversion fails in torch's frontend, retry in a
  venv with `torch==2.7.*`.
- **Whisper's Core ML half.** `run_coreml` feeds the Core ML encoder's output
  into transformers' decoder through `generate(encoder_outputs=…)`. This path
  is untested. If transformers rejects it, pass only `encoder_outputs` or
  write the forced-prefix loop the way `export_executorch.whisper_greedy`
  does.
- **First Core ML load is slow.** The Neural Engine compiles on first use.
  Load time is reported separately from RTF; do not read it as speed.
- **whisper.cpp and Core ML.** whisper.cpp finds the Core ML encoder by name:
  `<model>-encoder.mlmodelc` beside `<model>.bin`. `build` makes
  `ggml-model-f16-coreml.bin` (a symlink) and its compiled encoder, so the
  plain `f16` stays a Metal-only baseline.
- **Orpheus tokenizer.** It lists one token (`<|audio|>`, id 156939) past the
  embedding matrix. `export_orpheus.trim_tokenizer` removes it after the merge,
  or llama.cpp's converter stops.
- **Orpheus judge.** The MMS judge and SNAC run on the CPU here, which is
  fine for 10 sentences.
- **ExecuTorch.** The Core ML delegate must exist in the installed executorch
  wheel, both to export and for the Python runtime to execute the .pte. If
  the runtime lacks it, record that and skip the check.
- **Memory.** The CTC model's Core ML conversion traces a 1B model: close other
  apps on a 16 GB Mac.

## 8. Findings (fill in while running)

<!-- One line per check that failed, missed its threshold, or surprised you:
     what happened, why, and what was changed. Numbers belong in the report. -->

## 9. Done means

1. The steps in section 6 have run: `validate`, and `executorch` at least
   attempted.
2. `docs/porting_report.md` is regenerated and shows the `@mac` rows.
3. Section 8 above lists every failure or surprise.
4. Code fixes are committed with messages that say what broke and why.
5. The Mac-specific registry wording is updated where the Mac proved it
   wrong, e.g. `validate_here` for `npu-ane`, `mac-mlx` and `mac-executorch`
   in `porting/registry.py`.
6. Everything is pushed to `bantaywika`, or to a branch off it if the user
   prefers. Never push to `main` without asking.
7. The user gets `mac-results.tar`, which holds test transcripts, so copy it
   by hand, never commit it.

## 10. Guardrails for the session

- Commit code, docs and the regenerated report. Never commit
  `finetune_runs/`, `third_party/`, `venv_*`, the eval pack, audio,
  transcripts or anything under `splits/`. `.gitignore` covers most of it:
  check `git status` before each commit.
- No secrets are needed: every model is public. Do not create or copy tokens.
- Do not upload artefacts or results anywhere; the user decides on
  publishing.
- Commits end with the `Co-Authored-By` line the session's instructions
  give. Use the same git author as this branch's history
  (`git log -1 --format='%an <%ae>'`) unless the user says otherwise.
