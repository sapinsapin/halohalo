# Porting pipeline: halohalo models on Arm, NPUs, Apple, the browser and AMD

The published models are PyTorch checkpoints that run on an Nvidia GPU. This
pipeline turns each one into the artefact each other kind of hardware loads,
and checks every artefact against the original on the same held-out clips
before anyone ships it. Results: [porting_report.md](porting_report.md).

Code: [`porting/`](../porting). Entry point:

```bash
python3 -m porting plan -v                    # what applies where, and in which phase
bash scripts/port_phase1.sh                   # workstation: build + validate + report (CPU only)
bash scripts/port_phase2.sh                   # RTX PRO 6000 VM: Orpheus and the >1B models
```

## 1. Targets and the toolchain that reaches each

| target | hardware | toolchain | artefact | precision |
|---|---|---|---|---|
| arm-cpu-onnx | Arm64 Linux/Android, Graviton, Raspberry Pi, Apple CPU | optimum-onnx → ONNX Runtime quantisation | ONNX | int8 dynamic, fp32 |
| arm-cpu-ggml | any CPU (NEON/KleidiAI on Arm); Metal, Vulkan, HIP builds | whisper.cpp / llama.cpp converters | ggml / GGUF | f16, q8_0, q5_0, q4_K_M |
| arm-mobile-executorch | Android/iOS CPU | ExecuTorch, XNNPACK backend | .pte | fp32, int8, 8da4w |
| npu-qnn | Qualcomm Hexagon (Snapdragon phones, Copilot+ PCs) | static QDQ int8 at fixed shapes → ORT QNN EP | ONNX (QDQ) | int8 |
| npu-ryzenai | AMD Ryzen AI (XDNA) | the same QDQ graph → Vitis AI EP (AMD Quark for tuned int8) | ONNX (QDQ) | int8 |
| npu-openvino | Intel Core Ultra NPU, iGPU, CPU | optimum-intel (stateful decoder) + NNCF | OpenVINO IR | fp16, int8, int4 |
| npu-ane | Apple Neural Engine | coremltools | .mlpackage | fp16 |
| mac-mlx | Apple silicon GPU | mlx-whisper / mlx-lm | MLX safetensors | fp16, 8-bit, 4-bit |
| mac-executorch | Apple silicon via ExecuTorch Core ML/MPS | ExecuTorch Core ML partitioner | .pte | fp16 |
| web-webgpu | browsers (WebGPU; wasm fallback) | the ONNX export in the Transformers.js layout | ONNX + `onnx/*_{fp16,quantized}.onnx` | fp16 (WebGPU), q8 (wasm) |
| web-mediapipe | browsers, Android, iOS via MediaPipe LLM Inference | ai-edge-torch → LiteRT → .task bundle | .task | int8 |
| web-webllm | browsers via WebLLM | MLC convert/compile | MLC weights + .wasm | q4f16_1 |
| amd-rocm | AMD Instinct/Radeon GPUs | PyTorch ROCm as-is; ORT MIGraphX EP; llama.cpp HIP | checkpoint / ONNX / GGUF | fp16, int8 |

One export often serves several targets. The ONNX export feeds Arm CPUs, the
browser, both NPU families and AMD GPUs. The GGUF file feeds Arm, Apple, AMD
and Nvidia through llama.cpp.

## 2. Which model goes where

A recipe is written per architecture, not per checkpoint: every
`whisper-small-pld-<lang>` ports the same way. `porting/registry.py` holds the
table, including the reasons a pair does not apply:

| family | ports to | does not port to, and why |
|---|---|---|
| Whisper (small, large-v3) | all but MediaPipe and WebLLM | MediaPipe and WebLLM run decoder-only LLMs; Whisper is encoder-decoder |
| wav2vec2 CTC (omniASR 1B/7B) | ONNX, ExecuTorch, all NPUs, Core ML, web, AMD | no maintained ggml or MLX wav2vec2 runtime; Core ML and ExecuTorch reach the same Macs |
| Orpheus-3B TTS (Llama) | every target but the ANE | a 3B autoregressive decode does not map onto the ANE usefully; MLX is the Mac path |
| SNAC 24 kHz decoder | ONNX, web, ExecuTorch, Core ML | ships beside every Orpheus port |
| fastText language ID | the browser (WebAssembly) | nothing to convert elsewhere: the C++ library builds natively on Arm |

NPUs get the part of a model that has a fixed shape and no loop. For Whisper,
that is the encoder at its 30-second window, which is where nearly all the
compute is. The decoder stays on the CPU or GPU, the split WhisperKit and
whisper.cpp's Core ML mode use. The CTC model is one forward pass, so it goes
whole, cut into fixed 10-second windows.

## 3. Phases

The phase is derived from the model's size, not assigned by hand
(`registry.phase_of`):

- **Phase 1, the workstation** (RTX 3070 8 GB, 31 GB RAM). Everything whose
  conversion fits in RAM and whose check runs on CPU or in ~6 GB of VRAM:
  whisper-small ×10, whisper-large-v3 ×9 and the omniASR 1B CTC models ×4
  (each except NPU calibration above 500M parameters), SNAC and the language
  ID. 224 of 367 ports.
- **Phase 2, the RTX PRO 6000 VM** (96 GB GPU, 214 GB RAM). The nine
  Orpheus-3B adapters on every LLM runtime (merge, int4 calibration and an
  end-to-end TTS check need the big card), the omniASR 7B model, and static
  int8 NPU calibration of every model over 500M parameters (whisper-large-v3
  encoders, the omniASR CTC models). 143 ports.

Phase 1 runs entirely on CPU with CUDA hidden and `nice`, so it runs beside a
training queue on the local GPU without taking it.

## 4. How a port is checked

Every artefact transcribes the same **evalpack**: the first 100 clips of the
frozen speaker- and prompt-disjoint test split per language, saved once as a
plain `.npz` so no toolchain venv needs the dataset. Calibration for static
int8 uses 64 clips from the *train* split, so calibration never sees a test
clip. Two numbers per artefact:

- **Accuracy**: CER and WER against the human transcript, normalised the way
  the `-norm` models are scored. This is what the port is worth to a user.
- **Parity**: CER of the port's transcript against the PyTorch fp32
  transcript of the same model. This isolates what porting lost: 0 means the
  port behaves exactly like the original.

Plus size on disk and real-time factor at a fixed 4 threads. The RTF compares
variants on one CPU; it is not a prediction for an Arm phone.

SNAC's decoder adds random noise inside, so two PyTorch runs already differ.
Its port is scored by log-mel distance from PyTorch next to that
PyTorch-vs-PyTorch floor. The language ID is scored on identical top labels.

**What this machine cannot run, and the stand-in used**:

| target | checked here as | checked for real on |
|---|---|---|
| Arm CPU | the same ONNX graph / C++ runtime on x86 | GitHub `ubuntu-24.04-arm` runners, a Raspberry Pi 5 |
| Qualcomm / AMD NPU | the QDQ graph on the ORT CPU EP (same int8 arithmetic) | a Snapdragon X laptop (QNN EP), a Ryzen AI laptop (Vitis AI EP) |
| Intel NPU | the OpenVINO CPU plugin on the same IR | a Core Ultra laptop, `WhisperPipeline(path, "NPU")` |
| Apple (MLX) | MLX's Linux CPU backend, same weights and graph | any M-series Mac |
| Apple (Core ML, ExecuTorch Core ML) | not checked; built only | a Mac (`macos-14` runner): `coremlc compile`, then predict |
| Browser WebGPU | Node + Transformers.js / onnxruntime-node (CPU); the fp16 graph on ORT CPU | Chrome with WebGPU |
| AMD GPU | the fp32 ONNX and GGUF results stand in | an MI-series or Radeon card with ROCm |

## 5. Recipes worth knowing

- **ONNX, Whisper.** optimum-onnx's `automatic-speech-recognition-with-past`
  export, keeping only the encoder and the *merged* decoder (one graph with
  and without KV cache), which is what Transformers.js and onnxruntime-web
  load. Dynamic int8 needs `EnableSubgraph`, or the merged decoder's branches
  are skipped and the "int8" file stays fp32-sized.
- **NPU QDQ.** Fix every dynamic dimension, run `quant_pre_process`, then
  static QDQ with per-channel int8 weights and uint8 activations calibrated by
  min/max. On ORT 1.30, leave `CalibMaxIntermediateOutputs` unset: it drops the
  batches it should fold in and fails with "No data is collected".
- **whisper.cpp.** HF checkpoint → OpenAI layout (`porting/whisper_openai.py`)
  → `convert-pt-to-ggml.py` → `whisper-quantize`. Save the OpenAI checkpoint in
  fp16: the converter marks 2-D tensors f16 without converting them.
- **MLX.** The same OpenAI layout, conv weights transposed to MLX's
  `(out, kernel, in)`, quantised with `nn.quantize` (group 64).
- **OpenVINO.** optimum-intel's exporter rather than converting the ONNX graphs,
  because it writes the decoder as a stateful model, which OpenVINO GenAI and
  the NPU plugin need.
- **CTC in the browser.** onnxruntime-web plus a 60-line decoder
  (`porting/web/ctc.mjs`): the vocabulary is 32 characters, so no tokenizer
  library is needed.
- **Large models in the browser: 4-bit.** The 963M CTC model's fp16 graph
  (1.8 GB of weights in a side file) never finished loading in Chrome:
  onnxruntime-web stages weights through a 4 GB WebAssembly heap. With 4-bit
  block-quantised MatMul weights and fp16 elsewhere (`MatMulNBits`, block 32:
  Transformers.js's `q4f16`) it is one 543 MB file, loads in 15 s and runs at
  RTF 0.29 on the workstation's RTX 3070.
- **NPU calibration has a memory ceiling.** Static int8 calibration of the
  963M CTC graph at its 10 s window peaked at 25 GB of RAM. An out-of-memory
  kill in WSL fails systemd's `init.scope` and takes every session down, so
  models over 500M parameters calibrate on the VM (Phase 2). The pipeline also
  runs graph preparation in a child process, and `scripts/mem_guard.sh` stops
  the largest export before the kernel's OOM killer would.
- **Config compatibility.** The fine-tunes were saved by transformers 5; the
  toolchains pin 4.x. `porting/hfcompat.py` renames `extra_special_tokens` to
  `additional_special_tokens` and unfolds the feature extractor from
  `processor_config.json`.

## 6. Running it

```bash
bash porting/setup_venvs.sh all          # one venv per toolchain, on D:, CPU torch
python3 -m porting plan -v
MODELS=demo  bash scripts/port_phase1.sh # one model per family, full evalpack
MODELS=phase bash scripts/port_phase1.sh # every Phase 1 model; siblings get a 20-clip check
```

On the workstation, launch long runs from Windows so a `wsl.exe` client lives
as long as the job; a job detached inside WSL dies when the last client exits:

```powershell
Start-Process -WindowStyle Hidden wsl.exe -ArgumentList 'bash -c "cd /mnt/d/halohalo && bash scripts/port_phase1.sh >> finetune_runs/port/phase1.log 2>&1"'
```

**The browser check** runs the same `onnx-web` folders in a real browser. It
files its result with the other runtimes' results, so it shows up in the report:

```bash
python3 porting/web/serve.py        # serves only the porting folders, never the repo root
```

Open `http://localhost:8765/web/`, choose WebGPU or WebAssembly and the
precision, and press Run. On the workstation's RTX 3070, Chrome ran
whisper-small fp16 at RTF 0.43 with transcripts identical to PyTorch on all
100 clips.

**Supervision.** WSL on the workstation has been crashing on CPU machine-check
exceptions. Run long jobs under `scripts/wsl_supervise.ps1`, which starts a
resumable job again after each crash:

```powershell
Start-Process -WindowStyle Hidden powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\wsl_supervise.ps1','-Name','port','-Command','cd /mnt/d/halohalo && bash scripts/port_phase1.sh >> finetune_runs/port/phase1.log 2>&1'
```

**Phase 2** on the VM: [`scripts/port_phase2.sh`](../scripts/port_phase2.sh)
chains every Orpheus toolchain and then the ASR ports that waited for the big
card, in one tmux session, so the GPU never waits on a person. A rough budget
at the preemptible rate of about $1.08 an hour:

| work | estimate |
|---|---:|
| Orpheus, per language: merge, GGUF, MLX, ORT GenAI, OpenVINO, ExecuTorch, WebLLM, MediaPipe | ~1.5 h |
| Orpheus, per language: end-to-end checks on 5 runtimes, 10 sentences each | ~1 h |
| nine languages | ~22 h, ~$24 |
| whisper-large-v3 NPU calibration ×9, omniASR 7B exports | ~5 h, ~$5 |

`DRY=1 LANGS=ceb N=3` runs the llama.cpp path on the workstation CPU first, to
catch mistakes before paying for the card.

## 7. Adding a model or a target

- **A model of a known family:** add a `Model(...)` row to `porting/registry.py`.
  The plan, phase and recipes follow.
- **A new family:** add a `Family`, its `APPLICABILITY` rows (a "no" needs a
  reason), and its `RECIPES` entry in `porting/__main__.py`.
- **A new target:** add a `Target` with how it is validated, then the
  applicability rows and a recipe.
