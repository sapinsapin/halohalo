"""What can be ported where, and in which phase.

Three tables, declarative on purpose so the plan is reviewable without running
anything:

- FAMILIES: model architectures, because a recipe is written per architecture,
  not per checkpoint — every whisper-small-pld-<lang> ports the same way.
- MODELS: the published checkpoints, with their family and size.
- TARGETS: hardware classes, each with the toolchain that reaches it, the
  artefact format it consumes, and whether this workstation can validate it.

APPLICABILITY says, per (family, target), whether a recipe exists and why not
when it does not. A "no" with a reason is a result, not a gap: MediaPipe and
WebLLM are LLM runtimes and have nothing to offer a CTC acoustic model.

Phase is derived, not assigned (see phase_of): Phase 1 is whatever the
workstation (RTX 3070, ~6 GB free VRAM, 31 GB RAM) can convert *and* validate;
Phase 2 is the rest, for the RTX PRO 6000 VM.
"""

from dataclasses import dataclass, field

# ------------------------------------------------------------------ hardware
LOCAL = {"vram_gib": 6.0, "ram_gib": 27.0, "name": "RTX 3070 workstation"}
# Largest model whose NPU (static QDQ int8) calibration fits the workstation.
# Measured: whisper-small's encoder (88M) fits; the 963M CTC model peaked at
# 25 GB with ORT's calibrator and OOM-risked WSL, so it waits for the VM.
NPU_CALIB_MAX_M = 500
CLOUD = {"vram_gib": 95.0, "ram_gib": 214.0, "name": "RTX PRO 6000 VM"}


@dataclass(frozen=True)
class Family:
    name: str
    kind: str              # asr-seq2seq | asr-ctc | llm | codec | text-classifier
    notes: str


FAMILIES = {
    "whisper": Family("whisper", "asr-seq2seq",
                      "Encoder-decoder with autoregressive decoding and a KV cache. "
                      "Ports as two graphs; NPUs take the encoder, the decoder stays on CPU/GPU."),
    "wav2vec2-ctc": Family("wav2vec2-ctc", "asr-ctc",
                           "Encoder plus a linear CTC head: one forward pass, no decoding "
                           "loop. The most NPU-friendly model we have."),
    "orpheus": Family("orpheus", "llm",
                      "Llama-3.2-3B emitting SNAC codec tokens; a LoRA adapter must be "
                      "merged first. Every LLM runtime applies; the SNAC decoder must ship beside it."),
    "snac": Family("snac", "codec",
                   "SNAC 24 kHz decoder, 19.8M parameters, convolutional. Turns Orpheus's "
                   "tokens into audio on every target."),
    "fasttext": Family("fasttext", "text-classifier",
                       "fastText language identifier. Not a neural graph to lower: the "
                       "C++ library builds for Arm and compiles to WebAssembly as-is."),
}


@dataclass(frozen=True)
class Model:
    repo: str
    family: str
    params_m: float                 # millions
    note: str = ""

    @property
    def fp16_gib(self) -> float:
        return self.params_m * 1e6 * 2 / 2**30


ORG = "sapinsapin"
LANGS9 = ["bcl", "ceb", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]
MODELS = (
    [Model(f"{ORG}/whisper-small-pld-{l}", "whisper", 242)
     for l in ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]]
    + [Model(f"{ORG}/whisper-large-v3-pld-{l}-norm", "whisper", 1543) for l in LANGS9]
    + [Model(f"{ORG}/omniASR_W2V_1B_SSL-ctc-char-pld_{l}{s}", "wav2vec2-ctc", 963)
       for l in ["ceb", "pam"] for s in ("", "-norm")]
    + [Model(f"{ORG}/omniASR_W2V_7B_SSL-ctc-char-pld_ceb", "wav2vec2-ctc", 6500)]
    + [Model(f"{ORG}/orpheus-3b-0.1-pretrained-char-pld-{l}", "orpheus", 3300,
             "LoRA on unsloth/orpheus-3b-0.1-pretrained; merged before export")
       for l in ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pam", "tsg", "war"]]
    + [Model("hubertsiuzdak/snac_24khz", "snac", 19.8, "upstream codec, not ours")]
    + [Model(f"{ORG}/halo-lid", "fasttext", 2.2)]
)


@dataclass(frozen=True)
class Target:
    id: str
    hardware: str
    toolchain: str
    artefact: str
    precision: str
    validate_here: str          # "yes" | "proxy: ..." | "no: ..."
    examples: str = ""


TARGETS = [
    Target("arm-cpu-onnx", "Arm CPU (Linux/Android arm64, Graviton, Raspberry Pi, Apple silicon CPU)",
           "optimum-onnx export -> onnxruntime.quantization", "ONNX", "int8 (dynamic), fp32",
           "proxy: same graph on x86 ONNX Runtime; Arm run via GitHub arm64 runners",
           "ONNX Runtime CPU EP; NEON/i8mm kernels on Arm"),
    Target("arm-cpu-ggml", "Arm CPU (and any CPU; AMD/Nvidia via Vulkan/HIP/CUDA builds)",
           "whisper.cpp / llama.cpp converters + quantize", "GGUF", "q8_0, q5_1, q4_K",
           "proxy: x86 build of the same C++ runtime"),
    Target("arm-mobile-executorch", "Android/iOS Arm CPUs",
           "optimum-executorch / torch.export -> XNNPACK", ".pte", "fp32, int8",
           "proxy: ExecuTorch runtime on x86 (XNNPACK has x86 kernels)"),
    Target("npu-qnn", "Qualcomm Hexagon NPU (Snapdragon phones, Copilot+ PCs)",
           "ONNX QDQ static int8, fixed shapes -> ONNX Runtime QNN EP", "ONNX (QDQ)",
           "int8 activations and weights, calibrated",
           "proxy: QDQ graph numerics on CPU; the NPU needs a Snapdragon device"),
    Target("npu-ane", "Apple Neural Engine",
           "coremltools", ".mlpackage", "fp16",
           "no: converts on Linux, predicts only on macOS"),
    Target("npu-openvino", "Intel NPU (Core Ultra), also Intel CPU/iGPU",
           "optimum-intel export (stateful decoder) + NNCF int8 weights", "OpenVINO IR", "fp16, int8",
           "proxy: OpenVINO CPU plugin on this AMD CPU"),
    Target("npu-ryzenai", "AMD Ryzen AI NPU (XDNA)",
           "ONNX QDQ static int8 (AMD Quark for the tuned path) -> Vitis AI EP", "ONNX (QDQ)",
           "int8 (A8W8)", "proxy: QDQ numerics on CPU; the NPU needs a Ryzen AI laptop"),
    Target("mac-mlx", "Apple silicon (M1-M4) GPU via Metal",
           "mlx-whisper convert / mlx-lm convert", "MLX safetensors", "fp16, 4-bit",
           "proxy: MLX Linux CPU backend loads and runs the weights"),
    Target("mac-executorch", "Apple silicon via ExecuTorch Core ML / MPS backends",
           "optimum-executorch with coreml/mps recipe", ".pte", "fp16",
           "no: the Core ML and MPS partitioners export only on macOS"),
    Target("web-webgpu", "Browsers with WebGPU (Chrome, Edge, Safari 26), wasm fallback",
           "optimum-onnx export in the Transformers.js layout", "ONNX + onnx/*_{fp16,quantized}.onnx",
           "fp16 (WebGPU), q8 (wasm)", "proxy: Transformers.js on Node (onnxruntime-node)"),
    Target("web-mediapipe", "Browsers/Android/iOS via MediaPipe LLM Inference API",
           "ai-edge-torch generative converter -> .task bundle", ".task (LiteRT)",
           "int8 / int4", "no: needs the bundle and a MediaPipe runtime"),
    Target("web-webllm", "Browsers with WebGPU via WebLLM",
           "mlc_llm convert_weight / gen_config / compile --device webgpu", "MLC weights + .wasm",
           "q4f16_1", "no: needs emscripten and a WebGPU browser"),
    Target("amd-rocm", "AMD Instinct/Radeon GPUs",
           "PyTorch ROCm (checkpoint as-is); ONNX Runtime MIGraphX EP; llama.cpp HIP", "HF / ONNX / GGUF",
           "fp16, int8", "no: needs an AMD GPU; the ONNX and GGUF artefacts are shared with other targets"),
]
TARGET_BY_ID = {t.id: t for t in TARGETS}

Y, N = "yes", "no"
# (family, target) -> (applies, how or why not)
APPLICABILITY = {
    ("whisper", "arm-cpu-onnx"): (Y, "encoder + merged decoder with KV cache"),
    ("whisper", "arm-cpu-ggml"): (Y, "whisper.cpp; the most-used Arm path for Whisper"),
    ("whisper", "arm-mobile-executorch"): (Y, "optimum-executorch whisper recipe"),
    ("whisper", "npu-qnn"): (Y, "encoder on the NPU at a fixed 30 s window; decoder on CPU"),
    ("whisper", "npu-ane"): (Y, "encoder in Core ML on the ANE (the WhisperKit split)"),
    ("whisper", "npu-openvino"): (Y, "encoder + decoder IR; optimum-intel supports Whisper"),
    ("whisper", "npu-ryzenai"): (Y, "encoder on the NPU, decoder on CPU"),
    ("whisper", "mac-mlx"): (Y, "mlx-whisper"),
    ("whisper", "mac-executorch"): (Y, "optimum-executorch coreml recipe"),
    ("whisper", "web-webgpu"): (Y, "Transformers.js WhisperForConditionalGeneration"),
    ("whisper", "web-mediapipe"): (N, "MediaPipe has no speech-recognition task; its LLM API takes decoder-only text models"),
    ("whisper", "web-webllm"): (N, "WebLLM runs decoder-only LLMs, not encoder-decoder ASR"),
    ("whisper", "amd-rocm"): (Y, "checkpoint as-is on PyTorch ROCm; ONNX on MIGraphX; GGUF on HIP"),

    ("wav2vec2-ctc", "arm-cpu-onnx"): (Y, "single graph, dynamic length"),
    ("wav2vec2-ctc", "arm-cpu-ggml"): (N, "no maintained ggml runtime for wav2vec2; use ONNX or ExecuTorch"),
    ("wav2vec2-ctc", "arm-mobile-executorch"): (Y, "torch.export of encoder + head -> XNNPACK"),
    ("wav2vec2-ctc", "npu-qnn"): (Y, "the best NPU fit: fixed window, no decoding loop"),
    ("wav2vec2-ctc", "npu-ane"): (Y, "single graph in Core ML at a fixed window"),
    ("wav2vec2-ctc", "npu-openvino"): (Y, "single graph IR"),
    ("wav2vec2-ctc", "npu-ryzenai"): (Y, "fixed-window QDQ graph"),
    ("wav2vec2-ctc", "mac-mlx"): (N, "no maintained MLX wav2vec2; Core ML or ExecuTorch reach the same Macs"),
    ("wav2vec2-ctc", "mac-executorch"): (Y, "torch.export -> Core ML partitioner"),
    ("wav2vec2-ctc", "web-webgpu"): (Y, "onnxruntime-web + a 60-line JS CTC decoder (porting/web/ctc.mjs)"),
    ("wav2vec2-ctc", "web-mediapipe"): (N, "LLM-only API"),
    ("wav2vec2-ctc", "web-webllm"): (N, "LLM-only runtime"),
    ("wav2vec2-ctc", "amd-rocm"): (Y, "checkpoint as-is; ONNX on MIGraphX"),

    ("orpheus", "arm-cpu-onnx"): (Y, "onnxruntime-genai int4 model builder"),
    ("orpheus", "arm-cpu-ggml"): (Y, "llama.cpp GGUF q4_K_M; SNAC decoder in ONNX beside it"),
    ("orpheus", "arm-mobile-executorch"): (Y, "ExecuTorch Llama export, XNNPACK int4 (8da4w)"),
    ("orpheus", "npu-qnn"): (Y, "ExecuTorch QNN Llama path; needs the QNN SDK"),
    ("orpheus", "npu-ane"): (N, "3B autoregressive decode does not map to the ANE usefully; use MLX on Macs"),
    ("orpheus", "npu-openvino"): (Y, "optimum-intel OpenVINO int4"),
    ("orpheus", "npu-ryzenai"): (Y, "Ryzen AI hybrid LLM flow (ONNX GenAI, AWQ int4)"),
    ("orpheus", "mac-mlx"): (Y, "mlx-lm convert, 4-bit"),
    ("orpheus", "mac-executorch"): (Y, "ExecuTorch Llama with the MPS/Core ML backend"),
    ("orpheus", "web-webgpu"): (Y, "onnxruntime-web with the int4 ONNX; or WebLLM below"),
    ("orpheus", "web-mediapipe"): (Y, "ai-edge-torch Llama recipe -> .task; LLM Inference API"),
    ("orpheus", "web-webllm"): (Y, "MLC compile of a Llama-3.2 architecture, q4f16_1"),
    ("orpheus", "amd-rocm"): (Y, "vLLM or llama.cpp HIP"),

    ("snac", "arm-cpu-onnx"): (Y, "decoder graph; ships with every Orpheus target"),
    ("snac", "web-webgpu"): (Y, "same ONNX in onnxruntime-web"),
    ("snac", "npu-ane"): (Y, "Core ML decoder"),
    ("snac", "arm-mobile-executorch"): (Y, "torch.export -> XNNPACK"),

    ("fasttext", "arm-cpu-onnx"): (N, "not needed: the fastText C++ library builds natively on Arm"),
    ("fasttext", "web-webgpu"): (Y, "fasttext.wasm (CPU in the browser; no GPU needed at 8.5 MB)"),
}


def phase_of(model: Model, target: Target) -> tuple[int, str]:
    """Phase 1 if the workstation can convert *and* check the result; else 2.

    Conversion peak RAM is taken as ~3x the fp32 weights (export traces, keeps
    a copy, and writes the graph); validation needs the fp16 weights plus
    activations in ~6 GB of VRAM, or runs on CPU for CPU targets.
    """
    fp32_gib = model.params_m * 1e6 * 4 / 2**30
    if fp32_gib * 3 > LOCAL["ram_gib"]:
        return 2, f"conversion needs ~{fp32_gib * 3:.0f} GB RAM (have {LOCAL['ram_gib']:.0f})"
    if model.family == "orpheus":
        return 2, "LoRA merge + int4 calibration + end-to-end TTS check need the big card"
    if target.id in ("web-mediapipe", "web-webllm") and model.family == "orpheus":
        return 2, "LLM compile toolchains"
    if model.fp16_gib > LOCAL["vram_gib"] - 1.5 and target.id not in (
            "arm-cpu-onnx", "arm-cpu-ggml", "arm-mobile-executorch", "npu-openvino"):
        return 2, f"validation needs {model.fp16_gib:.1f} GB fp16 + activations on GPU"
    if model.params_m > NPU_CALIB_MAX_M and target.id in ("npu-qnn", "npu-ryzenai"):
        return 2, (f"static int8 calibration above {NPU_CALIB_MAX_M}M params does not fit the "
                   "workstation (the 963M CTC graph peaked at 25 GB RAM, measured 2026-09-26)")
    return 1, "fits the workstation"


def plan(models=MODELS, targets=TARGETS):
    """Every (model, target) pair that applies, with its phase and reason."""
    rows = []
    for m in models:
        for t in targets:
            applies, how = APPLICABILITY.get((m.family, t.id), (N, "not applicable"))
            if applies != Y:
                continue
            ph, why = phase_of(m, t)
            rows.append({"model": m.repo, "family": m.family, "target": t.id,
                         "phase": ph, "why": why, "how": how,
                         "validate_here": t.validate_here})
    return rows
