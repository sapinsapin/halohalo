"""Orpheus-3B TTS to every LLM runtime. Phase 2 (RTX PRO 6000 VM); the
llama.cpp path also runs on the workstation's CPU as a dry run.

  python -m porting.export_orpheus sapinsapin/orpheus-3b-0.1-pretrained-char-pld-ceb --steps merge gguf
  python -m porting.export_orpheus <repo> --steps merge gguf mlx onnx openvino executorch webllm mediapipe

Our Orpheus models are LoRA adapters on unsloth/orpheus-3b-0.1-pretrained (a
Llama-3.2-3B with 28k extra audio tokens). Every runtime wants plain weights,
so `merge` comes first and everything else starts from the merged bf16
checkpoint. Each step runs in the environment it names (porting/setup_venvs.sh
llm on the VM):

  merge       peft merge_and_unload -> merged/ (bf16 safetensors)       any torch venv
  gguf        llama.cpp convert_hf_to_gguf + llama-quantize              Arm CPU, Apple Metal, AMD HIP/Vulkan
              -> gguf/model-{f16,q8_0,q4_k_m}.gguf
  mlx         mlx_lm.convert -q -> mlx/q4/ (and mlx/q8/)                 Apple silicon
  onnx        onnxruntime-genai model builder, int4 -> onnx-genai/{cpu,webgpu}/   Arm CPU, WebGPU, DirectML
  openvino    optimum-intel int4 weights -> openvino/int4/               Intel CPU/iGPU/NPU
  executorch  optimum-executorch xnnpack 8da4w -> executorch/model.pte   Android/iOS CPU
  webllm      mlc_llm convert_weight/gen_config/compile q4f16_1 -> webllm/   browsers (WebGPU)
  mediapipe   ai-edge-torch llama recipe + bundler -> mediapipe/model.task   MediaPipe LLM Inference

The SNAC decoder travels beside every one of these (porting/export_snac.py);
the app loop is: prompt ids -> LLM samples audio tokens -> 7-token frames ->
SNAC -> 24 kHz audio. porting/validate_orpheus.py runs that loop per runtime
and scores the audio with the independent MMS-1b-all judge.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"
TOOLS = Path(os.environ.get("PORT_TOOLS", "/mnt/d/halohalo/third_party"))
BASE = "unsloth/orpheus-3b-0.1-pretrained"
TOK = os.environ.get("HF_TOKEN")


def sh(*args, **kw):
    print("  $", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], check=True, **kw)


def merge(repo, root):
    """LoRA -> plain weights. bf16 on CPU fits in ~14 GB RAM; on the VM, GPU."""
    out = root / "merged"
    if (out / "config.json").exists():
        return out
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": dev})
    model = PeftModel.from_pretrained(base, repo, token=TOK).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    # the adapter repo carries the tokenizer it was trained with
    AutoTokenizer.from_pretrained(repo, token=TOK).save_pretrained(out)
    from porting.hfcompat import sanitize_tokenizer_config
    sanitize_tokenizer_config(out / "tokenizer_config.json")
    trim_tokenizer(out, model.config.vocab_size)
    return out


def trim_tokenizer(d: Path, n: int):
    """Drop tokenizer entries past the embedding matrix. Orpheus's tokenizer
    lists 156,940 tokens for 156,939 embeddings (an unused `<|audio|>` at id
    156939); llama.cpp's converter asserts every id fits and stops. Growing the
    embeddings instead would add an output row the sampler could pick."""
    tj = json.loads((d / "tokenizer.json").read_text())
    tj["added_tokens"] = [t for t in tj.get("added_tokens", []) if t["id"] < n]
    vocab = tj.get("model", {}).get("vocab")
    if isinstance(vocab, dict):
        tj["model"]["vocab"] = {k: v for k, v in vocab.items() if v < n}
    (d / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
    tc_path = d / "tokenizer_config.json"
    tc = json.loads(tc_path.read_text())
    dropped = set()
    atd = tc.get("added_tokens_decoder")
    if isinstance(atd, dict):
        dropped = {v.get("content") for k, v in atd.items() if int(k) >= n}
        tc["added_tokens_decoder"] = {k: v for k, v in atd.items() if int(k) < n}
    for key in ("additional_special_tokens", "extra_special_tokens"):
        if isinstance(tc.get(key), list):
            tc[key] = [t for t in tc[key] if t not in dropped]
    tc_path.write_text(json.dumps(tc, indent=1, ensure_ascii=False))


def gguf(merged, root):
    """The format with the widest reach: llama.cpp on Arm (NEON/KleidiAI),
    Apple (Metal), AMD (HIP, Vulkan), Nvidia, and Ollama/LM Studio."""
    lc = TOOLS / "llama.cpp"
    out = root / "gguf"
    out.mkdir(parents=True, exist_ok=True)
    f16 = out / "model-f16.gguf"
    if not f16.exists():
        sh(sys.executable, lc / "convert_hf_to_gguf.py", merged, "--outtype", "f16", "--outfile", f16)
    q = lc / "build" / "bin" / "llama-quantize"
    for t in ("Q8_0", "Q4_K_M"):
        dst = out / f"model-{t.lower()}.gguf"
        if not dst.exists():
            sh(q, f16, dst, t, stdout=subprocess.DEVNULL)
    return {p.name: round(p.stat().st_size / 2**20) for p in out.glob("*.gguf")}


def mlx(merged, root):
    from mlx_lm import convert
    info = {}
    for bits in (4, 8):
        out = root / "mlx" / f"q{bits}"
        if not out.exists():
            convert(str(merged), mlx_path=str(out), quantize=True, q_bits=bits, q_group_size=64)
        info[f"q{bits}"] = round(sum(p.stat().st_size for p in out.glob("*.safetensors")) / 2**20)
    return info


def onnx_genai(merged, root):
    """onnxruntime-genai's builder: int4 weight-only (block 32), the form the
    ORT GenAI runtime loads on CPU (Arm64 included), CUDA, DirectML and WebGPU."""
    info = {}
    for ep in ("cpu", "webgpu"):
        out = root / "onnx-genai" / ep
        if not (out / "genai_config.json").exists():
            sh(sys.executable, "-m", "onnxruntime_genai.models.builder", "-i", merged, "-o", out,
               "-p", "int4", "-e", ep, "-c", root / "onnx-genai" / "_cache")
        info[ep] = round(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**20)
    return info


def openvino(merged, root):
    out = root / "openvino" / "int4"
    if not (out / "openvino_model.xml").exists():
        sh("optimum-cli", "export", "openvino", "-m", merged, "--task", "text-generation-with-past",
           "--weight-format", "int4", "--group-size", "128", "--ratio", "1.0", out)
    return round(sum(p.stat().st_size for p in out.glob("*.bin")) / 2**20)


def executorch(merged, root):
    """optimum-executorch's Llama recipe: int8 dynamic activations x int4
    weights (8da4w), custom SDPA and KV cache ops, 8-bit embeddings."""
    out = root / "executorch"
    if not list(out.glob("*.pte")):
        sh("optimum-cli", "export", "executorch", "--model", merged, "--task", "text-generation",
           "--recipe", "xnnpack", "--use_custom_sdpa", "--use_custom_kv_cache",
           "--qlinear", "8da4w", "--qembedding", "8w", "--output_dir", out)
    return {p.name: round(p.stat().st_size / 2**20) for p in out.glob("*.pte")}


def webllm(merged, root):
    """MLC: q4f16_1 weights + a WebGPU wasm library. compile needs emscripten
    (emsdk) on PATH and the mlc-llm wasm runtime; the weights and config do not."""
    out = root / "webllm"
    if not (out / "mlc-chat-config.json").exists():
        sh("mlc_llm", "convert_weight", merged, "--quantization", "q4f16_1", "-o", out)
        # Orpheus is prompted with raw token ids, not a chat template
        sh("mlc_llm", "gen_config", merged, "--quantization", "q4f16_1",
           "--conv-template", "LM", "-o", out)
    wasm = out / "orpheus-q4f16_1-webgpu.wasm"
    if not wasm.exists() and shutil.which("emcc"):
        sh("mlc_llm", "compile", out / "mlc-chat-config.json", "--device", "webgpu", "-o", wasm)
    return {"weights_mb": round(sum(p.stat().st_size for p in out.glob("params_shard_*")) / 2**20),
            "wasm": wasm.exists()}


def mediapipe(merged, root):
    """ai-edge-torch's Llama 3.2 3B recipe -> LiteRT, int8 dynamic, then the
    MediaPipe bundler. Orpheus keeps Llama's shapes, but its vocabulary is
    156,940 tokens, not 128,256, so the recipe's config is widened to match."""
    import ai_edge_torch.generative.examples.llama.llama as llama
    from ai_edge_torch.generative.utilities import converter
    from transformers import AutoConfig
    out = root / "mediapipe"
    out.mkdir(parents=True, exist_ok=True)
    vocab = AutoConfig.from_pretrained(merged).vocab_size
    cfg_fn = llama.get_3b_model_config
    llama.get_3b_model_config = lambda **kw: (lambda c: (setattr(c, "vocab_size", vocab), c)[1])(cfg_fn(**kw))
    model = llama.build_3b_model(str(merged))
    converter.convert_to_tflite(model, output_path=str(out), output_name_prefix="orpheus",
                                prefill_seq_len=256, kv_cache_max_len=2048, quantize="dynamic_int8")
    from mediapipe.tasks.python.genai import bundler
    tfl = next(out.glob("orpheus*.tflite"))
    bundler.create_bundle(bundler.BundleConfig(
        tflite_model=str(tfl), tokenizer_model=str(merged / "tokenizer.json"),
        start_token="<|begin_of_text|>", stop_tokens=["<custom_token_2>"],
        output_filename=str(out / "orpheus.task")))
    return round((out / "orpheus.task").stat().st_size / 2**20)


STEPS = {"gguf": gguf, "mlx": mlx, "onnx": onnx_genai, "openvino": openvino,
         "executorch": executorch, "webllm": webllm, "mediapipe": mediapipe}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--steps", nargs="+", default=["merge", "gguf"],
                    choices=["merge", *STEPS])
    args = ap.parse_args()
    root = ART / args.repo.split("/")[-1]
    root.mkdir(parents=True, exist_ok=True)
    log_f = root / "build_orpheus.json"
    log = json.loads(log_f.read_text()) if log_f.exists() else {}
    merged = root / "merged"
    for s in args.steps:
        t0 = time.perf_counter()
        try:
            if s == "merge":
                merge(args.repo, root)
                res = "ok"
            else:
                res = STEPS[s](merged, root)
            log[s] = {"result": res, "seconds": round(time.perf_counter() - t0)}
        except Exception as e:           # one toolchain failing must not stop the rest
            log[s] = {"error": f"{type(e).__name__}: {e}"[:500]}
        print(f"{s}: {log[s]}", flush=True)
        log_f.write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
