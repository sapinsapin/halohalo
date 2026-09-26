"""ExecuTorch: PyTorch's on-device runtime for Android and iOS. Runs in venv_port_et.

  python -m porting.export_executorch sapinsapin/whisper-small-pld-ceb
  python -m porting.export_executorch sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm
  python -m porting.export_executorch <repo> --backend coreml    # needs macOS

XNNPACK is the CPU backend on both platforms (NEON on Arm). Variants:

  executorch/xnnpack/        fp32, the parity baseline
  executorch/xnnpack-int8/   CTC only: int8 weights and dynamically quantised
                             activations (PT2E + XNNPACKQuantizer), ~4x smaller

Whisper goes through optimum-executorch, which exports the encoder and a
decoder with a static KV cache as methods of one .pte. The CTC model is a
single torch.export at a fixed 10 s window (dynamic length through wav2vec2's
positional convolution is fragile; mobile apps stream in fixed windows anyway).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"
CTC_WINDOW_S = 10


# ------------------------------------------------------------------ CTC
def ctc_module(repo):
    import torch
    from transformers import Wav2Vec2ForCTC
    model = Wav2Vec2ForCTC.from_pretrained(repo, token=os.environ.get("HF_TOKEN"),
                                           attn_implementation="eager").eval()

    class CTC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, input_values):
            return self.m(input_values).logits

    return CTC().eval()


def ctc_export(repo, out: Path, quantize: bool, backend: str) -> dict:
    import torch
    from executorch.exir import to_edge_transform_and_lower
    mod = ctc_module(repo)
    x = (torch.zeros(1, CTC_WINDOW_S * 16000),)
    if quantize:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config)
        try:
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        except ImportError:
            from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
        q = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))
        gm = torch.export.export(mod, x).module()
        gm = prepare_pt2e(gm, q)
        gm(*x)                      # dynamic activations: one pass initialises observers
        mod = convert_pt2e(gm)
    ep = torch.export.export(mod, x)
    if backend == "coreml":
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner
        part = [CoreMLPartitioner()]
    else:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        part = [XnnpackPartitioner()]
    prog = to_edge_transform_and_lower(ep, partitioner=part).to_executorch()
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.pte").write_bytes(prog.buffer)
    from porting.hfcompat import config_dir
    for f in ("vocab.json", "preprocessor_config.json"):
        (out / f).write_text((config_dir(repo) / f).read_text())
    return {"model.pte": round(len(prog.buffer) / 2**20, 1), "window_s": CTC_WINDOW_S}


# ------------------------------------------------------------------ Whisper
def whisper_export(repo, out: Path, backend: str) -> dict:
    """optimum-executorch's recipe: encoder + decoder with a static KV cache."""
    import shutil
    from optimum.exporters.executorch import main_export
    from porting.hfcompat import config_dir
    main_export(model_name_or_path=repo, task="automatic-speech-recognition",
                recipe=backend, output_dir=str(out), token=os.environ.get("HF_TOKEN"))
    for f in config_dir(repo).iterdir():
        if not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    return {p.name: round(p.stat().st_size / 2**20, 1) for p in out.glob("*.pte")}


# ------------------------------------------------------------------ runtime
def run_program(d: Path, audio, family: str, threads: int, on_loaded=lambda: None):
    """Transcribe with the ExecuTorch Python runtime (same kernels as on device)."""
    import torch
    torch.set_num_threads(threads)
    if family == "wav2vec2-ctc":
        from executorch.runtime import Runtime
        from porting import frontends
        method = Runtime.get().load_program(str(d / "model.pte")).load_method("forward")
        cfg = frontends.ctc_config(d)
        run = lambda x: method.execute([torch.from_numpy(x)])[0].numpy()
        on_loaded()
        return [frontends.ctc_transcribe(run, a, cfg, CTC_WINDOW_S) for a in audio]
    from optimum.executorch import ExecuTorchModelForSpeechSeq2Seq
    from transformers import WhisperProcessor
    model = ExecuTorchModelForSpeechSeq2Seq.from_pretrained(str(d))
    proc = WhisperProcessor.from_pretrained(str(d))
    on_loaded()
    # the exported decoder has a static KV cache; stay inside it
    limit = min(225, int(getattr(model, "max_cache_size", 448)) - 5)
    return [proc.tokenizer.decode(whisper_greedy(model.forward, d, proc(
        np.asarray(a, np.float32), sampling_rate=16000, return_tensors="pt").input_features, limit),
        skip_special_tokens=True).strip() for a in audio]


def whisper_greedy(forward, model_dir: Path, feats, max_new=225):
    """The decode loop an app runs around an exported Whisper: force the
    prefix the fine-tune was trained with (start, language, task,
    no-timestamps), then greedy with the generation config's suppressed
    tokens — what transformers' generate does. optimum-executorch's own
    transcribe() starts from the start token alone and suppresses nothing,
    which lets the model pick its own language slot.

    forward(features, decoder_ids (1,1), cache_position (1,), encoder_out|None)
    -> (logits, encoder_out)."""
    import torch
    gc = json.loads((model_dir / "generation_config.json").read_text())
    lang = (gc.get("language") or "tagalog").strip("<|>")
    lang_tok = next((k for k in gc["lang_to_id"] if k.strip("<|>") in (lang, {"tagalog": "tl", "english": "en"}.get(lang))),
                    "<|tl|>")
    prefix = [gc["decoder_start_token_id"], gc["lang_to_id"][lang_tok], gc["task_to_id"]["transcribe"],
              gc["no_timestamps_token_id"]]
    suppress = torch.tensor(gc.get("suppress_tokens") or [], dtype=torch.long)
    begin = torch.tensor(gc.get("begin_suppress_tokens") or [], dtype=torch.long)
    eos = gc.get("eos_token_id", 50257)
    enc, logits = None, None
    for i, t in enumerate(prefix):
        logits, enc = forward(feats, torch.tensor([[t]]), torch.tensor([i]), enc)
    out, pos = [], len(prefix)
    for step in range(max_new):
        lg = logits[0, -1].clone()
        lg[suppress] = -float("inf")
        if step == 0:
            lg[begin] = -float("inf")
        nxt = int(lg.argmax())
        if nxt == eos:
            break
        out.append(nxt)
        logits, enc = forward(feats, torch.tensor([[nxt]]), torch.tensor([pos]), enc)
        pos += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--backend", default="xnnpack", choices=["xnnpack", "coreml"])
    args = ap.parse_args()
    name = args.repo.split("/")[-1]
    root = ART / name / "executorch"
    t0 = time.perf_counter()
    info = {}
    if "whisper" in args.repo:
        info[args.backend] = whisper_export(args.repo, root / args.backend, args.backend)
    else:
        info[args.backend] = ctc_export(args.repo, root / args.backend, False, args.backend)
        if args.backend == "xnnpack":
            info["xnnpack-int8"] = ctc_export(args.repo, root / "xnnpack-int8", True, "xnnpack")
    info["seconds"] = round(time.perf_counter() - t0)
    root.mkdir(parents=True, exist_ok=True)
    (root / ("build.json" if args.backend == "xnnpack" else f"build_{args.backend}.json")).write_text(
        json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
