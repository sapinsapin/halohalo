"""SNAC 24 kHz decoder: turns Orpheus's audio tokens into a waveform, so it
ships beside every Orpheus port. Small (19.8M parameters), so it is Phase 1
even though Orpheus is not.

  python -m porting.export_snac hubertsiuzdak/snac_24khz                      # venv_port_onnx
  python -m porting.export_snac hubertsiuzdak/snac_24khz --target executorch  # venv_port_et
  python -m porting.export_snac hubertsiuzdak/snac_24khz --target coreml      # venv_port_coreml

Inputs are the three code streams (12, 23 and 47 Hz; one Orpheus frame of
7 tokens = 1 + 2 + 4 codes). ONNX keeps the time axes dynamic. ExecuTorch and
Core ML take a fixed chunk of FRAMES frames — how Orpheus streams anyway
(decode a few frames at a time, keep the middle) — because mobile and NPU
runtimes want static shapes.

The decoder adds Gaussian noise inside (SNAC's NoiseBlock), so two PyTorch
runs already differ; porting/validate_snac.py measures a port against that
floor instead of expecting identical samples.
"""

import argparse
import json
import os
import time
from pathlib import Path

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"
FRAMES = 16          # fixed chunk for static-shape runtimes: 16 frames ~ 1.4 s


def load(repo):
    import types

    import torch
    from snac import SNAC
    model = SNAC.from_pretrained(repo).eval()

    # repeat_interleave(stride) exports as a Reshape to the traced length, so
    # the graph only accepts the example's frame count. The same upsampling as
    # unsqueeze-expand-reshape keeps the time axis symbolic.
    def from_codes(self, codes):
        z_q = 0.0
        for i in range(self.n_codebooks):
            z = self.quantizers[i].out_proj(self.quantizers[i].decode_code(codes[i]))
            s = self.quantizers[i].stride
            if s > 1:
                z = z.unsqueeze(-1).expand(-1, -1, -1, s).reshape(z.shape[0], z.shape[1], -1)
            z_q = z_q + z
        return z_q

    model.quantizer.from_codes = types.MethodType(from_codes, model.quantizer)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, c0, c1, c2):
            return self.m.decode([c0, c1, c2])

    return model, Decoder().eval()


def example(frames=FRAMES):
    import torch
    return (torch.randint(0, 4096, (1, frames)), torch.randint(0, 4096, (1, 2 * frames)),
            torch.randint(0, 4096, (1, 4 * frames)))


def to_onnx(repo, out: Path) -> dict:
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.transformers.float16 import convert_float_to_float16
    _, dec = load(repo)
    out.mkdir(parents=True, exist_ok=True)
    f32 = out / "decoder.onnx"
    names = ["audio_codes.0", "audio_codes.1", "audio_codes.2"]
    torch.onnx.export(dec, example(), str(f32), input_names=names, output_names=["audio_values"],
                      dynamic_axes={"audio_codes.0": {1: "t0"}, "audio_codes.1": {1: "t1"},
                                    "audio_codes.2": {1: "t2"}, "audio_values": {2: "samples"}},
                      opset_version=17, dynamo=False)
    m16 = convert_float_to_float16(onnx.load(str(f32)), keep_io_types=True,
                                   op_block_list=["RandomNormalLike", "RandomNormal"])
    onnx.save(m16, str(out / "decoder_fp16.onnx"))
    quantize_dynamic(str(f32), str(out / "decoder_quantized.onnx"), weight_type=QuantType.QInt8)
    return {p.name: round(p.stat().st_size / 2**20, 1) for p in out.glob("*.onnx")}


def to_executorch(repo, out: Path) -> dict:
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    _, dec = load(repo)
    out.mkdir(parents=True, exist_ok=True)
    ep = torch.export.export(dec, example())
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    (out / "decoder.pte").write_bytes(prog.buffer)
    return {"decoder.pte": round(len(prog.buffer) / 2**20, 1), "frames": FRAMES}


def to_coreml(repo, out: Path) -> dict:
    import coremltools as ct
    import numpy as np
    import torch
    _, dec = load(repo)
    traced = torch.jit.trace(dec, example())
    ml = ct.convert(traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.iOS17,
                    inputs=[ct.TensorType(name=n, shape=s.shape, dtype=np.int32)
                            for n, s in zip(["c0", "c1", "c2"], example())],
                    outputs=[ct.TensorType(name="audio_values")])
    out.mkdir(parents=True, exist_ok=True)
    ml.save(str(out / "decoder.mlpackage"))
    return {"decoder.mlpackage": "saved", "frames": FRAMES}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--target", default="onnx", choices=["onnx", "executorch", "coreml"])
    args = ap.parse_args()
    root = ART / args.repo.split("/")[-1]
    t0 = time.perf_counter()
    fn, sub, marker = {"onnx": (to_onnx, "snac", "build_snac.json"),
                       "executorch": (to_executorch, "executorch", "executorch/build.json"),
                       "coreml": (to_coreml, "coreml", "coreml/build.json")}[args.target]
    info = fn(args.repo, root / sub)
    info["seconds"] = round(time.perf_counter() - t0)
    (root / marker).parent.mkdir(parents=True, exist_ok=True)
    (root / marker).write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
