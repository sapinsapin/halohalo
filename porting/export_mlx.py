"""MLX: Apple silicon (Metal, unified memory). Runs in venv_port_mlx.

  python -m porting.export_mlx sapinsapin/whisper-small-pld-ceb

Writes mlx-whisper's layout (config.json of model dimensions + weights.safetensors)
for three variants, the same way mlx-examples' convert.py does:

  mlx/fp16   the default on a Mac
  mlx/q8     8-bit, group 64 — near-lossless, half the memory
  mlx/q4     4-bit, group 64 — for 8 GB Macs and the larger checkpoints

MLX's Linux CPU wheel runs the same graph, so the transcripts are validated
here; speed on a Mac is a Metal number and is measured there.
"""

import argparse
import json
import os
from pathlib import Path

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"
QUANT = {"fp16": None, "q8": {"group_size": 64, "bits": 8}, "q4": {"group_size": 64, "bits": 4}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--variants", nargs="+", default=list(QUANT))
    args = ap.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    import torch
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_whisper.whisper import ModelDimensions, Whisper
    from porting.whisper_openai import convert

    name = args.repo.split("/")[-1]
    root = ART / name
    pt = root / "openai" / "model.pt"
    if not pt.exists():
        convert(args.repo)
    ck = torch.load(pt, map_location="cpu", weights_only=True)
    meta = json.loads((root / "openai" / "halohalo.json").read_text())

    def to_mlx(k, v):
        k = k.replace("mlp.0", "mlp1").replace("mlp.2", "mlp2")
        if "conv" in k and v.ndim == 3:           # MLX conv1d is (out, kernel, in)
            v = v.transpose(1, 2)
        return k, mx.array(v.numpy()).astype(mx.float16)

    weights = dict(to_mlx(k, v) for k, v in ck["model_state_dict"].items()
                   if k != "encoder.positional_embedding")   # sinusoids, rebuilt by MLX
    for v in args.variants:
        out = root / "mlx" / v
        out.mkdir(parents=True, exist_ok=True)
        cfg = dict(ck["dims"])
        w = weights
        if QUANT[v]:
            model = Whisper(ModelDimensions(**ck["dims"]), mx.float16)
            model.update(tree_unflatten(list(weights.items())))
            nn.quantize(model, **QUANT[v])
            w = dict(tree_flatten(model.parameters()))
            w.pop("encoder._positional_embedding", None)
            cfg["quantization"] = QUANT[v]
        mx.save_safetensors(str(out / "weights.safetensors"), w)
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
        (out / "halohalo.json").write_text(json.dumps(meta, indent=1))
        mb = (out / "weights.safetensors").stat().st_size / 2**20
        print(f"{name}: mlx/{v} {mb:.0f} MB")


if __name__ == "__main__":
    main()
