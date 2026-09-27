"""Parity of a ported SNAC decoder against PyTorch.

  python -m porting.validate_snac hubertsiuzdak/snac_24khz --variant fp32|fp16|int8   # ORT, venv_port_onnx
  python -m porting.validate_snac hubertsiuzdak/snac_24khz --variant executorch        # venv_port_et

Codes come from encoding the ceb evalpack clips with SNAC's own encoder
(resampled to 24 kHz). Each clip is decoded by PyTorch twice and by the port
once; the score is the log-mel distance (dB) port-vs-PyTorch next to the
PyTorch-vs-PyTorch floor that SNAC's internal noise alone produces. A port at
the floor is as faithful as the original is to itself.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from porting.evalpack import load as load_pack

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"
RESULTS = FINETUNE_DIR / "port" / "results"
THREADS = int(os.environ.get("PORT_THREADS", "4"))


def logmel(x, sr=24000, n_fft=1024, hop=256, n_mels=80):
    import torch
    x = torch.as_tensor(np.asarray(x, dtype=np.float32)).flatten()
    spec = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs() ** 2
    f = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    edges = torch.tensor(np.linspace(mel(0), mel(sr / 2), n_mels + 2))
    edges = 700 * (10 ** (edges / 2595) - 1)
    fb = torch.zeros(n_mels, len(f))
    for i in range(n_mels):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = torch.clamp(torch.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), min=0)
    return 10 * torch.log10(fb @ spec + 1e-8)


def dist_db(a, b):
    n = min(len(np.asarray(a).flatten()), len(np.asarray(b).flatten()))
    return float((logmel(np.asarray(a).flatten()[:n]) - logmel(np.asarray(b).flatten()[:n])).abs().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    import torch
    from snac import SNAC
    torch.set_num_threads(THREADS)
    name = args.repo.split("/")[-1]
    root = ART / name
    model = SNAC.from_pretrained(args.repo).eval()
    audio, _ = load_pack("ceb")
    audio = audio[:args.n]

    if args.variant == "executorch":
        from executorch.runtime import Runtime
        from porting.export_snac import FRAMES
        method = Runtime.get().load_program(str(root / "executorch" / "decoder.pte")).load_method("forward")

        def port(codes):          # fixed chunks of FRAMES frames, concatenated
            outs = []
            t0 = codes[0].shape[1]
            for s in range(0, t0 - FRAMES + 1, FRAMES):
                chunk = [codes[0][:, s:s + FRAMES], codes[1][:, 2 * s:2 * (s + FRAMES)],
                         codes[2][:, 4 * s:4 * (s + FRAMES)]]
                outs.append(method.execute(chunk)[0].numpy().flatten())
            return np.concatenate(outs) if outs else np.zeros(1, np.float32)
    elif args.variant.startswith("coreml"):     # coreml, coreml-ane, coreml-gpu, coreml-cpu (macOS)
        import coremltools as ct
        from porting.export_snac import FRAMES
        from porting.validate import COREML_UNITS
        unit = args.variant.split("-")[1] if "-" in args.variant else "all"
        ml = ct.models.MLModel(str(root / "coreml" / "decoder.mlpackage"),
                               compute_units=getattr(ct.ComputeUnit, COREML_UNITS[unit]))

        def port(codes):
            outs = []
            for s in range(0, codes[0].shape[1] - FRAMES + 1, FRAMES):
                feed = {"c0": codes[0][:, s:s + FRAMES].numpy().astype(np.int32),
                        "c1": codes[1][:, 2 * s:2 * (s + FRAMES)].numpy().astype(np.int32),
                        "c2": codes[2][:, 4 * s:4 * (s + FRAMES)].numpy().astype(np.int32)}
                outs.append(np.asarray(ml.predict(feed)["audio_values"], np.float32).flatten())
            return np.concatenate(outs) if outs else np.zeros(1, np.float32)
    else:
        import onnxruntime as ort
        f = {"fp32": "decoder.onnx", "fp16": "decoder_fp16.onnx", "int8": "decoder_quantized.onnx"}[args.variant]
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        sess = ort.InferenceSession(str(root / "snac" / f), so, providers=["CPUExecutionProvider"])

        def port(codes):
            feeds = {f"audio_codes.{i}": c.numpy() for i, c in enumerate(codes)}
            return sess.run(None, feeds)[0].flatten()

    floor, gap, secs, wall = [], [], 0.0, 0.0
    with torch.inference_mode():
        for a in audio:
            x = torch.as_tensor(np.asarray(a, np.float32))[None, None]
            x24 = torch.nn.functional.interpolate(x, scale_factor=1.5, mode="linear")
            codes = model.encode(x24)
            if args.variant == "executorch" or args.variant.startswith("coreml"):   # whole chunks only
                from porting.export_snac import FRAMES
                k = (codes[0].shape[1] // FRAMES) * FRAMES
                if k == 0:
                    continue
                codes = [codes[0][:, :k], codes[1][:, :2 * k], codes[2][:, :4 * k]]
            ref_a = model.decode(codes).numpy().flatten()
            ref_b = model.decode(codes).numpy().flatten()
            t0 = time.perf_counter()
            got = port(codes)
            wall += time.perf_counter() - t0
            secs += len(got) / 24000
            floor.append(dist_db(ref_a, ref_b))
            gap.append(dist_db(ref_a, got))
    host = os.environ.get("PORT_HOST")
    tag = f"{args.variant}@{host}" if host else args.variant
    res = {"repo": args.repo, "variant": tag, "clips": len(gap),
           "logmel_db_port_vs_torch": round(float(np.mean(gap)), 3),
           "logmel_db_torch_vs_torch": round(float(np.mean(floor)), 3),
           "rtf": round(wall / max(secs, 1e-9), 4), "threads": THREADS}
    res["parity"] = "at floor" if res["logmel_db_port_vs_torch"] <= 1.25 * res["logmel_db_torch_vs_torch"] + 0.1 else "above floor"
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"snac-{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
