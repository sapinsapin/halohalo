"""Core ML: the Apple Neural Engine (and Apple GPU/CPU). Runs in venv_port_coreml.

  python -m porting.export_coreml sapinsapin/whisper-small-pld-ceb
  python -m porting.export_coreml sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm

Converts on Linux; predicts only on macOS, so nothing here is validated
locally — the macOS runner does it (docs/porting_pipeline.md).

Whisper: only the encoder goes to Core ML, the split WhisperKit and
whisper.cpp both use (the encoder is ~all the FLOPs and has a fixed 30 s
shape the ANE likes; the autoregressive decoder stays on CPU/GPU). The input
is named `logmel_data` and the output `output`, which is what whisper.cpp's
Core ML bridge calls, so on a Mac:

  xcrun coremlc compile coreml/encoder.mlpackage coreml/
  mv coreml/encoder.mlmodelc ggml/ggml-model-f16-encoder.mlmodelc   # beside the ggml model

CTC: the whole model at a fixed 10 s window (the same window the NPU QDQ
graph uses), fp16. At 963M parameters it exceeds what the ANE schedules well;
Core ML then runs it on the GPU, which is still the fastest Mac path for it.
"""

import argparse
import json
import os
import time
from pathlib import Path

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"
CTC_WINDOW = 10 * 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    args = ap.parse_args()
    import coremltools as ct
    import numpy as np
    import torch
    tok = os.environ.get("HF_TOKEN")
    name = args.repo.split("/")[-1]
    out = ART / name / "coreml"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if "whisper" in args.repo:
        from transformers import WhisperForConditionalGeneration
        model = WhisperForConditionalGeneration.from_pretrained(args.repo, token=tok, attn_implementation="eager").eval()
        enc = model.get_encoder()

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = enc

            def forward(self, logmel_data):
                return self.enc(logmel_data).last_hidden_state

        x = torch.zeros(1, model.config.num_mel_bins, 3000)
        traced = torch.jit.trace(Encoder().eval(), x)
        ml = ct.convert(traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.macOS14,
                        inputs=[ct.TensorType(name="logmel_data", shape=x.shape, dtype=np.float32)],
                        outputs=[ct.TensorType(name="output")])
        path = out / "encoder.mlpackage"
    else:
        from transformers import Wav2Vec2ForCTC
        model = Wav2Vec2ForCTC.from_pretrained(args.repo, token=tok, attn_implementation="eager").eval()

        class CTC(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, input_values):
                return self.m(input_values).logits

        x = torch.zeros(1, CTC_WINDOW)
        traced = torch.jit.trace(CTC().eval(), x)
        ml = ct.convert(traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.macOS14,
                        inputs=[ct.TensorType(name="input_values", shape=x.shape, dtype=np.float32)],
                        outputs=[ct.TensorType(name="logits")])
        path = out / "model.mlpackage"
    ml.short_description = f"{args.repo}, ported by halohalo porting/export_coreml.py"
    ml.save(str(path))
    mb = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20
    info = {path.name: round(mb, 1), "seconds": round(time.perf_counter() - t0),
            "coremltools": ct.__version__, "validated": "no: predicts only on macOS"}
    (out / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
