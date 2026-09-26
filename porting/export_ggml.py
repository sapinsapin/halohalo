"""whisper.cpp: the most-used Whisper runtime on Arm CPUs (Android, Raspberry
Pi, Graviton), and on AMD/Intel via Vulkan or HIP. Runs in venv_port_onnx.

  python -m porting.export_ggml sapinsapin/whisper-small-pld-ceb

HF checkpoint -> OpenAI layout (porting.whisper_openai) -> whisper.cpp's own
convert-pt-to-ggml.py -> its quantize tool:

  ggml/ggml-model-f16.bin     reference
  ggml/ggml-model-q8_0.bin    near-lossless
  ggml/ggml-model-q5_0.bin    the usual phone choice

Tools: whisper.cpp and openai/whisper (for the mel filters and tokenizer the
converter reads) are cloned to $PORT_TOOLS (default /mnt/d/halohalo/third_party)
by `bash porting/setup_venvs.sh ggml`.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"
TOOLS = Path(os.environ.get("PORT_TOOLS", "/mnt/d/halohalo/third_party"))
QUANTS = ["q8_0", "q5_0"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--quants", nargs="*", default=QUANTS)
    args = ap.parse_args()
    from porting.whisper_openai import convert

    name = args.repo.split("/")[-1]
    root = ART / name
    pt = root / "openai" / "model.pt"
    if not pt.exists():
        convert(args.repo)
    out = root / "ggml"
    out.mkdir(parents=True, exist_ok=True)
    f16 = out / "ggml-model-f16.bin"
    if not f16.exists():
        subprocess.run([sys.executable, str(TOOLS / "whisper.cpp" / "models" / "convert-pt-to-ggml.py"),
                        str(pt), str(TOOLS / "whisper"), str(out)], check=True)
        (out / "ggml-model.bin").rename(f16)
    shutil.copy(root / "openai" / "halohalo.json", out / "halohalo.json")
    q = TOOLS / "whisper.cpp" / "build" / "bin" / "whisper-quantize"
    if not q.exists():
        q = q.with_name("quantize")
    for t in args.quants:
        dst = out / f"ggml-model-{t}.bin"
        if not dst.exists():
            subprocess.run([str(q), str(f16), str(dst), t], check=True,
                           stdout=subprocess.DEVNULL)
    sizes = {p.name: round(p.stat().st_size / 2**20, 1) for p in out.glob("*.bin")}
    (out / "build.json").write_text(json.dumps(sizes, indent=1))
    print(f"{name}: {sizes}")


if __name__ == "__main__":
    main()
