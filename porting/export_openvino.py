"""OpenVINO: Intel CPUs, iGPUs and the Core Ultra NPU. Runs in venv_port_ov.

  python -m porting.export_openvino sapinsapin/whisper-small-pld-ceb

optimum-intel's exporter, not a conversion of the ONNX graphs: it writes the
decoder as a *stateful* model (KV cache kept inside the runtime instead of
round-tripping through inputs and outputs), which is what OpenVINO GenAI's
WhisperPipeline wants and what makes the NPU path possible. Two variants:

  openvino/fp16   weights fp16 — iGPU and NPU default
  openvino/int8   NNCF int8 weight-only compression — CPU default, half the size

The NPU plugin needs static shapes; OpenVINO GenAI reshapes Whisper itself at
load time (`ov_genai.WhisperPipeline(path, "NPU")`), so one export serves all
three devices. The CTC model is exported as a plain IR the same way.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--variants", nargs="+", default=["fp16", "int8"])
    args = ap.parse_args()
    from optimum.intel import OVModelForCTC, OVModelForSpeechSeq2Seq, OVWeightQuantizationConfig

    tok = os.environ.get("HF_TOKEN")
    name = args.repo.split("/")[-1]
    whisper = "whisper" in args.repo
    root = ART / name / "openvino"
    log = {}
    for v in args.variants:
        out = root / v
        if (out / ".complete").exists():
            print(f"{name}: openvino/{v} exists")
            continue
        t0 = time.perf_counter()
        kw = {"export": True, "token": tok, "compile": False}
        if v == "int8":
            kw["quantization_config"] = OVWeightQuantizationConfig(bits=8)
        else:
            kw["load_in_8bit"] = False
        cls = OVModelForSpeechSeq2Seq if whisper else OVModelForCTC
        m = cls.from_pretrained(args.repo, **kw)
        if v == "fp16":
            m.half()
        m.save_pretrained(out)
        # tokenizer, processor and generation config beside the IR
        from porting.hfcompat import config_dir
        for f in config_dir(args.repo).iterdir():
            if f.suffix in (".json", ".txt") and f.name != "config.json" and not (out / f.name).exists():
                shutil.copy(f, out / f.name)
        (out / ".complete").touch()        # a crash mid-copy must not look finished
        mb = sum(p.stat().st_size for p in out.rglob("*.bin")) / 2**20
        log[v] = {"seconds": round(time.perf_counter() - t0), "weights_mb": round(mb, 1)}
        print(f"{name}: openvino/{v} {mb:.0f} MB ({log[v]['seconds']}s)")
    root.mkdir(parents=True, exist_ok=True)
    (root / "build.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
