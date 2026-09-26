"""The language identifier (fastText) in the browser: WebAssembly vs the
Python/C++ build it was trained with. Runs in the main venv (has fasttext).

  python -m porting.validate_lid sapinsapin/halo-lid --variant wasm

Nothing is converted — fastText compiles to wasm as-is and reads the same
model.ftz — so the check is parity: the same top label on every sentence,
probabilities within float tolerance. Sentences are the TTS evaluation
manifest (50 per language, ten languages), not training text.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"
RESULTS = FINETUNE_DIR / "port" / "results"


def build(repo) -> Path:
    from huggingface_hub import hf_hub_download
    out = ART / repo.split("/")[-1] / "web"
    out.mkdir(parents=True, exist_ok=True)
    m = out / "model.ftz"
    if not m.exists():
        shutil.copy(hf_hub_download(repo, "model.ftz", token=os.environ.get("HF_TOKEN")), m)
    (out / "build.json").write_text(json.dumps(
        {"model.ftz": round(m.stat().st_size / 2**20, 2), "runtime": "fasttext.wasm.js 1.0.0"}, indent=1))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--variant", default="wasm")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    m = build(args.repo)
    if args.build_only:
        return
    import fasttext
    rows = json.loads((FINETUNE_DIR / "tts_eval" / "manifest.json").read_text())
    texts = [r["text"].replace("\n", " ") for r in rows]
    gold = [r["lang"] for r in rows]
    py = fasttext.load_model(str(m))
    py_labels, py_probs = [], []
    for t in texts:
        l, p = py.predict(t, k=1)
        py_labels.append(l[0])
        py_probs.append(float(p[0]))
    job = RESULTS / "tmp_lid_job.json"
    job.parent.mkdir(parents=True, exist_ok=True)
    job.write_text(json.dumps({"model": str(m), "texts": texts}))
    out = subprocess.run(["node", str(ROOT / "porting" / "web" / "lid.mjs"), str(job)],
                         capture_output=True, text=True, cwd=ROOT / "porting" / "web")
    if out.returncode:
        sys.exit(out.stderr[-2000:])
    wasm = json.loads(out.stdout.strip().splitlines()[-1])
    strip = lambda l: (l or "").replace("__label__", "")
    same = sum(a == b for a, b in zip(py_labels, wasm["labels"]))
    res = {"repo": args.repo, "variant": args.variant, "sentences": len(texts),
           "identical_top_label": same / len(texts),
           "max_prob_diff": max(abs(a - b) for a, b in zip(py_probs, wasm["probs"])),
           "accuracy_python": sum(strip(l) == g for l, g in zip(py_labels, gold)) / len(gold),
           "accuracy_wasm": sum(strip(l) == g for l, g in zip(wasm["labels"], gold)) / len(gold),
           "model_mb": round(m.stat().st_size / 2**20, 2)}
    (RESULTS / args.repo.split("/")[-1]).mkdir(parents=True, exist_ok=True)
    (RESULTS / args.repo.split("/")[-1] / f"lid-{args.variant}.json").write_text(json.dumps(res, indent=1))
    job.unlink()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
