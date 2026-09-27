"""Make a repo's small config files loadable by the transformers each
toolchain pins.

The fine-tunes were saved by transformers 5, whose tokenizer_config.json names
the special-token list `extra_special_tokens` and adds a few bookkeeping keys;
transformers 4.x (pinned by optimum-onnx, optimum-intel, optimum-executorch)
expects `additional_special_tokens` and fails on the list. The tokens
themselves live in tokenizer.json either way, so the rename loses nothing.
"""

import json
import os
import shutil
from pathlib import Path

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"
V5_ONLY = ("backend", "is_local", "local_files_only")


def sanitize_tokenizer_config(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    c = json.loads(path.read_text())
    extra = c.get("extra_special_tokens")
    if isinstance(extra, list):
        c.pop("extra_special_tokens")
        c.setdefault("additional_special_tokens", extra)
    for k in V5_ONLY:
        c.pop(k, None)
    path.write_text(json.dumps(c, indent=1, ensure_ascii=False))


def config_dir(repo: str) -> Path:
    """The repo's *.json files (config, tokenizer, processor, generation),
    sanitised, in ART/<name>/hf-json/. Weights still load from the hub cache."""
    from huggingface_hub import snapshot_download
    out = ART / repo.split("/")[-1] / "hf-json"
    if not (out / "config.json").exists():
        src = Path(snapshot_download(repo, token=os.environ.get("HF_TOKEN"), allow_patterns=["*.json", "*.txt"]))
        out.mkdir(parents=True, exist_ok=True)
        # the snapshot folder is shared with earlier full downloads: take only
        # the small files, not a stray model.safetensors
        for f in src.iterdir():
            if f.suffix in (".json", ".txt"):
                shutil.copy(f, out / f.name)
        sanitize_tokenizer_config(out / "tokenizer_config.json")
        if not (out / "preprocessor_config.json").exists() and (out / "processor_config.json").exists():
            # transformers 5 folds the feature extractor into processor_config.json
            pc = json.loads((out / "processor_config.json").read_text())
            fe = pc.get("feature_extractor")
            if isinstance(fe, dict):
                (out / "preprocessor_config.json").write_text(json.dumps(fe, indent=1))
    return out
