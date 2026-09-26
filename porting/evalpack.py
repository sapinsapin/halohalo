"""The fixed clips every ported artefact is checked against.

  venv/bin/python3 -m porting.evalpack --languages ceb pam --n 100

Takes the first N clips of each language's frozen speaker- and prompt-disjoint
test split and writes them as float32 16 kHz arrays plus references to
$FINETUNE_DIR/port/evalpack/<lang>.npz. Built once with the training venv
(which has the dataset loader); every toolchain venv then reads a plain .npz
and needs no `datasets`, no PLD access and no network.

The same clips also serve as the calibration set for static int8 (QDQ) export,
taken from the *train* split instead so calibration never sees a test clip:
<lang>_calib.npz.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:                                  # the toolchain venvs only read the .npz
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs"))
OUT = FINETUNE_DIR / "port" / "evalpack"


def write(path, rows):
    audio = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        audio[i] = np.asarray(r["audio"]["array"], dtype=np.float32)
    np.savez_compressed(path, audio=audio,
                        text=np.array([r["text"] for r in rows], dtype=object),
                        speaker=np.array([str(r["speaker_id"]) for r in rows], dtype=object))


def load(lang: str, split: str = "test"):
    """-> (list of float32 arrays at 16 kHz, list of reference strings)."""
    name = f"{lang}.npz" if split == "test" else f"{lang}_calib.npz"
    d = np.load(OUT / name, allow_pickle=True)
    return list(d["audio"]), [str(t) for t in d["text"]]


def export_wavs(lang: str) -> Path:
    """evalpack/wav/<lang>/NNN.wav + refs.json, for runtimes and pages that
    read files (whisper.cpp, the browser check)."""
    from porting.frontends import write_wavs
    audio, refs = load(lang)
    d = OUT / "wav" / lang
    write_wavs(audio, d)
    (d / "refs.json").write_text(json.dumps(refs, ensure_ascii=False))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--languages", nargs="+", default=["ceb", "pam"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--wavs-only", action="store_true", help="just (re)write evalpack/wav/<lang>/")
    args = ap.parse_args()
    if args.wavs_only:
        for lang in args.languages:
            print(export_wavs(lang))
        return
    from halolib.finetune import load_speech_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for lang in args.languages:
        ds = load_speech_dataset("pld", task="asr", language=lang, max_samples=None,
                                 token=os.environ.get("HF_TOKEN"))
        test = [ds["test"][i] for i in range(min(args.n, len(ds["test"])))]
        calib = [ds["train"][i] for i in range(min(args.calib, len(ds["train"])))]
        write(OUT / f"{lang}.npz", test)
        write(OUT / f"{lang}_calib.npz", calib)
        secs = sum(len(r["audio"]["array"]) for r in test) / 16000
        meta[lang] = {"test_clips": len(test), "test_seconds": round(secs, 1),
                      "calib_clips": len(calib)}
        print(f"  {lang}: {len(test)} test clips ({secs:.0f} s), {len(calib)} calibration clips")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
