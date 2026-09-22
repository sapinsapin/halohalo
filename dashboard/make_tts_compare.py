"""Regenerate tts_compare.json and samples/compare/ — the "Compare voices" tab:
the same unseen sentences spoken by a person and by each TTS system.

Why a listening tab instead of a live one: Orpheus is a 3B codec language
model, and on the Space's free CPU it generates ~1.5 tokens/s — about 2.7
minutes per three-second sentence (measured, 2 threads, bf16). So its audio is
generated offline with the evaluation's own settings and shipped as files.

Inputs are the outputs of scripts/tts_eval.py:

  <eval>/manifest.json            the frozen sentences, speaker, human clip
  <eval>/ref/<lang>/NN.wav        the human recording
  <eval>/out/<system>/<lang>/NN.wav
  <results>                       results.json from the run that scored them

Scores come from one results.json so that every number in a row was produced
by the same judge on the same sentences; Orpheus scores come from its model
cards, which were written by that same run. Run locally, where the eval
outputs live:

  python make_tts_compare.py --eval ../finetune_runs/tts_eval \\
      --results ../finetune_runs_nebius/tts/tts_eval/results.json
"""

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).parent
ORG = "sapinsapin"
ORPHEUS = f"{ORG}/orpheus-3b-0.1-pretrained-char-pld-{{lang}}"
# output dir name under <eval>/out  ->  label shown in the tab
SYSTEMS = {"speecht5": "SpeechT5", "mms": "MMS-TTS",
           "orpheus_char": "Orpheus 3B",
           "qwen3tts_base": "Qwen3-TTS 1.7B, zero-shot"}


def card_metrics(repo):
    try:
        card = Path(hf_hub_download(repo, "README.md",
                                    token=os.environ.get("HF_TOKEN")))
    except Exception:                                      # noqa: BLE001
        return {}
    got = re.findall(r"^\|\s*([a-z_]+)\s*\|\s*([0-9.]+)\s*\|\s*$",
                     card.read_text(encoding="utf-8"), re.M)
    return {k: float(v) for k, v in got}


def to_ogg(src: Path, dst: Path):
    """Vorbis keeps a clip near 25 KB where 16-bit WAV at 24 kHz is ~150 KB;
    the Space repo carries every one of these."""
    wav, sr = sf.read(src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    if peak > 1.0:                        # a few SNAC decodes overshoot
        wav = wav / peak * 0.95
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, wav, sr, format="OGG", subtype="VORBIS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--per-lang", type=int, default=2)
    args = ap.parse_args()

    rows = json.loads((args.eval / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    out_dir = ROOT / "samples" / "compare"
    # this script owns the directory: a smaller --per-lang than last time must
    # not leave the previous run's extra clips behind to be uploaded
    for old in out_dir.glob("*.ogg"):
        old.unlink()

    langs = {}
    for lang in sorted({r["lang"] for r in rows}):
        def score(system, field="cer"):
            return (results.get(system, {}).get(lang) or {}).get(field)
        orph = card_metrics(ORPHEUS.format(lang=lang))
        scores = {
            "human": score("reference"),
            "speecht5": score("speecht5"), "mms": score("mms"),
            "orpheus": orph.get("cer"),
            "speecht5_spk_sim": score("speecht5", "spk_sim"),
            "orpheus_spk_sim": orph.get("spk_sim"),
            "qwen3tts_base": score("qwen3tts_base"),
            "qwen3tts_base_spk_sim": score("qwen3tts_base", "spk_sim"),
        }
        clips = []
        for r in [r for r in rows if r["lang"] == lang][:args.per_lang]:
            files = {}
            src = {"human": args.eval / "ref" / lang / f"{r['i']:02d}.wav"}
            for system in SYSTEMS:
                src[system] = args.eval / "out" / system / lang / f"{r['i']:02d}.wav"
            # the eval writes each Orpheus adapter to its own dir
            per_lang = args.eval / "out" / f"orpheus_char_pld_{lang}" / lang / f"{r['i']:02d}.wav"
            if per_lang.exists():
                src["orpheus_char"] = per_lang
            for key, path in src.items():
                if path.exists():
                    rel = Path("samples/compare") / f"{lang}_{r['i']:02d}_{key}.ogg"
                    to_ogg(path, ROOT / rel)
                    files[key] = rel.as_posix()
            clips.append({"text": r["text"], "speaker": r["speaker_id"],
                          "files": files})
        langs[lang] = {"scores": scores, "clips": clips}
        have = sorted({k for c in clips for k in c["files"]})
        print(f"  {lang}: {len(clips)} sentences, systems {have}")

    out = ROOT / "tts_compare.json"
    out.write_text(json.dumps({"as_of": date.today().isoformat(),
                               "systems": SYSTEMS, "languages": langs},
                              ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    kb = sum(p.stat().st_size for p in out_dir.glob("*.ogg")) / 1024
    print(f"{out}: {len(langs)} languages, {kb:.0f} KB of audio")


if __name__ == "__main__":
    main()
