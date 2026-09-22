"""
Synthesize the frozen eval sentences with a finetuned Qwen3-TTS, into the
layout scripts/tts_eval.py scores. Runs in venv_qwen; the scoring runs in the
main venv afterwards:

  venv_qwen/bin/python3 scripts/synth_qwen_tts.py --language ceb pam
  venv/bin/python3 scripts/tts_eval.py --stage score --device cuda \
      --model qwen3tts_pld_ceb --model qwen3tts_pld_pam

The model is in base mode, so a voice is supplied per sentence as a reference
clip plus its transcript. The reference is a *different* sentence by the same
speaker from the frozen set: the same sentence's own recording would hand the
model the answer, and a score built on that would measure copying. Speakers
with only one sentence in the set fall back to their own clip, and the count is
reported so it can be discounted.

Base model zero-shot is synthesized alongside as its own row, because a
finetune that loses to the model it started from is a result worth having.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/data/finetune_runs"))
WORK = FINETUNE_DIR / "tts_eval"
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MAX_TOKENS = int(25 * 12.5)


def pick_refs(rows):
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
    refs, self_refs = {}, 0
    for r in rows:
        others = [o for o in by_spk[r["speaker_id"]] if o["i"] != r["i"]]
        if others:
            refs[r["i"]] = others[(r["i"] * 7) % len(others)]   # fixed, spread out
        else:
            refs[r["i"]] = r
            self_refs += 1
    return refs, self_refs


def synth(model_path, name, rows, device):
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(model_path, device_map=device,
                                          dtype=torch.bfloat16,
                                          attn_implementation="sdpa")
    langs = sorted({r["lang"] for r in rows})
    for lang in langs:
        lrows = [r for r in rows if r["lang"] == lang]
        refs, self_refs = pick_refs(lrows)
        out_dir = WORK / "out" / name / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        done = 0
        for r in lrows:
            out = out_dir / f"{r['i']:02d}.wav"
            if out.exists():
                done += 1
                continue
            ref = refs[r["i"]]
            # Cap generation. The default is 8192 codec tokens, eleven minutes
            # at 12.5 Hz, and the first run produced 136 s clips for 4 s
            # sentences: a model that has not learned to stop will fill that.
            # 25 s is generous for any sentence in the set; a clip that hits
            # it is a failure the duration ratio in the score will show.
            wavs, sr = model.generate_voice_clone(
                text=r["text"], language="Auto",
                ref_audio=ref["ref"], ref_text=ref["text"],
                max_new_tokens=MAX_TOKENS)
            sf.write(str(out), wavs[0], sr)
            done += 1
        print(f"  {name} {lang}: {done} wavs ({self_refs} self-referenced)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--language", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-base", action="store_true",
                    help="skip the zero-shot base-model row")
    args = ap.parse_args()

    rows = [r for r in json.loads((WORK / "manifest.json").read_text())
            if r["lang"] in args.language]
    for lang in args.language:
        final = FINETUNE_DIR / f"qwen3tts_pld_{lang}" / "final"
        if not final.is_dir():
            print(f"  no finetune for {lang} at {final}, skipping")
            continue
        synth(str(final), f"qwen3tts_pld_{lang}",
              [r for r in rows if r["lang"] == lang], args.device)
    if not args.no_base:
        synth(BASE, "qwen3tts_base", rows, args.device)


if __name__ == "__main__":
    main()
