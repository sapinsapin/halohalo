"""
Export PLD or FSC to the JSONL that Qwen3-TTS's finetuning code expects.

Plan item P3 in docs/tts_sota_plan.md. Upstream (QwenLM/Qwen3-TTS, Apache-2.0)
ships `finetuning/{prepare_data,dataset,sft_12hz}.py`, so we do not write a
trainer — we write its input. One row per clip:

    {"audio": "<24kHz wav>", "text": "...", "language": "...",
     "ref_audio": "<24kHz wav of the same speaker, different utterance>"}

Then upstream's prepare_data.py adds `audio_codes` (batched codec encoding, 32
clips at a time) and sft_12hz.py trains on the result.

Three things this script exists to get right:

1. **24 kHz on disk.** `TTSDataset.extract_mels` asserts `sr == 24000` and PLD
   is 16 kHz, so the export resamples. 16k -> 24k is exactly 3/2, so polyphase
   does it in ~1.5 ms a clip where librosa's default took 60 ms (measured on
   the VM, 2026-09-19).

2. **ref_audio must not be the clip itself.** The reference is what the model
   conditions on to know the voice. Point it at the target clip and the model
   is handed the answer: training loss collapses and inference, where no such
   clip exists, falls apart. Each row therefore references a *different*
   utterance by the same speaker, chosen deterministically.

3. **Speakers stay inside their split.** The pairing draws only from the same
   split, so a test row never references training audio.

  python scripts/export_qwen_tts.py --dataset pld --language ceb
  python scripts/export_qwen_tts.py --dataset pld --language ceb --split test
"""

import argparse
import json
import os
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from scipy.signal import resample_poly

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

QWEN_SR = 24000
# Qwen's own language names; PLD's ISO 639-3 codes are not among the ten it was
# trained on, so "Auto" is the honest setting — the dataset's default too.
LANGUAGE = "Auto"


def to_qwen_rate(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == QWEN_SR:
        return wav
    r = Fraction(QWEN_SR, int(sr)).limit_denominator(1000)
    return resample_poly(wav, r.numerator, r.denominator).astype(np.float32)


def pick_references(speakers: list[str], seed: int = 13) -> list[int]:
    """For each row, the index of another row by the same speaker.

    Speakers with only one clip in this split reference themselves — there is
    no alternative, and dropping them would silently shrink the corpus. The
    count is reported so the trade-off is visible rather than assumed.
    """
    by_speaker: dict[str, list[int]] = {}
    for i, s in enumerate(speakers):
        by_speaker.setdefault(s, []).append(i)

    rng = random.Random(seed)
    refs, singletons = [], 0
    for i, s in enumerate(speakers):
        others = [j for j in by_speaker[s] if j != i]
        if others:
            refs.append(rng.choice(others))
        else:
            refs.append(i)
            singletons += 1
    if singletons:
        print(f"  {singletons} clips are their speaker's only one in this "
              f"split and reference themselves")
    return refs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", choices=["pld", "fsc", "livestream"],
                    default="pld")
    ap.add_argument("--language", default=None, help="ISO 639-3 filter (pld)")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--max-samples", type=int, default=20000)
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="output dir (default $PLD_WORK_DIR/qwen_tts/<tag>)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from halolib.finetune import load_speech_dataset

    tag = args.dataset + (f"_{args.language}" if args.language else "")
    out_dir = Path(args.out or (Path(os.environ.get("PLD_WORK_DIR", "."))
                                / "qwen_tts" / tag))
    wav_dir = out_dir / f"{args.split}_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    ds = load_speech_dataset(args.dataset, task="tts",
                             max_samples=args.max_samples,
                             token=os.environ.get("HF_TOKEN"),
                             num_proc=args.num_proc,
                             language=args.language)
    split = ds[args.split]
    print(f"{tag} {args.split}: {len(split)} clips -> {out_dir}")

    paths, texts, speakers = [], [], []
    for i, row in enumerate(split):
        audio = row["audio"]
        wav = to_qwen_rate(np.asarray(audio["array"], dtype=np.float32),
                           audio["sampling_rate"])
        p = wav_dir / f"{i:06d}.wav"
        # PCM_16 rather than the float32 default: half the disk for a
        # corpus that is 16-bit at source anyway, and the VM shares one
        # 484 GB root disk between weights, caches and checkpoints
        sf.write(p, wav, QWEN_SR, subtype="PCM_16")
        paths.append(str(p))
        texts.append(row["text"])
        speakers.append(str(row["speaker_id"]))

    refs = pick_references(speakers)
    jsonl = out_dir / f"{args.split}.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i, (p, t) in enumerate(zip(paths, texts)):
            f.write(json.dumps({"audio": p, "text": t, "language": LANGUAGE,
                                "ref_audio": paths[refs[i]]},
                               ensure_ascii=False) + "\n")

    print(f"  wrote {len(paths)} rows: {jsonl}")
    print(f"  next: venv/bin/python3 Qwen3-TTS/finetuning/prepare_data.py "
          f"--input_jsonl {jsonl} --output_jsonl {out_dir}/{args.split}_coded.jsonl")


if __name__ == "__main__":
    main()
