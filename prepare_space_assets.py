"""
Build the static assets the demo Space needs: preloaded audio clips, their
transcripts, and speaker embeddings.

Two reasons the embeddings are precomputed here rather than in the Space:
SpeechT5 cannot synthesize or convert without an x-vector, and running
speechbrain inside a free CPU Space would add a heavy dependency and a
cold-start penalty to every request. A 512-float vector per voice is 2KB.

Runs on CPU by default so it can sit alongside GPU training.

Usage:
  python prepare_space_assets.py --out spaces/halohalo-speech-demo
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

SR = 16000
LANGS = ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]
LANG_NAMES = {"bcl": "Bikol", "ceb": "Cebuano", "eng": "English (PH)",
              "fil": "Filipino", "hil": "Hiligaynon", "ilo": "Ilocano",
              "pag": "Pangasinan", "pam": "Kapampangan", "tsg": "Tausug",
              "war": "Waray"}

CLIPS_PER_LANG = 2         # preloaded demo audio
VOICES_PER_LANG = 2        # voice-conversion target presets
EMB_CLIPS = 12             # clips averaged into one voice embedding
MIN_SECS, MAX_SECS = 2.0, 8.0


def build_embedder(device: str):
    from speechbrain.inference.speaker import EncoderClassifier
    savedir = Path(os.environ.get("HF_HOME", "~/.cache")).expanduser() / "speechbrain-xvect"
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=str(savedir), run_opts={"device": device})


def embed(embedder, wavs: list[np.ndarray]) -> np.ndarray:
    """Mean x-vector over several clips — one clip is a noisy voice estimate."""
    embs = []
    for w in wavs:
        with torch.no_grad():
            e = embedder.encode_batch(torch.tensor(w, dtype=torch.float32).unsqueeze(0))
            embs.append(torch.nn.functional.normalize(e, dim=2).squeeze().cpu().numpy())
    return np.mean(embs, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="spaces/halohalo-speech-demo")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import soundfile as sf

    from halolib.pld import index_corpus

    root = Path(os.environ.get(
        "PLD_RAW", "/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD"))
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "xvectors").mkdir(parents=True, exist_ok=True)

    print("Indexing PLD...")
    entries, _ = index_corpus(root)

    by_lang = defaultdict(list)
    for e in entries:
        if (e["speech_type"] == "read" and not e["text_is_prompt"]
                and 4 <= e["num_words"] <= 14):
            by_lang[e["language"]].append(e)

    rng = random.Random(7)
    embedder = build_embedder(args.device)
    manifest = {"languages": {}, "source": "UP Diliman DSP Laboratory (PLD)"}

    for lang in LANGS:
        pool = by_lang.get(lang, [])
        if not pool:
            print(f"  {lang}: no clips, skipping")
            continue

        by_spk = defaultdict(list)
        for e in pool:
            by_spk[e["speaker_id"]].append(e)
        speakers = sorted(by_spk, key=lambda s: -len(by_spk[s]))

        def read(entry):
            w, sr = sf.read(entry["wav_path"], dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            return w, len(w) / sr

        # demo clips: distinct speakers, bounded length
        clips = []
        for spk in speakers:
            if len(clips) >= CLIPS_PER_LANG:
                break
            cands = rng.sample(by_spk[spk], min(6, len(by_spk[spk])))
            for c in cands:
                try:
                    w, dur = read(c)
                except Exception:
                    continue
                if MIN_SECS <= dur <= MAX_SECS:
                    name = f"{lang}_{len(clips)}.wav"
                    sf.write(out / "samples" / name, w, SR)
                    clips.append({"file": f"samples/{name}",
                                  "transcript": c["sentence"],
                                  "speaker": c["speaker_id"],
                                  "seconds": round(dur, 2)})
                    break

        # voice presets: a mean x-vector per speaker
        voices = []
        for spk in speakers[:VOICES_PER_LANG]:
            wavs = []
            for c in rng.sample(by_spk[spk], min(EMB_CLIPS, len(by_spk[spk]))):
                try:
                    w, dur = read(c)
                except Exception:
                    continue
                if 1.0 <= dur <= 15.0:
                    wavs.append(w)
            if not wavs:
                continue
            vec = embed(embedder, wavs)
            vname = f"xvectors/{lang}_{spk}.npy"
            np.save(out / vname, vec)
            gender = by_spk[spk][0].get("gender", "unknown")
            voices.append({"id": spk, "file": vname, "gender": gender,
                           "clips": len(wavs)})

        manifest["languages"][lang] = {
            "name": LANG_NAMES[lang], "clips": clips, "voices": voices}
        print(f"  {lang}: {len(clips)} clips, {len(voices)} voices")

    (out / "assets.json").write_text(json.dumps(manifest, indent=2,
                                                ensure_ascii=False),
                                     encoding="utf-8")
    print(f"Wrote {out / 'assets.json'}")


if __name__ == "__main__":
    main()
