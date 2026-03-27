"""
Filipino Speech Corpus → Hugging Face AudioFolder format (Whisper-compatible)

Output structure:
  {OUTPUT_DIR}/
    train/
      metadata.jsonl
      audio/
        {stem}_{idx:04d}.wav
    test/
      metadata.jsonl
      audio/
        {stem}_{idx:04d}.wav

Load with:
  dataset = load_dataset("audiofolder", data_dir=OUTPUT_DIR)

metadata.jsonl fields (file_name is relative to the split directory):
  file_name, sentence, language, duration,
  speaker_id, gender, age_group, speech_type
"""

import json
import os
import re
import random
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

CORPUS_DIR  = Path(os.environ["CORPUS_DIR"])
OUTPUT_DIR  = Path(os.environ["OUTPUT_DIR"])
TARGET_SR   = 16000
MIN_DUR     = 1.5    # excludes isolated words (mean word dur ~0.6s)
MAX_DUR     = 30.0
MIN_WORDS   = 3      # skip single/double word segments (phonetic word lists)
SPLIT_RATIO = 0.9
RANDOM_SEED = 42

TRS_DIRS = {
    "read":        CORPUS_DIR / "Volume 6 (Transcriptions)" / "hand transcribed read speech",
    "spontaneous": CORPUS_DIR / "Volume 6 (Transcriptions)" / "hand transcribed spontaneous speech",
    "machine":     CORPUS_DIR / "Volume 6 (Transcriptions)" / "machine pre-segmented transcribed read speech",
}

WAV_DIRS = [
    CORPUS_DIR / "Volume 1",
    CORPUS_DIR / "Volume 2",
    CORPUS_DIR / "Volume 3",
    CORPUS_DIR / "Volume 4",
    CORPUS_DIR / "Volume 5 (Spontaneous Speech)",
]

SKIP_RE = re.compile(r"^\.\.$|^\{.*?\}$|^$", re.DOTALL)

AGE_MAP = ["20-27", "28-35", "36-43", "44-51", "52-60"]


def parse_speaker_meta(stem: str) -> dict:
    m = re.match(r"(\d+)_xx(\d)(\d)xxxx_(\d+)", stem)
    if not m:
        return {"speaker_id": stem, "gender": "unknown", "age_group": "unknown"}
    return {
        "speaker_id": m.group(1),
        "gender":     "female" if m.group(2) == "1" else "male",
        "age_group":  AGE_MAP[int(m.group(3))] if int(m.group(3)) < len(AGE_MAP) else "unknown",
    }


def clean_sentence(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    return " ".join(text.split()).strip()


def parse_trs(trs_path: Path) -> list[dict]:
    try:
        tree = ET.parse(trs_path)
    except ET.ParseError:
        print(f"  [warn] XML parse error: {trs_path.name}")
        return []

    segments = []
    for turn in tree.iter("Turn"):
        children = list(turn)
        for i, elem in enumerate(children):
            if elem.tag != "Sync":
                continue
            start = float(elem.get("time", 0))
            end = None
            for j in range(i + 1, len(children)):
                if children[j].tag == "Sync":
                    end = float(children[j].get("time", 0))
                    break
            if end is None:
                end = start + 5.0

            text = (elem.tail or "").strip()
            if SKIP_RE.match(text):
                continue
            text = clean_sentence(text)
            if not text:
                continue

            dur = end - start
            if dur <= 0 or dur < MIN_DUR or dur > MAX_DUR:
                continue
            if len(text.split()) < MIN_WORDS:
                continue

            segments.append({"start": start, "end": end, "text": text})

    return segments


def find_wav(stem: str) -> Path | None:
    for wav_dir in WAV_DIRS:
        for f in wav_dir.rglob(f"{stem}.wav"):
            return f
    return None


def extract_segment(wav_path: Path, start: float, end: float) -> np.ndarray | None:
    try:
        data, sr = sf.read(wav_path, always_2d=False)
    except Exception as e:
        print(f"  [warn] {wav_path.name}: {e}")
        return None

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != TARGET_SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return data[int(start * sr): int(end * sr)]


def collect_entries() -> list[dict]:
    """Collect all (meta, segment) pairs without loading audio into RAM."""
    entries = []
    skipped = 0

    for speech_type, trs_dir in TRS_DIRS.items():
        if not trs_dir.exists():
            print(f"  [skip] {trs_dir.name} not found")
            continue

        trs_files = sorted(f for f in trs_dir.glob("*.trs") if not f.name.endswith(".trs~"))
        print(f"  {speech_type}: {len(trs_files)} files")

        for trs_file in trs_files:
            stem = trs_file.stem
            wav_path = find_wav(stem)
            if wav_path is None:
                skipped += 1
                continue

            meta = parse_speaker_meta(stem)
            for i, seg in enumerate(parse_trs(trs_file)):
                entries.append({
                    "wav_path":   wav_path,
                    "start":      seg["start"],
                    "end":        seg["end"],
                    "stem":       stem,
                    "idx":        i,
                    "sentence":   seg["text"],
                    "duration":   round(seg["end"] - seg["start"], 3),
                    "speech_type": speech_type,
                    **meta,
                })

    print(f"\n  Skipped (no WAV): {skipped}")
    return entries


def write_split(entries: list[dict], split_dir: Path) -> int:
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(split_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for entry in entries:
            audio = extract_segment(entry["wav_path"], entry["start"], entry["end"])
            if audio is None or len(audio) == 0:
                continue

            clip_name = f"{entry['stem']}_{entry['idx']:04d}.wav"
            sf.write(audio_dir / clip_name, audio, TARGET_SR)

            f.write(json.dumps({
                "file_name":   f"audio/{clip_name}",
                "sentence":    entry["sentence"],
                "language":    "fil",
                "duration":    entry["duration"],
                "speaker_id":  entry["speaker_id"],
                "gender":      entry["gender"],
                "age_group":   entry["age_group"],
                "speech_type": entry["speech_type"],
            }, ensure_ascii=False) + "\n")
            written += 1

    return written


def main():
    print(f"Corpus : {CORPUS_DIR}")
    print(f"Output : {OUTPUT_DIR}\n")

    entries = collect_entries()
    print(f"Total segments collected: {len(entries)}")

    random.seed(RANDOM_SEED)
    random.shuffle(entries)
    split = int(len(entries) * SPLIT_RATIO)

    print("\nWriting train split...")
    n_train = write_split(entries[:split], OUTPUT_DIR / "train")

    print("Writing test split...")
    n_test = write_split(entries[split:], OUTPUT_DIR / "test")

    print(f"\nDone.")
    print(f"  train: {n_train}  →  {OUTPUT_DIR / 'train'}")
    print(f"  test : {n_test}   →  {OUTPUT_DIR / 'test'}")


if __name__ == "__main__":
    main()
