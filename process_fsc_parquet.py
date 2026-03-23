"""
Filipino Speech Corpus → Hugging Face Parquet dataset

Stores all sentence-level segments as a single dataset with raw audio bytes.
Downstream consumers apply their own filters and resample as needed.

Usage:
  Whisper/ASR:
    ds = load_dataset("sapinsapin/filipinospeechcorpus")
    ds = ds.filter(lambda x: x["num_words"] >= 3 and x["duration"] >= 1.5)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

  TTS:
    ds = ds.filter(lambda x: x["speech_type"] == "read" and 1.0 <= x["duration"] <= 10.0)
    ds = ds.cast_column("audio", Audio(sampling_rate=22050))
"""

import io
import json
import logging
import os
import re
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import soundfile as sf
from datasets import Audio, Dataset
from dotenv import load_dotenv
from huggingface_hub import DatasetCard, HfApi

load_dotenv(Path(__file__).parent / ".env")

CORPUS_DIR  = Path(os.environ["CORPUS_DIR"])
HF_REPO     = os.environ["HF_REPO"]
HF_TOKEN    = os.environ["HF_TOKEN"]
SPLIT_RATIO  = 0.9
RANDOM_SEED  = 42
SHARD_SIZE   = 2000   # rows per shard — keeps each shard ~30MB in RAM
NUM_WORKERS  = os.cpu_count() or 4
LOG_DIR      = Path("/mnt/c/halohalo/fsc_shards")
RESUME_LOG   = LOG_DIR / "progress.jsonl"
RUN_LOG      = LOG_DIR / "run.log"

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
AGE_MAP  = ["20-27", "28-35", "36-43", "44-51", "52-60"]


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(RUN_LOG),
            logging.StreamHandler(),
        ],
    )


def load_completed_shards() -> set[str]:
    """Return set of 'split/shard_idx' strings already uploaded."""
    if not RESUME_LOG.exists():
        return set()
    done = set()
    with RESUME_LOG.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
                done.add(f"{entry['split']}/{entry['shard_idx']}")
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def log_shard_done(split: str, shard_idx: int, shard_name: str, rows: int, mb: float):
    with RESUME_LOG.open("a") as f:
        f.write(json.dumps({
            "split":      split,
            "shard_idx":  shard_idx,
            "shard_name": shard_name,
            "rows":       rows,
            "mb":         round(mb, 2),
            "ts":         datetime.now(timezone.utc).isoformat(),
        }) + "\n")


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
            if dur <= 0:
                continue

            segments.append({"start": start, "end": end, "text": text})

    return segments


def build_wav_index() -> dict[str, Path]:
    """Build stem→path lookup once instead of rglob per file."""
    index = {}
    for wav_dir in WAV_DIRS:
        for f in wav_dir.rglob("*.wav"):
            index[f.stem] = f
    return index


def audio_to_bytes(wav_path: Path, start: float, end: float, sr: int) -> bytes | None:
    try:
        data, file_sr = sf.read(wav_path, always_2d=False)
    except Exception as e:
        print(f"  [warn] {wav_path.name}: {e}")
        return None

    if data.ndim > 1:
        data = data.mean(axis=1)

    if file_sr != sr:
        import librosa
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)

    segment = data[int(start * sr): int(end * sr)]
    if len(segment) == 0:
        return None

    buf = io.BytesIO()
    sf.write(buf, segment, sr, format="WAV")
    return buf.getvalue()


def _process_wav_group(entries: list[dict]) -> list[dict]:
    """Read a WAV file once and slice all segments from it."""
    wav_path = entries[0]["wav_path"]
    sr = 16000
    try:
        data, file_sr = sf.read(wav_path, always_2d=False)
    except Exception as e:
        print(f"  [warn] {wav_path.name}: {e}")
        return []
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        import librosa
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)

    rows = []
    for entry in entries:
        segment = data[int(entry["start"] * sr): int(entry["end"] * sr)]
        if len(segment) == 0:
            continue
        buf = io.BytesIO()
        sf.write(buf, segment, sr, format="WAV")
        rows.append({
            "audio":       {"bytes": buf.getvalue(), "path": None},
            "sentence":    entry["sentence"],
            "duration":    entry["duration"],
            "num_words":   entry["num_words"],
            "speaker_id":  entry["speaker_id"],
            "gender":      entry["gender"],
            "age_group":   entry["age_group"],
            "speech_type": entry["speech_type"],
            "source_file": entry["source_file"],
        })
    return rows


def _index_trs_file(args: tuple) -> list[dict]:
    """Parse one TRS file — runs in ThreadPoolExecutor."""
    trs_file, speech_type, wav_index = args
    stem = trs_file.stem
    wav_path = wav_index.get(stem)
    if wav_path is None:
        return []
    meta = parse_speaker_meta(stem)
    rows = []
    for seg in parse_trs(trs_file):
        rows.append({
            "wav_path":    wav_path,
            "start":       seg["start"],
            "end":         seg["end"],
            "sentence":    seg["text"],
            "duration":    round(seg["end"] - seg["start"], 3),
            "num_words":   len(seg["text"].split()),
            "speaker_id":  meta["speaker_id"],
            "gender":      meta["gender"],
            "age_group":   meta["age_group"],
            "speech_type": speech_type,
            "source_file": stem,
        })
    return rows


def rows_to_dataset(rows: list[dict]) -> Dataset:
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    ds = Dataset.from_dict(cols)
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def main():
    setup_logging()
    log = logging.getLogger(__name__)
    log.info(f"Corpus: {CORPUS_DIR} | workers: {NUM_WORKERS}")

    completed = load_completed_shards()
    if completed:
        log.info(f"Resuming — {len(completed)} shards already done: {sorted(completed)}")

    # --- Pass 1: build WAV index once, then parse TRS files in parallel ---
    log.info("Pass 1: building WAV index...")
    wav_index = build_wav_index()
    log.info(f"  WAV files indexed: {len(wav_index)}")

    log.info("Pass 1: indexing segments (parallel TRS parse)...")
    tasks = [
        (trs_file, speech_type, wav_index)
        for speech_type, trs_dir in TRS_DIRS.items()
        if trs_dir.exists()
        for trs_file in sorted(f for f in trs_dir.glob("*.trs") if not f.name.endswith(".trs~"))
    ]
    index = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for rows in pool.map(_index_trs_file, tasks):
            if not rows:
                skipped += 1
            index.extend(rows)

    log.info(f"  Total segments: {len(index)} | Skipped (no WAV): {skipped}")

    random.seed(RANDOM_SEED)
    random.shuffle(index)
    split = int(len(index) * SPLIT_RATIO)
    splits = {"train": index[:split], "test": index[split:]}

    # --- Pass 2: process audio in parallel, upload in background thread ---
    LOG_DIR.mkdir(exist_ok=True)
    HfApi(token=HF_TOKEN).create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)

    def upload_and_delete(path: Path, repo_path: str, split_name: str, shard_idx: int, rows: int, mb: float):
        log.info(f"  uploading {path.name}...")
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=str(path),
            path_in_repo=repo_path,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        path.unlink()
        log_shard_done(split_name, shard_idx, path.name, rows, mb)
        log.info(f"  uploaded + logged: {path.name}")

    for split_name, entries in splits.items():
        n_shards = (len(entries) + SHARD_SIZE - 1) // SHARD_SIZE
        log.info(f"\nPass 2: {split_name} — {len(entries)} segments, {n_shards} shards")
        upload_future = None

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as proc_pool, \
             ThreadPoolExecutor(max_workers=1) as upload_pool:

            for shard_idx in range(n_shards):
                key = f"{split_name}/{shard_idx}"
                if key in completed:
                    log.info(f"  shard {shard_idx+1}/{n_shards} already done, skipping")
                    continue

                batch = entries[shard_idx * SHARD_SIZE: (shard_idx + 1) * SHARD_SIZE]
                # group by WAV so each file is read once per worker
                wav_groups: dict[Path, list[dict]] = {}
                for e in batch:
                    wav_groups.setdefault(e["wav_path"], []).append(e)
                groups = list(wav_groups.values())
                rows = [r for results in proc_pool.map(_process_wav_group, groups, chunksize=10) for r in results]

                if not rows:
                    log.warning(f"  shard {shard_idx+1}/{n_shards} produced 0 rows, skipping")
                    continue

                ds = rows_to_dataset(rows)
                shard_name = f"{split_name}-{shard_idx:05d}-of-{n_shards:05d}.parquet"
                shard_path = LOG_DIR / shard_name
                ds.to_parquet(str(shard_path))
                mb = shard_path.stat().st_size / 1024 / 1024
                log.info(f"  shard {shard_idx+1}/{n_shards} → {shard_name} ({mb:.1f} MB, {len(rows)} rows)")

                if upload_future is not None:
                    upload_future.result()
                upload_future = upload_pool.submit(
                    upload_and_delete, shard_path, f"data/{shard_name}",
                    split_name, shard_idx, len(rows), mb
                )

            if upload_future is not None:
                upload_future.result()

    # --- Push dataset card ---
    card = DatasetCard("""\
---
language:
- fil
task_categories:
- automatic-speech-recognition
tags:
- filipino
- tagalog
- speech
- tts
- whisper
citation: "@article{sagumdevelopment,\n  title={DEVELOPMENT OF A FILIPINO SPEECH CORPUS},\n  author={Sagum, Ramil}\n}"
---

# Filipino Speech Corpus

Sentence-level segments from the Filipino Speech Corpus (FSC), stored as a
Hugging Face Parquet dataset with raw 16kHz mono audio.

## Citation

If you use this dataset, please cite the original corpus:
[Development of a Filipino Speech Corpus](http://www.wins.or.kr/DataPool/Board/xxxx/18xx/1812/DEVELOPMENT%20OF%20A%20FILIPINO%20SPEECH%20CORPUS.pdf)

```
@article{sagumdevelopment,
  title={DEVELOPMENT OF A FILIPINO SPEECH CORPUS},
  author={Sagum, Ramil}
}
```

## Usage

**Whisper / ASR:**
```python
from datasets import load_dataset, Audio
ds = load_dataset("sapinsapin/filipinospeechcorpus")
ds = ds.filter(lambda x: x["num_words"] >= 3 and x["duration"] >= 1.5)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
```

**TTS (LJSpeech-compatible, 22050Hz):**
```python
ds = ds.filter(lambda x: x["speech_type"] == "read" and 1.0 <= x["duration"] <= 10.0)
ds = ds.cast_column("audio", Audio(sampling_rate=22050))
```

## Schema

| Field | Type | Description |
|---|---|---|
| `audio` | `Audio(16000)` | 16kHz mono WAV segment |
| `sentence` | `str` | Transcription |
| `duration` | `float` | Segment duration in seconds |
| `num_words` | `int` | Word count |
| `speaker_id` | `str` | Speaker identifier |
| `gender` | `str` | `male` / `female` |
| `age_group` | `str` | Age range e.g. `20-27` |
| `speech_type` | `str` | `read` / `spontaneous` / `machine` |
| `source_file` | `str` | Original TRS stem |

## Code

Processing code is available at https://github.com/sapinsapin/pretraining

## Splits

| Split | Rows |
|---|---|
| `train` | 90% |
| `test` | 10% |
""")
    card.push_to_hub(HF_REPO, token=HF_TOKEN)
    log.info("Done.")


if __name__ == "__main__":
    main()
