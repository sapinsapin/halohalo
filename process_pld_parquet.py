"""
UP-DSP Philippine Language Dataset (PLD) → Hugging Face Parquet dataset

Mirrors process_fsc_parquet.py: shard utterances into parquet files carrying
raw audio bytes, upload each shard as it is produced, and record progress so an
interrupted run resumes where it stopped.

PLD needs no segmentation step — the corpus ships one WAV per prompt with the
text inline in the session .log, so the pipeline is index → shard → upload.
Audio is already 16kHz mono PCM_16, so shards embed the original file bytes
unchanged; anything off-spec is decoded and resampled.

Usage:
  python process_pld_parquet.py                    # all languages, push to Hub
  python process_pld_parquet.py --languages BIK    # one language
  python process_pld_parquet.py --no-push          # write shards locally only
  python process_pld_parquet.py --card-only        # just refresh the card

Downstream:
  ds = load_dataset("sapinsapin/pld")
  ds = ds.filter(lambda x: not x["text_is_prompt"])          # drop prompt-only rows
  ds = ds.filter(lambda x: x["language"] == "bcl")           # pick a language
"""

import argparse
import io
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset
from dotenv import load_dotenv
from huggingface_hub import DatasetCard, HfApi

from halolib.pld import index_corpus

load_dotenv(Path(__file__).parent / ".env")

PLD_DIR   = Path(os.environ.get(
    "PLD_DIR", "/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD"))
HF_REPO   = os.environ.get("HF_PLD_REPO", "sapinsapin/pld")
HF_TOKEN  = os.environ.get("HF_TOKEN")

TARGET_SR    = 16000
SPLIT_RATIO  = 0.9
RANDOM_SEED  = 42
SHARD_SIZE   = 1500          # ~150-200MB/shard at PLD's ~5s mean duration
NUM_WORKERS  = os.cpu_count() or 4

# keep scratch + resume log on D: — C: has no headroom for 40GB of shards
WORK_DIR   = Path(os.environ.get("PLD_WORK_DIR", "/mnt/d/halohalo/pld_shards"))
RESUME_LOG = WORK_DIR / "progress.jsonl"
RUN_LOG    = WORK_DIR / "run.log"

# columns written to parquet, in order (audio is added separately)
META_COLS = [
    "sentence", "num_words", "language", "language_name", "corpus_language",
    "speech_type", "prompt_category", "prompt_source", "text_is_prompt",
    "speaker_id", "gender", "age", "speaker_dialect", "mother_dialect",
    "father_dialect", "profession", "session_id", "session_environment",
    "source_file",
]


def setup_logging():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(RUN_LOG), logging.StreamHandler()],
    )


def load_completed_shards() -> set[str]:
    if not RESUME_LOG.exists():
        return set()
    done = set()
    with RESUME_LOG.open() as f:
        for line in f:
            try:
                e = json.loads(line)
                done.add(f"{e['split']}/{e['shard_idx']}")
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def log_shard_done(split: str, shard_idx: int, name: str, rows: int, mb: float,
                   stats: dict):
    with RESUME_LOG.open("a") as f:
        f.write(json.dumps({
            "split": split, "shard_idx": shard_idx, "shard_name": name,
            "rows": rows, "mb": round(mb, 2), "stats": stats,
            "ts": datetime.now(timezone.utc).isoformat(),
        }) + "\n")


def load_prior_stats() -> tuple[dict, dict[str, int]]:
    """Replay the resume log so a restarted run can still build a card that
    describes the whole dataset, not just the shards this process wrote."""
    acc = {"languages": {}, "types": {}, "speakers": set()}
    rows = {"train": 0, "test": 0}
    if not RESUME_LOG.exists():
        return acc, rows
    with RESUME_LOG.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[e["split"]] = rows.get(e["split"], 0) + e.get("rows", 0)
            if e.get("stats"):
                merge_stats(acc, e["stats"])
    return acc, rows


def _read_audio_bytes(path: Path, fmt: str) -> tuple[bytes, float] | None:
    """Return (encoded_bytes, duration), resampling to 16kHz mono if needed.

    WAV files that are already on-spec are passed through byte-for-byte; FLAC
    always re-encodes (lossless, roughly half the size — worth the CPU when the
    corpus is tens of GB and has to cross a home uplink).
    """
    try:
        info = sf.info(path)
    except Exception as e:
        logging.warning(f"  unreadable {path.name}: {e}")
        return None

    on_spec = info.samplerate == TARGET_SR and info.channels == 1
    if on_spec and fmt == "wav":
        return path.read_bytes(), info.duration

    try:
        data, sr = sf.read(path, always_2d=False)
    except Exception as e:
        logging.warning(f"  unreadable {path.name}: {e}")
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
    if len(data) == 0:
        return None

    buf = io.BytesIO()
    sf.write(buf, data, TARGET_SR, format=fmt.upper(),
             subtype="PCM_16" if fmt == "wav" else None)
    return buf.getvalue(), len(data) / TARGET_SR


def _process_chunk(task: tuple[list[dict], str]) -> list[dict]:
    """Worker: turn index entries into parquet rows."""
    entries, fmt = task
    rows = []
    for e in entries:
        got = _read_audio_bytes(Path(e["wav_path"]), fmt)
        if got is None:
            continue
        audio_bytes, duration = got
        row = {"audio": {"bytes": audio_bytes, "path": None},
               "duration": round(duration, 3)}
        row.update({c: e[c] for c in META_COLS})
        rows.append(row)
    return rows


def shard_stats(rows: list[dict]) -> dict:
    """Compact per-shard aggregates, persisted so --resume can rebuild the
    dataset card without re-reading audio that was already uploaded."""
    langs, types, speakers = {}, {}, set()
    for r in rows:
        c, n, d = r["language"], r["language_name"], r["duration"]
        lang = langs.setdefault(c, {"name": n, "n": 0, "sec": 0.0})
        lang["n"] += 1
        lang["sec"] += d
        t = types.setdefault(r["speech_type"], {"n": 0, "sec": 0.0})
        t["n"] += 1
        t["sec"] += d
        speakers.add(r["speaker_id"])
    return {"languages": langs, "types": types, "speakers": sorted(speakers)}


def merge_stats(acc: dict, part: dict):
    for c, v in part["languages"].items():
        a = acc["languages"].setdefault(c, {"name": v["name"], "n": 0, "sec": 0.0})
        a["n"] += v["n"]
        a["sec"] += v["sec"]
    for t, v in part["types"].items():
        a = acc["types"].setdefault(t, {"n": 0, "sec": 0.0})
        a["n"] += v["n"]
        a["sec"] += v["sec"]
    acc["speakers"].update(part["speakers"])


def rows_to_dataset(rows: list[dict]) -> Dataset:
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    return Dataset.from_dict(cols).cast_column("audio", Audio(sampling_rate=TARGET_SR))


def build_card(stats: dict) -> DatasetCard:
    lang_yaml = "\n".join(f"- {c}" for c in stats["languages"])
    lang_rows = "\n".join(
        f"| `{c}` | {n} | {u:,} | {h:,.1f} |"
        for c, n, u, h in stats["per_language"])
    type_rows = "\n".join(
        f"| `{t}` | {u:,} | {h:,.1f} | {m:.1f}s |" for t, u, h, m in stats["per_type"])

    return DatasetCard(f"""\
---
language:
{lang_yaml}
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- philippines
- philippine-languages
- low-resource
- speech
- multilingual
size_categories:
- 100K<n<1M
---

# Philippine Language Dataset (PLD)

Utterance-level recordings from the University of the Philippines Digital
Signal Processing Laboratory's Philippine Language Dataset, packaged as a
Hugging Face Parquet dataset with 16kHz mono audio.

Every row is one prompted recording: the corpus ships pre-segmented WAVs with
the prompt text stored inline in each session log, so no forced alignment or
segmentation was applied. Text is normalized only for whitespace and unicode
punctuation — orthography and dialectal spelling are preserved as recorded.

**{stats['utterances']:,} utterances · {stats['hours']:,.1f} hours · \
{stats['speakers']:,} speakers · {len(stats['languages'])} languages**

## ⚠️ Read before training: `text_is_prompt`

Rows where `speech_type == "spontaneous"` do **not** carry a transcript. The
session logs store the *elicitation question* that was put to the speaker
(e.g. "Saen an dream destination mo?"), while the audio is 20-90 seconds of
their free-speech answer. The same question text repeats verbatim across
different speakers. These rows are flagged `text_is_prompt = true` and must be
excluded from any supervised (audio, text) training:

```python
ds = ds.filter(lambda x: not x["text_is_prompt"])
```

They remain in the dataset because the audio is genuine spontaneous speech,
useful for self-supervised pretraining, speaker modeling, or re-transcription.

## Languages

| Code | Language | Utterances | Hours |
|---|---|---|---|
{lang_rows}

English word and sentence lists (`EngW.txt`, `EngSen.txt`) were read by the
same speakers. Those rows are labeled `language = "eng"`, with
`corpus_language` retaining the Philippine language collection they came from,
so per-language filters stay clean.

## Speech types

| Type | Utterances | Hours | Mean duration |
|---|---|---|---|
{type_rows}

- `read` — full prompted sentences (news, medical, literature, essays); the
  material best suited to TTS
- `isolated` — single words and short phrases from word lists
- `digits` — spoken digit strings
- `spontaneous` — free speech; **prompt-only text**, see the warning above

## Usage

**ASR finetuning (Whisper):**
```python
from datasets import load_dataset, Audio
ds = load_dataset("{HF_REPO}")
ds = ds.filter(lambda x: not x["text_is_prompt"] and 0.3 <= x["duration"] <= 30.0)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
```

**TTS finetuning (read speech only):**
```python
ds = ds.filter(lambda x: x["speech_type"] == "read"
                         and 1.0 <= x["duration"] <= 15.0
                         and x["num_words"] >= 3)
```

**Single language:**
```python
ds = ds.filter(lambda x: x["language"] == "bcl")   # Bikol
```

## Schema

| Field | Type | Description |
|---|---|---|
| `audio` | `Audio(16000)` | 16kHz mono WAV |
| `sentence` | `str` | Prompt text read by the speaker (see `text_is_prompt`) |
| `duration` | `float` | Seconds |
| `num_words` | `int` | Whitespace word count |
| `language` | `str` | ISO 639-3 of the spoken content (`eng` for English lists) |
| `language_name` | `str` | Human-readable language name |
| `corpus_language` | `str` | Language collection the session belongs to |
| `speech_type` | `str` | `read` / `isolated` / `digits` / `spontaneous` |
| `prompt_category` | `str` | Prompt list, e.g. `News`, `Medical`, `BodyParts` |
| `prompt_source` | `str` | Original prompt filename |
| `text_is_prompt` | `bool` | `true` when text is an elicitation question, not a transcript |
| `speaker_id` | `str` | Language-namespaced speaker key, e.g. `BIK_0800` |
| `gender` | `str` | `male` / `female` / `unknown` |
| `age` | `int` | Speaker age, `-1` when unrecorded |
| `speaker_dialect` | `str` | Self-reported dialect |
| `mother_dialect` / `father_dialect` | `str` | Parents' dialects |
| `profession` | `str` | Self-reported profession |
| `session_id` | `str` | Recording session identifier |
| `session_environment` | `str` | Recording environment note |
| `source_file` | `str` | Original WAV stem |

## Splits

| Split | Rows |
|---|---|
| `train` | {stats['train']:,} |
| `test` | {stats['test']:,} |

Split is a seeded random 90/10 over utterances, so speakers appear in both
splits. For speaker-disjoint evaluation, re-split on `speaker_id`.

## Source

Collected by the UP Diliman Digital Signal Processing Laboratory. Please cite
the original corpus authors when using this data.

## Code

Processing code: https://github.com/sapinsapin/pretraining
""")


def compute_stats(acc: dict, rows: dict[str, int]) -> dict:
    """Turn accumulated shard aggregates into the numbers the card prints."""
    total_n = sum(v["n"] for v in acc["languages"].values())
    total_sec = sum(v["sec"] for v in acc["languages"].values())
    return {
        "utterances": total_n,
        "hours": total_sec / 3600,
        "speakers": len(acc["speakers"]),
        "languages": sorted(acc["languages"]),
        "per_language": sorted(
            ((c, v["name"], v["n"], v["sec"] / 3600)
             for c, v in acc["languages"].items()), key=lambda r: -r[3]),
        "per_type": sorted(
            ((t, v["n"], v["sec"] / 3600, v["sec"] / max(v["n"], 1))
             for t, v in acc["types"].items()), key=lambda r: -r[2]),
        "train": rows.get("train", 0), "test": rows.get("test", 0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--languages", help="comma-separated dir names, e.g. BIK,CEB")
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--no-push", action="store_true", help="write shards locally, keep them")
    ap.add_argument("--card-only", action="store_true", help="only refresh the dataset card")
    ap.add_argument("--audio-format", choices=["flac", "wav"], default="flac",
                    help="flac halves the upload/storage footprint, losslessly")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public; default is private, since PLD "
                         "is a third-party corpus whose redistribution terms "
                         "should be confirmed before it is published")
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)
    log.info(f"Corpus: {PLD_DIR} | repo: {HF_REPO} | workers: {args.workers}")

    langs = args.languages.split(",") if args.languages else None

    log.info("Pass 1: indexing session logs...")
    entries, counts = index_corpus(PLD_DIR, langs)
    log.info(f"  sessions {counts['sessions']:,} | utterances {counts['rows']:,} "
             f"| missing wav {counts['missing_wav']:,} "
             f"| empty sessions {counts['empty_sessions']:,}")
    if not entries:
        log.error("No entries found — is the corpus extracted?")
        return

    # durations come out of the sharding pass itself — a separate header pass
    # over hundreds of thousands of files on a 9p mount costs more than it
    # returns, since every row is decoded downstream anyway
    random.seed(RANDOM_SEED)
    order = list(range(len(entries)))
    random.shuffle(order)
    cut = int(len(order) * SPLIT_RATIO)
    splits = {"train": [entries[i] for i in order[:cut]],
              "test":  [entries[i] for i in order[cut:]]}

    acc, row_counts = load_prior_stats()

    api = HfApi(token=HF_TOKEN)
    if args.card_only:
        build_card(compute_stats(acc, row_counts)).push_to_hub(HF_REPO, token=HF_TOKEN)
        log.info("Card pushed.")
        return

    completed = load_completed_shards()
    if completed:
        log.info(f"Resuming — {len(completed)} shards already uploaded")

    if not args.no_push:
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True,
                        private=not args.public)
        log.info(f"  repo {HF_REPO} ({'public' if args.public else 'private'})")

    def upload_and_delete(path: Path, repo_path: str, split: str,
                          idx: int, rows: int, mb: float, stats: dict):
        for attempt in range(4):
            try:
                HfApi(token=HF_TOKEN).upload_file(
                    path_or_fileobj=str(path), path_in_repo=repo_path,
                    repo_id=HF_REPO, repo_type="dataset")
                break
            except Exception as e:
                if attempt == 3:
                    raise
                log.warning(f"  upload retry {attempt+1} for {path.name}: {e}")
                time.sleep(5 * (attempt + 1))
        path.unlink()
        log_shard_done(split, idx, path.name, rows, mb, stats)
        log.info(f"  uploaded {path.name} ({mb:.1f} MB, {rows} rows)")

    started = time.time()
    done_rows = 0

    for split, split_entries in splits.items():
        n_shards = (len(split_entries) + args.shard_size - 1) // args.shard_size
        log.info(f"\nPass 2: {split} — {len(split_entries):,} utterances, {n_shards} shards")
        pending = None

        with ProcessPoolExecutor(max_workers=args.workers) as proc_pool, \
             ThreadPoolExecutor(max_workers=1) as upload_pool:

            for idx in range(n_shards):
                if f"{split}/{idx}" in completed:
                    continue

                batch = split_entries[idx * args.shard_size:(idx + 1) * args.shard_size]
                chunks = [(batch[i::args.workers], args.audio_format)
                          for i in range(args.workers)]
                rows = [r for out in proc_pool.map(_process_chunk, chunks) for r in out]
                if not rows:
                    log.warning(f"  shard {idx+1}/{n_shards} produced 0 rows")
                    continue

                name = f"{split}-{idx:05d}-of-{n_shards:05d}.parquet"
                path = WORK_DIR / name
                rows_to_dataset(rows).to_parquet(str(path))
                mb = path.stat().st_size / 1024 / 1024
                st = shard_stats(rows)
                merge_stats(acc, st)
                row_counts[split] = row_counts.get(split, 0) + len(rows)

                done_rows += len(rows)
                rate = done_rows / max(time.time() - started, 1)
                log.info(f"  shard {idx+1}/{n_shards} {split} → {mb:.1f} MB, "
                         f"{len(rows)} rows | {rate:.0f} rows/s")

                if args.no_push:
                    log_shard_done(split, idx, name, len(rows), mb, st)
                    continue
                if pending is not None:
                    pending.result()
                pending = upload_pool.submit(
                    upload_and_delete, path, f"data/{name}", split, idx,
                    len(rows), mb, st)

            if pending is not None:
                pending.result()

    stats = compute_stats(acc, row_counts)
    log.info(f"\nTotals: {stats['utterances']:,} utterances | {stats['hours']:,.1f} h "
             f"| {stats['speakers']:,} speakers | {len(stats['languages'])} languages")

    if not args.no_push:
        build_card(stats).push_to_hub(HF_REPO, token=HF_TOKEN)
        log.info(f"Done — https://huggingface.co/datasets/{HF_REPO}")
    else:
        log.info(f"Done — shards in {WORK_DIR}")


if __name__ == "__main__":
    main()
