"""
Raw livestream sources → archival Hub dataset (`sapinsapin/halo-livestream-raw`)

This is the *input* side of the livestream pipeline. `process_livestream.py`
turns recordings into segmented ASR/TTS parquet; this publishes the recordings
themselves, so every derived artifact stays reproducible from a fixed input.

Usage:
  python push_livestream_raw.py --dry-run          # stage + report, no upload
  python push_livestream_raw.py                    # stage + upload new files
  python push_livestream_raw.py --card-only        # refresh the dataset card
  python push_livestream_raw.py --file-id <ID>     # one recording

Sources are discovered under LIVESTREAM_RAW_DIR (falling back to
LIVESTREAM_DIR) in any of three layouts — a {id}/ directory, loose
{id}.json + {id}.<ext> pairs, or a .zip. Audio is normalized by
halolib.raw: already-compressed streams are copied verbatim, only PCM is
encoded (to FLAC). Video, if any, is dropped.

Built for long recordings arriving over time:
  - staging is idempotent, so re-runs skip work already done
  - uploads are incremental — a recording already on the Hub with a matching
    size is not re-sent
  - past --large-threshold the transfer switches to upload_large_folder,
    which chunks, parallelizes and resumes after an interruption

Repo layout on the Hub:
  audio/{file_id}.{m4a|mp3|opus|flac}   archival audio track
  transcripts/{file_id}.json            operator transcript, as delivered
  index.jsonl                           one row per recording (hashes, durations)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import DatasetCard, HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from halolib import raw
from halolib.sources import livestream as ls

load_dotenv(Path(__file__).parent / ".env")

RAW_DIR = Path(os.environ.get("LIVESTREAM_RAW_DIR") or os.environ["LIVESTREAM_DIR"])
STAGE_DIR = Path(os.environ.get("LIVESTREAM_RAW_STAGE_DIR") or RAW_DIR.parent / "raw_staging")
REPO = os.environ.get("LIVESTREAM_RAW_HF_REPO", "sapinsapin/halo-livestream-raw")
PROCESSED_REPO = os.environ.get("LIVESTREAM_HF_REPO", "sapinsapin/halo-livestream")

# Above this staged size, use the chunked/resumable uploader.
LARGE_THRESHOLD_BYTES = 1 << 30  # 1 GiB

# Gating applied when this script creates the repo. Raw recordings must never
# be reachable ungated, so a new repo starts closed and is widened by hand.
NEW_REPO_GATING = "manual"

log = logging.getLogger("livestream-raw")


# --------------------------------------------------------------------- staging

def stage(file_id_filter: str | None) -> list[raw.RawRecord]:
    raw.require_ffmpeg()

    if not RAW_DIR.exists():
        raise SystemExit(f"LIVESTREAM_RAW_DIR does not exist: {RAW_DIR}")

    pairs = ls.find_pairs(RAW_DIR)
    if file_id_filter:
        pairs = [p for p in pairs if p[0] == file_id_filter]
    if not pairs:
        raise SystemExit(f"No (json, audio) sources found under {RAW_DIR}")

    records = []
    for file_id, json_path, audio_path in pairs:
        try:
            record = raw.build_record(file_id, json_path, audio_path, STAGE_DIR)
        except raw.RawError as err:
            log.error("  [skip] %s: %s", file_id, err)
            continue

        log.info(
            "  %s  %s  %s %dHz/%dch  %s  %s",
            file_id[:16],
            raw.human_duration(record.probe.duration),
            record.probe.codec,
            record.probe.sample_rate,
            record.probe.channels,
            raw.human_bytes(record.audio_bytes),
            "stream-copied" if record.stream_copied else "encoded flac",
        )
        records.append(record)

    return records



def current_gating(api: HfApi, token: str | None) -> str | None:
    """Whatever the Hub currently has, or None if the repo is ungated/absent."""
    try:
        gated = api.dataset_info(REPO, token=token).gated
    except RepositoryNotFoundError:
        return None
    return gated if isinstance(gated, str) else None


# ---------------------------------------------------------------------- upload

def remote_sizes(api: HfApi, token: str) -> dict[str, int]:
    """Path → size for what is already on the Hub. Empty if the repo is new."""
    try:
        info = api.dataset_info(REPO, files_metadata=True, token=token)
    except RepositoryNotFoundError:
        return {}
    return {s.rfilename: (s.size or 0) for s in (info.siblings or [])}


def pending_uploads(records: list[raw.RawRecord], remote: dict[str, int]) -> list[str]:
    """Repo-relative paths not yet present at the expected size."""
    pending = []
    for record in records:
        audio_key = f"audio/{record.audio_name}"
        if remote.get(audio_key) != record.audio_bytes:
            pending.append(audio_key)
        transcript_key = f"transcripts/{record.transcript_name}"
        staged = (STAGE_DIR / transcript_key).stat().st_size
        if remote.get(transcript_key) != staged:
            pending.append(transcript_key)
    return pending


def upload(api: HfApi, token: str, records: list[raw.RawRecord], staged_bytes: int) -> None:
    if staged_bytes >= LARGE_THRESHOLD_BYTES:
        log.info("Uploading %s via chunked resumable transfer", raw.human_bytes(staged_bytes))
        api.upload_large_folder(
            repo_id=REPO,
            folder_path=str(STAGE_DIR),
            repo_type="dataset",
            ignore_patterns=["*.part.*", ".cache/*"],
        )
        return

    log.info("Uploading %s", raw.human_bytes(staged_bytes))
    api.upload_folder(
        repo_id=REPO,
        folder_path=str(STAGE_DIR),
        repo_type="dataset",
        token=token,
        ignore_patterns=["*.part.*", ".cache/*"],
        commit_message=f"Add {len(records)} raw livestream recording(s)",
    )


# ------------------------------------------------------------------------ card

def build_card(records: list[raw.RawRecord]) -> str:
    total_seconds = sum(r.probe.duration for r in records)
    total_bytes = sum(r.audio_bytes for r in records)
    speakers = sum(r.meta.get("speaker_count") or 0 for r in records)
    languages = sorted({r.meta.get("primary_language", "") for r in records} - {""})

    rows = "\n".join(
        f"| `{r.file_id[:16]}…` | {raw.human_duration(r.probe.duration)} | "
        f"{r.probe.codec} {r.probe.sample_rate} Hz / {r.probe.channels} ch | "
        f"{raw.human_bytes(r.audio_bytes)} | {r.meta.get('speaker_count') or '—'} |"
        for r in sorted(records, key=lambda r: r.file_id)
    )

    return f"""---
language:
- tl
- fil
- en
pretty_name: halo-livestream-raw (unsegmented Taglish livestream sources)
size_categories:
- n<1K
task_categories:
- automatic-speech-recognition
- text-to-speech
multilinguality:
- multilingual
tags:
- filipino
- tagalog
- taglish
- code-switching
- spontaneous-speech
- livestream
- conversational
- philippines
- raw
- unsegmented
extra_gated_prompt: >-
  These are unsegmented livestream recordings of identifiable people speaking
  conversationally. Access is granted for speech research and dataset
  construction. By requesting access you agree not to redistribute the
  recordings, not to attempt to identify or contact the speakers, and not to
  use the audio for voice cloning or impersonation of the speakers.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
  I agree to the terms above: checkbox
---

# halo-livestream-raw

**Unsegmented source recordings behind [`{PROCESSED_REPO}`](https://huggingface.co/datasets/{PROCESSED_REPO}) — the archival input to the pipeline, not a training set.**

<div align="center">

**{len(records)} recording(s) · {raw.human_duration(total_seconds)} · {raw.human_bytes(total_bytes)}**

[![Pipeline](https://img.shields.io/badge/pipeline-github-black)](https://github.com/sapinsapin/halohalo)
[![Processed](https://img.shields.io/badge/processed-{PROCESSED_REPO.split('/')[-1]}-yellow)](https://huggingface.co/datasets/{PROCESSED_REPO})

</div>

> ### 🔒 Gated on purpose
>
> Full-length conversation between named, identifiable speakers is a very
> different privacy proposition from the short segments in the processed
> dataset, so access here is gated: request it and agree to the terms above.
> If what you want is segmented, quality-scored audio for training, reach for
> [`{PROCESSED_REPO}`](https://huggingface.co/datasets/{PROCESSED_REPO}) instead.

## What this is

`{PROCESSED_REPO}` publishes ~30-second segments with forced-alignment
confidence, round-trip CER, SNR and overlap flags. Those segments are derived.
This repository holds what they were derived *from*, so results stay
reproducible and future pipeline versions can be re-run over the same input
without re-collecting anything.

## Contents

```
audio/{{file_id}}.{{m4a|mp3|opus|flac}}   archival audio track
transcripts/{{file_id}}.json              operator transcript, as delivered
index.jsonl                               one row per recording
```

| Recording | Duration | Audio | Size | Speakers |
|---|---|---|---|---|
{rows}

Languages present: {', '.join(languages) or '—'} · speaker entries: {speakers}

## How the audio is stored

Audio is kept as close to as-delivered as possible:

- **Already-compressed sources (AAC/MP3/Opus) are stream-copied**, never
  re-encoded. Transcoding lossy audio to FLAC cannot recover what the encoder
  discarded, and it inflates size roughly tenfold — on the seed recording,
  9.0 MB AAC becomes 113 MB FLAC with a *different* decoded checksum. The
  copied stream round-trips bit-exactly.
- **Only uncompressed PCM is encoded**, to FLAC, where compression is lossless
  and also sidesteps the 4 GB WAV ceiling that multi-hour streams hit.
- **Video tracks are dropped.** The pipeline never reads them, and they carry
  the most personal data.

`index.jsonl` records `audio_sha256` for every track so integrity is checkable
after download.

## Transcript schema

```json
{{
  "metadata": {{
    "file_properties":  {{ "duration": "00:26:55", "audio_specifications": {{...}} }},
    "linguistic_profile": {{ "primary_language": "Taglish", "content_theme": "..." }},
    "speaker_profile":  {{ "speaker_count": 3, "speakers": [{{"speaker_id": "Speaker 1", ...}}] }}
  }},
  "transcription": [
    {{ "time_range": "02:25 - 02:33",
       "dialogue": [{{"s": "Speaker 1", "txt": "..."}}] }}
  ]
}}
```

Timings are **block-level only** — a block can bundle several turns with no
per-turn timestamps. The pipeline's `parse` stage interpolates a baseline from
character counts and the `align` stage replaces it with forced alignment. Treat
the time ranges here as approximate.

Reported `file_properties` come from the operator and may not match the
container; trust `index.jsonl`, which is probed from the actual file.

## Rebuilding the processed dataset

```bash
huggingface-cli download {REPO} --repo-type dataset --local-dir raw/
export LIVESTREAM_DIR=raw/
python process_livestream.py --stages parse,align,qc,export --push
```

Discovery accepts this layout directly, so a downloaded snapshot feeds straight
back into the pipeline.

## Limitations

- Small. This is a seed archive, not a corpus.
- Diarization and transcription are operator-supplied, not verified here.
- Speaker labels are per-recording; `Speaker 1` in two files is not the same
  person. The pipeline namespaces them at parse time.

## License and consent

Recordings were collected from public livestreams. No consent was obtained from
speakers for model training specifically — treat this as a research artifact
and honour the gated terms. If you are a speaker in one of these recordings and
want it removed, open a discussion on this repository.
"""


# ------------------------------------------------------------------------ main

def main() -> int:
    global REPO

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dry-run", action="store_true", help="stage and report, upload nothing")
    parser.add_argument("--card-only", action="store_true", help="only refresh the dataset card")
    parser.add_argument("--file-id", help="restrict to one recording id")
    parser.add_argument("--repo", default=REPO, help=f"target dataset repo (default {REPO})")
    parser.add_argument(
        "--gated",
        default="keep",
        choices=["keep", "manual", "auto", "off"],
        help=(
            "Hub access gating. Default 'keep' leaves an existing repo's setting "
            f"alone (a new repo is created with '{NEW_REPO_GATING}'), so a routine "
            "upload never overrides a change made in the dashboard."
        ),
    )
    parser.add_argument("--force", action="store_true", help="re-upload even if sizes match")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    REPO = args.repo

    token = os.environ.get("HF_TOKEN")  # falls back to the CLI login cache
    api = HfApi(token=token)

    log.info("Source dir : %s", RAW_DIR)
    log.info("Staging dir: %s", STAGE_DIR)
    log.info("Target repo: %s (gated=%s)", REPO, args.gated)
    log.info("")

    records = stage(args.file_id)
    if not records:
        log.error("Nothing staged.")
        return 1

    index_path = raw.write_index(records, STAGE_DIR)
    staged_bytes = sum(r.audio_bytes for r in records)
    log.info("")
    log.info(
        "Staged %d recording(s), %s audio, %s total",
        len(records),
        raw.human_bytes(staged_bytes),
        raw.human_duration(sum(r.probe.duration for r in records)),
    )
    log.info("Index: %s", index_path)

    if args.dry_run:
        log.info("\n--dry-run: nothing uploaded.")
        return 0

    # Create private first. A gated repo is publicly listed, so creating it
    # public up front would leave the files ungated until the card lands.
    try:
        api.dataset_info(REPO, token=token)
        created = False
    except RepositoryNotFoundError:
        api.create_repo(REPO, repo_type="dataset", private=True, token=token)
        log.info("Created %s (private)", REPO)
        created = True

    if not args.card_only:
        remote = {} if args.force else remote_sizes(api, token)
        pending = pending_uploads(records, remote)
        if pending:
            log.info("Uploading %d file(s): %s", len(pending), ", ".join(pending[:6]) +
                     ("…" if len(pending) > 6 else ""))
            upload(api, token, records, staged_bytes)
        else:
            log.info("All recordings already on the Hub at matching sizes — skipping upload.")
            api.upload_file(
                path_or_fileobj=str(index_path),
                path_in_repo="index.jsonl",
                repo_id=REPO,
                repo_type="dataset",
                token=token,
                commit_message="Refresh index",
            )

    DatasetCard(build_card(records)).push_to_hub(REPO, repo_type="dataset", token=token)
    log.info("Card pushed.")

    # Gating is a policy decision that outlives any one upload. Only a new repo
    # gets a gate applied by default; an existing one keeps whatever it has
    # unless --gated says otherwise, so routine uploads can't quietly widen or
    # narrow access that someone set deliberately.
    gating = NEW_REPO_GATING if (args.gated == "keep" and created) else args.gated

    if gating != "keep":
        api.update_repo_settings(
            REPO, repo_type="dataset", gated=(False if gating == "off" else gating), token=token
        )
        log.info("Gating set to %s.", gating)
    else:
        log.info("Gating left as-is (%s).", current_gating(api, token) or "ungated")

    if created:
        # Safe now: the gate is in place before anything becomes listable.
        api.update_repo_settings(REPO, repo_type="dataset", private=False, token=token)
        log.info("Repo made public (gated).")

    log.info("\nhttps://huggingface.co/datasets/%s", REPO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
