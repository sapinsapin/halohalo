"""
Raw-source normalization for the archival Hub dataset (`halo-livestream-raw`).

The raw dataset is the *input* side of the pipeline: the operator-supplied
transcript JSON plus the recording's audio track, kept as close to as-delivered
as possible so every downstream artifact can be rebuilt from it.

Normalization rule — the important one for long streams:

    If the source audio is already in a compressed codec (AAC, MP3, Opus...),
    the stream is COPIED, not re-encoded. Transcoding lossy audio to FLAC
    "for archival" is a common and expensive mistake: it cannot recover
    information the lossy encoder already discarded, it stores the decoder's
    output rather than the delivered bytes, and it inflates size by ~10x.
    Measured on the seed recording: 9.0 MB AAC -> 113 MB FLAC, with a
    *different* decoded-PCM checksum. Stream-copy round-trips bit-exactly.

    Only genuinely uncompressed sources (PCM WAV/AIFF) are encoded, to FLAC,
    where the compression is lossless and roughly halves the size. That also
    dodges the 4 GB RIFF/WAV size ceiling, which a multi-hour stream will hit.

Video, when present, is dropped: the pipeline never reads it, and it is the
part of a livestream recording most loaded with personal data.

Requires ffmpeg/ffprobe on PATH (already a documented dependency, see
requirements.txt).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Codecs we copy verbatim, mapped to the container we drop them into.
# Anything not listed is treated as uncompressed and encoded to FLAC.
COPY_CONTAINERS = {
    "aac": ".m4a",
    "alac": ".m4a",
    "mp3": ".mp3",
    "opus": ".opus",
    "vorbis": ".ogg",
    "flac": ".flac",
}

FLAC_COMPRESSION_LEVEL = "8"
HASH_CHUNK = 1 << 20


class RawError(RuntimeError):
    """Normalization failed for one recording."""


@dataclass
class AudioProbe:
    codec: str
    sample_rate: int
    channels: int
    duration: float
    size_bytes: int
    has_video: bool
    container: str

    @property
    def is_compressed(self) -> bool:
        return self.codec in COPY_CONTAINERS


@dataclass
class RawRecord:
    """One normalized recording, ready to stage into the upload folder."""

    file_id: str
    audio_name: str
    transcript_name: str
    probe: AudioProbe
    audio_sha256: str
    audio_bytes: int
    transcript_sha256: str
    stream_copied: bool
    source_name: str
    meta: dict = field(default_factory=dict)

    def index_row(self) -> dict:
        return {
            "file_id": self.file_id,
            "audio": f"audio/{self.audio_name}",
            "transcript": f"transcripts/{self.transcript_name}",
            "duration_seconds": round(self.probe.duration, 3),
            "codec": self.probe.codec,
            "sample_rate": self.probe.sample_rate,
            "channels": self.probe.channels,
            "audio_bytes": self.audio_bytes,
            "audio_sha256": self.audio_sha256,
            "transcript_sha256": self.transcript_sha256,
            "stream_copied": self.stream_copied,
            "source_had_video": self.probe.has_video,
            "source_file": self.source_name,
            **self.meta,
        }


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RawError(
            f"{', '.join(missing)} not found on PATH. Install ffmpeg "
            "(macOS: brew install ffmpeg / Debian: sudo apt install ffmpeg)."
        )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_audio(path: Path) -> AudioProbe:
    """Inspect the first audio stream and note whether video rides along."""
    result = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,sample_rate,channels",
        "-show_entries", "format=duration,size,format_name",
        "-of", "json", str(path),
    ])
    if result.returncode != 0:
        raise RawError(f"ffprobe failed on {path.name}: {result.stderr.strip()[-300:]}")

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    fmt = payload.get("format", {})

    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise RawError(f"{path.name}: no audio stream found")

    return AudioProbe(
        codec=(audio.get("codec_name") or "unknown").lower(),
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
        duration=float(fmt.get("duration") or 0.0),
        size_bytes=int(fmt.get("size") or path.stat().st_size),
        has_video=any(s.get("codec_type") == "video" for s in streams),
        container=(fmt.get("format_name") or "").split(",")[0],
    )


def normalize_audio(src: Path, dest_dir: Path, file_id: str) -> tuple[Path, AudioProbe, bool]:
    """Write the archival audio track for `src`. Returns (path, probe, copied).

    Idempotent: an existing output of non-zero size is reused, so re-running
    over a directory of already-processed streams costs one ffprobe each.
    """
    probe = probe_audio(src)
    copied = probe.is_compressed
    suffix = COPY_CONTAINERS[probe.codec] if copied else ".flac"
    dest = dest_dir / f"{file_id}{suffix}"

    if dest.exists() and dest.stat().st_size > 0:
        return dest, probe, copied

    dest_dir.mkdir(parents=True, exist_ok=True)
    # Keep the real extension last — ffmpeg infers the output format from it.
    tmp = dest.with_name(f"{dest.stem}.part{dest.suffix}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-map", "0:a:0"]
    cmd += ["-c:a", "copy"] if copied else ["-c:a", "flac", "-compression_level",
                                            FLAC_COMPRESSION_LEVEL]
    cmd.append(str(tmp))

    result = _run(cmd)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RawError(f"ffmpeg failed on {src.name}: {result.stderr.strip()[-300:]}")

    tmp.replace(dest)  # atomic: a killed run never leaves a half file in place
    return dest, probe, copied


def transcript_summary(json_path: Path) -> dict:
    """Pull the few fields worth having in the index, tolerating schema drift."""
    try:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
    linguistic = meta.get("linguistic_profile", {})
    speakers = meta.get("speaker_profile", {})
    transcription = doc.get("transcription", []) if isinstance(doc, dict) else []

    return {
        "primary_language": linguistic.get("primary_language", ""),
        "content_theme": linguistic.get("content_theme", ""),
        "speaker_count": speakers.get("speaker_count"),
        "transcript_blocks": len(transcription) if isinstance(transcription, list) else None,
    }


def index_path(stage_dir: Path) -> Path:
    return stage_dir / "index.jsonl"


def read_index(stage_dir: Path) -> dict[str, dict]:
    """Previously indexed rows, keyed by file_id. Empty if there is no index."""
    path = index_path(stage_dir)
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated write should not poison the whole index
            if isinstance(row, dict) and row.get("file_id"):
                rows[row["file_id"]] = row
    return rows


def build_record(
    file_id: str,
    json_path: Path,
    audio_src: Path,
    stage_dir: Path,
    known: dict[str, dict] | None = None,
) -> RawRecord:
    """Normalize one (json, audio) pair into `stage_dir` and describe it.

    A matching row in `known` (same file, same size) lets the digests be reused
    instead of re-read. Hashing is O(bytes), so without this a re-run over an
    archive of multi-hour streams re-reads the entire archive to add one file.
    """
    audio_dir = stage_dir / "audio"
    transcript_dir = stage_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    audio_path, probe, copied = normalize_audio(audio_src, audio_dir, file_id)

    transcript_path = transcript_dir / f"{file_id}.json"
    if not transcript_path.exists():
        shutil.copy2(json_path, transcript_path)

    audio_bytes = audio_path.stat().st_size
    cached = (known or {}).get(file_id) or {}
    reusable = (
        cached.get("audio_bytes") == audio_bytes
        and isinstance(cached.get("audio_sha256"), str)
        and isinstance(cached.get("transcript_sha256"), str)
    )

    return RawRecord(
        file_id=file_id,
        audio_name=audio_path.name,
        transcript_name=transcript_path.name,
        probe=probe,
        audio_sha256=cached["audio_sha256"] if reusable else sha256(audio_path),
        audio_bytes=audio_bytes,
        transcript_sha256=(
            cached["transcript_sha256"] if reusable else sha256(transcript_path)
        ),
        stream_copied=copied,
        source_name=audio_src.name,
        meta=transcript_summary(json_path),
    )


def merge_index(records: list[RawRecord], stage_dir: Path) -> list[dict]:
    """Fold this run's records into the index and return every row.

    The index describes the whole dataset, not one run. Processing a single
    recording (`--file-id`) must not rewrite it down to that one row — doing so
    would drop every other recording from the published index and from the card
    built off it.
    """
    rows = read_index(stage_dir)
    for record in records:
        rows[record.file_id] = record.index_row()

    ordered = sorted(rows.values(), key=lambda r: str(r.get("file_id", "")))
    path = index_path(stage_dir)
    tmp = path.with_suffix(".jsonl.part")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomic: never leave a half-written index behind
    return ordered


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}TB"


def human_duration(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
