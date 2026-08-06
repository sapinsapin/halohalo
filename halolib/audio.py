"""
Audio helpers shared by the speech preprocessors.

Decoding goes through ffmpeg (handles mp4/m4a/mov containers that soundfile
cannot open); slicing goes through soundfile's windowed read so a 30-minute
source is never fully loaded to pull a 5-second segment.
"""

import io
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ASR_SR = 16000
TTS_SR = 24000


def decode_audio(src: Path, dest: Path, sr: int, mono: bool = True) -> bool:
    """Decode/resample any container to WAV at `sr`. No-op if dest exists."""
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd.append(str(dest))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [warn] ffmpeg failed on {src.name}: {result.stderr.strip()[-300:]}")
        return False
    return True


def audio_duration(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def read_window(path: Path, start: float, end: float) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Read only [start, end) from a WAV — avoids loading the whole file."""
    try:
        sr = sf.info(path).samplerate
        data, _ = sf.read(
            path,
            start=max(0, int(start * sr)),
            stop=int(end * sr),
            always_2d=False,
        )
    except Exception as e:
        print(f"  [warn] {path.name}: {e}")
        return None, None

    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def slice_samples(data: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Slice an already-loaded array by seconds."""
    return data[max(0, int(start * sr)): int(end * sr)]


def wav_bytes(data: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def fingerprint(path: Path, seconds: float = 60.0) -> str:
    """MD5 of the first `seconds` of decoded PCM — identifies re-uploaded sources.

    Content-based rather than file-hash based, so the same stream re-exported
    with different container metadata still collides.
    """
    import hashlib

    data, sr = read_window(path, 0.0, seconds)
    if data is None:
        return ""
    return hashlib.md5(np.asarray(data, dtype=np.float32).tobytes()).hexdigest()


def estimate_bytes(duration_s: float, sr: int = ASR_SR,
                   bytes_per_sample: int = 2, compression: float = 0.6) -> float:
    """Rough on-disk size for `duration_s` of PCM after parquet compression."""
    return duration_s * sr * bytes_per_sample * compression
