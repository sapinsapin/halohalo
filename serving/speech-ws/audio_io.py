"""
Decoding uploaded audio for /v1/audio/transcriptions: soundfile for WAV, FLAC,
OGG and MP3; ffmpeg (when on PATH) for everything else a browser or a phone
may send, such as WebM/Opus from MediaRecorder or M4A.
"""

import io
import shutil
import subprocess

import numpy as np

from pcm import pcm16_to_f32, resample_to_16k
from settings import SR


class UnsupportedAudio(Exception):
    pass


def decode_to_16k(data: bytes, content_type: str | None = None) -> np.ndarray:
    """Any supported container -> float32 mono 16 kHz."""
    if not data:
        raise UnsupportedAudio("empty file")
    if content_type and content_type.split(";")[0].strip() in ("audio/pcm", "audio/L16"):
        return pcm16_to_f32(data)
    try:
        import soundfile as sf
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return resample_to_16k(wav.mean(axis=1), int(sr))
    except Exception as exc:                        # noqa: BLE001 - try ffmpeg next
        first = exc
    if shutil.which("ffmpeg"):
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1",
             "-ar", str(SR), "pipe:1"],
            input=data, capture_output=True, timeout=120)
        if proc.returncode == 0 and proc.stdout:
            return pcm16_to_f32(proc.stdout)
        raise UnsupportedAudio(f"ffmpeg could not decode the file: {proc.stderr[-200:].decode(errors='replace')}")
    raise UnsupportedAudio(f"could not decode the file ({type(first).__name__}); send WAV, FLAC or OGG")
