"""
Silero VAD wrapper (CPU) — speech-region detection used to snap segment
boundaries to actual speech and to measure speech coverage per segment.
"""

import json
from pathlib import Path

import numpy as np

SR = 16000


class SileroVAD:
    def __init__(self):
        from silero_vad import load_silero_vad
        self.model = load_silero_vad(onnx=True)

    def speech_regions(self, audio_16k: np.ndarray) -> list[tuple[float, float]]:
        """Return [(start_s, end_s), ...] speech spans over the whole array."""
        import torch
        from silero_vad import get_speech_timestamps
        ts = get_speech_timestamps(
            torch.from_numpy(audio_16k.astype(np.float32)),
            self.model,
            sampling_rate=SR,
            return_seconds=True,
        )
        return [(t["start"], t["end"]) for t in ts]


def cached_speech_regions(status_path: Path, wav16: Path, vad: SileroVAD) -> list[tuple[float, float]]:
    """Whole-file VAD, cached beside the status file (one run per source)."""
    cache = status_path.with_suffix(".vad.json")
    if cache.exists():
        try:
            return [tuple(r) for r in json.loads(cache.read_text())]
        except json.JSONDecodeError:
            pass

    import soundfile as sf
    data, sr = sf.read(wav16, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    regions = vad.speech_regions(np.asarray(data))
    cache.write_text(json.dumps([list(r) for r in regions]))
    return regions


def trim_to_speech(start: float, end: float,
                   regions: list[tuple[float, float]],
                   max_trim: float = 1.0, pad: float = 0.15) -> tuple[float, float]:
    """Snap [start, end] inward to the nearest speech edges.

    Trims at most `max_trim` per side (a larger requested trim means VAD and
    the aligner disagree — keep the aligner's boundary), then re-pads by `pad`
    so cuts don't clip phoneme onsets.
    """
    overlapping = [(s, e) for s, e in regions if e > start and s < end]
    if not overlapping:
        return start, end

    first_speech = overlapping[0][0]
    last_speech = overlapping[-1][1]

    new_start = start
    if 0 <= first_speech - start <= max_trim:
        new_start = first_speech
    new_end = end
    if 0 <= end - last_speech <= max_trim:
        new_end = last_speech

    new_start = max(0.0, new_start - pad)
    new_end = new_end + pad
    if new_end <= new_start:
        return start, end
    return new_start, new_end


def speech_ratio(start: float, end: float,
                 regions: list[tuple[float, float]]) -> float:
    """Fraction of [start, end] covered by VAD speech."""
    dur = end - start
    if dur <= 0:
        return 0.0
    covered = 0.0
    for s, e in regions:
        covered += max(0.0, min(e, end) - max(s, start))
    return round(min(1.0, covered / dur), 3)
