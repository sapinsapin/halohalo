"""
Sample-format plumbing shared by every engine: float32 waveforms to and from
the wire's PCM S16LE at 16 kHz, a seam-free streaming resampler for the
24 kHz codec output, silence, a WAV header, and the edge trimmer SpeechT5
needs.

numpy (and scipy for the resampler) only, so it is testable anywhere.
"""

from fractions import Fraction

import numpy as np

from settings import (BYTES_PER_SAMPLE, GAIN, SR, TRIM_KEEP_MS, TRIM_TAIL_MS,
                      TRIM_THRESHOLD)


def f32_to_pcm16(x: np.ndarray, gain: float = GAIN) -> bytes:
    """float32 in ~[-1, 1] -> PCM S16LE bytes.

    Clip before the multiply, scale by 32767 (not 32768, where +1.0 wraps to
    -32768: a full-scale click), and write '<i2' explicitly. No per-segment
    normalization: it pumps loudness between sentences.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1).copy()
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    if gain != 1.0:
        x *= np.float32(gain)
    np.clip(x, -1.0, 1.0, out=x)
    y = np.rint(x * 32767.0).astype(np.int16)
    return y.astype("<i2", copy=False).tobytes()


def pcm16_to_f32(data: bytes) -> np.ndarray:
    usable = len(data) - (len(data) % BYTES_PER_SAMPLE)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0


def silence(ms: int, sr: int = SR) -> bytes:
    return b"\x00\x00" * int(sr * ms / 1000)


def wav_bytes(pcm: bytes, sr: int = SR) -> bytes:
    """A 44-byte RIFF header in front of PCM S16LE mono."""
    import struct
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", n) + pcm)


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """One-shot resample of a float32 waveform to 16 kHz (ASR input)."""
    if sr == SR:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly
    r = Fraction(SR, int(sr)).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=np.float64), r.numerator,
                         r.denominator).astype(np.float32)


class StreamResampler:
    """Polyphase resampling of a stream, chunk by chunk, without seams.

    A chunk resampled on its own gets the filter's edge response at both
    ends, and a stream of such chunks ticks at every boundary. So each call
    resamples the new input together with a tail of what came before, and
    only emits the output for the middle: the oldest part of the tail was
    already emitted (it is context now), the newest part is held back until
    the next call gives it right-hand context. Input is consumed in multiples
    of the ratio's denominator so every input position maps to an exact
    output position and no sample is dropped or doubled.

    Built for the 24 kHz SNAC output (3:2 to the wire's 16 kHz); any ratio
    with a small denominator works. `flush()` releases what is still held.
    """

    def __init__(self, src_sr: int, dst_sr: int = SR, overlap: int = 256):
        self.src, self.dst = int(src_sr), int(dst_sr)
        r = Fraction(self.dst, self.src).limit_denominator(1000)
        self.up, self.down = r.numerator, r.denominator
        self.ov = max(self.down, (overlap // self.down) * self.down)
        self._tail = np.zeros(0, dtype=np.float32)      # last <= 2*ov input samples
        self._held = 0                                   # of which not yet emitted
        self._pending = np.zeros(0, dtype=np.float32)    # < down input samples

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.src == self.dst:
            return x
        buf = np.concatenate([self._pending, x])
        usable = buf.size - buf.size % self.down
        self._pending = buf[usable:]
        x = buf[:usable]
        if x.size == 0:
            return x
        from scipy.signal import resample_poly
        joined = np.concatenate([self._tail, x])
        out = resample_poly(joined.astype(np.float64), self.up, self.down)
        skip = (self._tail.size - self._held) * self.up // self.down
        new_held = min(self.ov, joined.size)
        end = out.size - new_held * self.up // self.down
        self._tail = joined[-min(2 * self.ov, joined.size):]
        self._held = new_held
        return out[skip:end].astype(np.float32)

    def flush(self) -> np.ndarray:
        if self.src == self.dst:
            return np.zeros(0, dtype=np.float32)
        pad = (-self._pending.size) % self.down
        x = np.concatenate([self._pending, np.zeros(pad, dtype=np.float32)])
        joined = np.concatenate([self._tail, x])
        self._pending = np.zeros(0, dtype=np.float32)
        if joined.size == 0:
            return np.zeros(0, dtype=np.float32)
        from scipy.signal import resample_poly
        out = resample_poly(joined.astype(np.float64), self.up, self.down)
        skip = (self._tail.size - self._held) * self.up // self.down
        self._tail, self._held = np.zeros(0, dtype=np.float32), 0
        return out[skip:].astype(np.float32)


def trim_edges(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Drop leading and trailing noise floor from one SpeechT5 segment
    (see tts-ws/tts.py for the measurements behind the threshold)."""
    n = max(1, sr // 50)                          # 20 ms frames
    usable = wav.size - wav.size % n
    if usable < n * 2:
        return wav
    frames = wav[:usable].reshape(-1, n).astype(np.float32)
    rms = np.sqrt((frames * frames).mean(axis=1))
    peak = float(rms.max())
    if peak <= 0.0:
        return wav[:0]
    voiced = np.nonzero(rms > peak * TRIM_THRESHOLD)[0]
    if voiced.size == 0:
        return wav[:0]
    pre = max(1, int(sr * TRIM_KEEP_MS / 1000) // n)
    post = max(1, int(sr * TRIM_TAIL_MS / 1000) // n)
    lo = max(0, int(voiced[0]) - pre) * n
    hi = min(wav.size, (int(voiced[-1]) + 1 + post) * n)
    return wav[lo:hi]


def frames_of(pcm: bytes, size: int) -> list[bytes]:
    """Split PCM into wire frames of `size` bytes; the last may be shorter.
    `size` must be even so no int16 sample straddles two frames."""
    if size % 2:
        raise ValueError("frame size must be even")
    return [pcm[i:i + size] for i in range(0, len(pcm), size)]
