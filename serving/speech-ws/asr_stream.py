"""
Endpointing for streaming recognition: turns a stream of PCM into the
phrases Whisper is asked to transcribe, and says when a partial result is
due. Pure numpy, so the policy is testable without a model.

Whisper is not a streaming model (one forward pass per 30 s window), so the
"streaming" a device sees is: audio is buffered while someone speaks, a
partial transcript of the phrase so far is produced every PARTIAL_EVERY_S,
and when the speaker pauses the phrase is transcribed once more and sent as
`final`. Energy-based voice activity by default; `vad=` accepts any callable
(float32 frame -> bool), so silero can be plugged in where it is installed.
"""

from dataclasses import dataclass, field

import numpy as np

from settings import (ASR_MAX_SEGMENT_S, ASR_PARTIAL_EVERY_S, SR, VAD_FRAME_MS,
                      VAD_MIN_SPEECH_MS, VAD_SILENCE_MS, VAD_THRESHOLD)


def energy_vad(frame: np.ndarray, threshold: float = VAD_THRESHOLD) -> bool:
    if frame.size == 0:
        return False
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) > threshold


@dataclass
class Segment:
    audio: np.ndarray
    start_s: float
    end_s: float
    final: bool


@dataclass
class Endpointer:
    sr: int = SR
    frame_ms: int = VAD_FRAME_MS
    silence_ms: int = VAD_SILENCE_MS
    min_speech_ms: int = VAD_MIN_SPEECH_MS
    partial_every_s: float = ASR_PARTIAL_EVERY_S
    max_segment_s: float = ASR_MAX_SEGMENT_S
    vad: object = energy_vad
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _speech: list = field(default_factory=list)
    _in_speech: bool = False
    _silence_frames: int = 0
    _speech_frames: int = 0
    _pos_frames: int = 0               # frames consumed since the start
    _seg_start: int = 0
    _since_partial: float = 0.0
    _preroll: list = field(default_factory=list)

    @property
    def frame_len(self) -> int:
        return int(self.sr * self.frame_ms / 1000)

    @property
    def consumed_s(self) -> float:
        return self._pos_frames * self.frame_ms / 1000

    def feed(self, audio: np.ndarray) -> list[Segment]:
        """Consume float32 16 kHz samples; return the segments now ready:
        partials (final=False) while a phrase goes on, a final when it ends."""
        out: list[Segment] = []
        buf = np.concatenate([self._pending, np.asarray(audio, dtype=np.float32).reshape(-1)])
        n = self.frame_len
        usable = buf.size - buf.size % n
        self._pending = buf[usable:]
        for i in range(0, usable, n):
            frame = buf[i:i + n]
            self._pos_frames += 1
            voiced = bool(self.vad(frame))
            if not self._in_speech:
                # keep a little pre-roll so onsets survive the detector's lag
                self._preroll.append(frame)
                if len(self._preroll) > 10:
                    self._preroll.pop(0)
                if voiced:
                    self._in_speech = True
                    self._seg_start = self._pos_frames - len(self._preroll)
                    self._speech = list(self._preroll)
                    self._preroll = []
                    self._speech_frames, self._silence_frames = 1, 0
                    self._since_partial = 0.0
                continue
            self._speech.append(frame)
            self._since_partial += self.frame_ms / 1000
            if voiced:
                self._speech_frames += 1
                self._silence_frames = 0
            else:
                self._silence_frames += 1
            seg_s = len(self._speech) * self.frame_ms / 1000
            ended = self._silence_frames * self.frame_ms >= self.silence_ms
            too_long = seg_s >= self.max_segment_s
            if ended or too_long:
                if self._speech_frames * self.frame_ms >= self.min_speech_ms:
                    out.append(self._segment(final=True))
                self._reset()
            elif self._since_partial >= self.partial_every_s and \
                    self._speech_frames * self.frame_ms >= self.min_speech_ms:
                self._since_partial = 0.0
                out.append(self._segment(final=False))
        return out

    def finish(self) -> list[Segment]:
        """The stream is over: whatever phrase is open is final."""
        out = []
        if self._in_speech and self._speech_frames * self.frame_ms >= self.min_speech_ms:
            out.append(self._segment(final=True))
        self._reset()
        self._pending = np.zeros(0, dtype=np.float32)
        return out

    def _segment(self, final: bool) -> Segment:
        audio = np.concatenate(self._speech) if self._speech else np.zeros(0, dtype=np.float32)
        start = self._seg_start * self.frame_ms / 1000
        return Segment(audio=audio, start_s=round(start, 3),
                       end_s=round(start + audio.size / self.sr, 3), final=final)

    def _reset(self):
        self._in_speech = False
        self._speech = []
        self._speech_frames = self._silence_frames = 0
        self._since_partial = 0.0
