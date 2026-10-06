"""
Model-free engines for tests and for proving a deployment path (tunnel,
gateway, tickets) before the weights are downloaded: TTS_ENGINE=fake emits a
tone whose length follows the text; ASR_ENGINE=fake returns the duration of
the audio it was given. Both honour the interfaces in engines/__init__.py.
"""

import threading
from typing import Iterator

import numpy as np

import settings as S
from pcm import f32_to_pcm16


class FakeTts:
    name = "fake-tone"
    device = "cpu"

    def __init__(self, langs):
        self.langs = tuple(langs)
        self.calls: list[tuple[str, str, str]] = []

    def served_langs(self):
        return self.langs

    def info(self):
        return {"engine": self.name, "device": self.device, "sample_rate": S.SR}

    def warm(self, lang):
        list(self.synth_stream("Kumusta po.", lang, {"id": "X"}, None))

    def synth_stream(self, text: str, lang: str, voice: dict,
                     cancel: threading.Event | None) -> Iterator[bytes]:
        """60 ms of a 440 Hz tone per character, in 2048-sample pieces, so
        a request produces several chunks and the framing code is exercised."""
        self.calls.append((text, lang, voice["id"]))
        n = int(S.SR * 0.06 * max(1, len(text)))
        t = np.arange(n, dtype=np.float32) / S.SR
        wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for i in range(0, n, 2048):
            if cancel is not None and cancel.is_set():
                return
            yield f32_to_pcm16(wav[i:i + 2048])


class FakeAsr:
    name = "fake-duration"
    device = "cpu"

    def __init__(self, langs):
        self.langs = tuple(langs)
        self.calls: list[tuple[int, str]] = []

    def served_langs(self):
        return self.langs

    def info(self):
        return {"engine": self.name, "device": self.device, "sample_rate": S.SR}

    def warm(self, lang):
        self.transcribe(np.zeros(S.SR, dtype=np.float32), lang)

    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        self.calls.append((int(audio.size), lang))
        return f"[{lang}] {audio.size / S.SR:.1f}s"
