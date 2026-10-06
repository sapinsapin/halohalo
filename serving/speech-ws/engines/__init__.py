"""
Engine registry. The server talks to two small interfaces and never to a
model library directly, so the TTS model can be swapped by env var and the
ASR model lives behind one `transcribe()`.

TTS engine:
    served_langs() -> tuple[str, ...]
    warm(lang)                                   load + one real forward pass
    synth_stream(text, lang, voice, cancel)      -> iterator of PCM S16LE 16 kHz bytes
    info() -> dict                               what /v1/models reports

ASR engine:
    served_langs() -> tuple[str, ...]
    warm(lang)
    transcribe(audio_f32_16k, lang) -> str
    info() -> dict
"""

from functools import lru_cache

import settings as S


@lru_cache(maxsize=1)
def get_tts():
    if S.TTS_ENGINE == "fake":
        from engines.fake import FakeTts
        return FakeTts(S.TTS_LANGS)
    if S.TTS_ENGINE == "speecht5":
        from engines.speecht5 import SpeechT5Engine
        return SpeechT5Engine(S.TTS_LANGS)
    if S.TTS_ENGINE in ("orpheus", "orpheus-3b"):
        from engines.orpheus import OrpheusEngine
        return OrpheusEngine(S.TTS_LANGS)
    if S.TTS_ENGINE in ("none", "off", ""):
        return None
    raise RuntimeError(f"unknown TTS_ENGINE={S.TTS_ENGINE!r} (orpheus, speecht5, none)")


@lru_cache(maxsize=1)
def get_asr():
    if not S.ASR_LANGS or S.ASR_ENGINE in ("none", "off"):
        return None
    if S.ASR_ENGINE == "fake":
        from engines.fake import FakeAsr
        return FakeAsr(S.ASR_LANGS)
    from engines.whisper_asr import WhisperAsr
    return WhisperAsr(S.ASR_LANGS)


def pick_device(preferred: str = "") -> str:
    if preferred:
        return preferred
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:          # tests without torch
        return "cpu"
