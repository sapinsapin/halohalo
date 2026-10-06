"""
SpeechT5 (sapinsapin/speecht5_tts-pld-<lang>) with the HiFi-GAN vocoder and
precomputed x-vectors: the engine serving/tts-ws runs, here as the fallback
when Orpheus is not wanted (a CPU host, or all ten languages - Orpheus has no
Pangasinan adapter yet). Same synthesis code, GPU-aware. The repos are
private since 2026-09-22, so HF_TOKEN must be set to load them.

Not a streaming model: generate_speech returns a whole waveform, so one
segment yields one chunk and the segmenter (textseg.py) is what makes the
audio start early.
"""

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

import settings as S
from engines import pick_device
from pcm import f32_to_pcm16, trim_edges

log = logging.getLogger("speecht5")
ROOT = Path(__file__).resolve().parent.parent


class SpeechT5Engine:
    name = "speecht5"

    def __init__(self, langs):
        self.langs = tuple(langs)
        self.device = pick_device(S.TTS_DEVICE)
        self._lock = threading.Lock()

    def served_langs(self) -> tuple[str, ...]:
        return self.langs

    def info(self) -> dict:
        return {"engine": self.name, "device": self.device, "repo": S.SPEECHT5_REPO,
                "vocoder": S.VOCODER_REPO, "sample_rate": S.SR}

    def warm(self, lang: str) -> None:
        from voices import resolve_voice
        list(self.synth_stream("Kumusta po.", lang, resolve_voice(lang, None), None))

    def synth_stream(self, text: str, lang: str, voice: dict,
                     cancel: threading.Event | None) -> Iterator[bytes]:
        if cancel is not None and cancel.is_set():
            return
        torch = self._torch()
        tokenizer, model = self._model(lang)
        if not voice.get("file"):
            from voices import resolve_voice
            voice = resolve_voice(lang, None)       # a bare speaker id has no x-vector
        inputs = tokenizer(text, return_tensors="pt")
        with self._lock, torch.no_grad():
            speech = model.generate_speech(
                inputs["input_ids"].to(self.device),
                self._xvector(voice["file"]),
                vocoder=self._vocoder(),
                maxlenratio=S.MAXLENRATIO,
            )
        wav = speech.float().cpu().numpy()
        cap = int(S.SR * S.MAX_SEG_SECONDS)
        if wav.size > cap:
            log.warning("segment hit the %.0fs cap - truncating", S.MAX_SEG_SECONDS)
            wav = wav[:cap]
        wav = trim_edges(wav)
        if wav.size:
            yield f32_to_pcm16(wav)

    @lru_cache(maxsize=1)
    def _torch(self):
        import torch
        torch.set_num_threads(S.TORCH_THREADS)
        return torch

    @lru_cache(maxsize=10)
    def _model(self, lang: str):
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Tokenizer
        repo = S.SPEECHT5_REPO.format(lang=lang)
        log.info("loading %s on %s", repo, self.device)
        return (SpeechT5Tokenizer.from_pretrained(repo),
                SpeechT5ForTextToSpeech.from_pretrained(repo).eval().to(self.device))

    @lru_cache(maxsize=1)
    def _vocoder(self):
        from transformers import SpeechT5HifiGan
        return SpeechT5HifiGan.from_pretrained(S.VOCODER_REPO).eval().to(self.device)

    @lru_cache(maxsize=32)
    def _xvector(self, rel: str):
        return self._torch().tensor(np.load(ROOT / rel)).unsqueeze(0).to(self.device)
