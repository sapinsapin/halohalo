"""
Whisper large-v3 finetunes (sapinsapin/whisper-large-v3-pld-<lang>-norm): the
most accurate ASR the org has published per language on the honest,
speaker- and prompt-disjoint split (dashboard/asr_registry.json;
halolib.finetune._BEST_ASR). English has no large-v3 finetune yet and falls
back to whisper-small-pld-eng.

Every model trains under Whisper's <|tl|> token except English
(scripts/tts_eval.py WHISPER_LANG), so that is what generation is told. The
-norm models return normalised text: lower case, no punctuation.

Models load lazily and at most ASR_MAX_RESIDENT stay in memory (large-v3 is
about 3 GB in fp16); on a 128 GB GB10 that can be all nine.
"""

import logging
import threading
from collections import OrderedDict

import numpy as np

import settings as S
from engines import pick_device
from voices import WHISPER_LANGUAGE

log = logging.getLogger("asr")

WINDOW_S = 30.0          # Whisper's input window
OVERLAP_S = 0.5


class WhisperAsr:
    name = "whisper-large-v3"

    def __init__(self, langs):
        self.langs = tuple(langs)
        self.device = pick_device(S.ASR_DEVICE)
        self._models: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()

    def served_langs(self) -> tuple[str, ...]:
        return self.langs

    def repo_for(self, lang: str) -> str:
        return S.ASR_REPO_ENG if lang == "eng" else S.ASR_REPO.format(lang=lang)

    def info(self) -> dict:
        return {"engine": self.name, "device": self.device, "repo": S.ASR_REPO,
                "repo_eng": S.ASR_REPO_ENG, "resident": list(self._models),
                "sample_rate": S.SR}

    def warm(self, lang: str) -> None:
        self.transcribe(np.zeros(S.SR, dtype=np.float32), lang)

    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        """Float32 16 kHz mono -> text. Longer than 30 s is transcribed in
        overlapping windows and joined."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""
        processor, model, torch = self._load(lang)
        window = int(WINDOW_S * S.SR)
        step = int((WINDOW_S - OVERLAP_S) * S.SR)
        pieces = []
        for start in range(0, max(1, audio.size), step):
            chunk = audio[start:start + window]
            if chunk.size == 0:
                break
            pieces.append(self._decode(processor, model, torch, chunk, lang))
            if start + window >= audio.size:
                break
        return " ".join(p for p in pieces if p).strip()

    def _decode(self, processor, model, torch, chunk: np.ndarray, lang: str) -> str:
        feats = processor(chunk, sampling_rate=S.SR, return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=model.dtype)
        with self._lock, torch.inference_mode():
            ids = model.generate(feats, language=WHISPER_LANGUAGE.get(lang, "tagalog"),
                                 task="transcribe", max_new_tokens=440, num_beams=1)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def _load(self, lang: str):
        with self._lock:
            if lang in self._models:
                self._models.move_to_end(lang)
                return self._models[lang]
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            repo = self.repo_for(lang)
            log.info("loading %s on %s", repo, self.device)
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            processor = WhisperProcessor.from_pretrained(repo)
            model = WhisperForConditionalGeneration.from_pretrained(repo, dtype=dtype)
            model = model.eval().to(self.device)
            self._models[lang] = (processor, model, torch)
            while len(self._models) > max(1, S.ASR_MAX_RESIDENT):
                evicted, _ = self._models.popitem(last=False)
                log.info("evicted %s from the ASR cache", evicted)
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            return self._models[lang]
