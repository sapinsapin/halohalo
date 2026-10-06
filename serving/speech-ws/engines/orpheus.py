"""
Orpheus 3B with the sapinsapin char-frontend LoRA adapters, streamed.

The published best TTS for the PLD languages (dashboard/tts_compare.json:
round-trip CER 4-12 % against SpeechT5's 30-300 %). One base model is loaded
once; one adapter per language is attached with PEFT and switched per
request. Audio comes out as SNAC codec tokens, decoded four frames at a time
as they are generated (orpheus_tokens.FrameWindow), resampled 24 -> 16 kHz
with a seam-free streaming resampler, and handed to the socket as PCM S16LE.
Time to first audio is therefore the time to the first 28 tokens, not to the
whole sentence.

Two generator backends feed the same decoder:

  transformers  HF generate + PEFT, exactly the published recipe
                (finetune_orpheus.synthesize_samples). Simple and correct;
                on a GB10 it runs a 3B model at roughly 25-35 tokens/s, which
                is about a third of realtime (82 tokens/s of speech), so the
                client buffers a sentence before playing - as the Agora
                contract already tells devices to do.
  llamacpp      a llama-server with the base model as GGUF and the adapters
                as LoRA GGUFs (docs/GB10.md). Two to three times faster on
                the memory-bound GB10; this module only parses its token
                stream. Not yet measured on the GB10 - see the docs.
"""

import json
import logging
import queue
import threading
import time
from typing import Iterator

import numpy as np

import settings as S
from engines import pick_device
from orpheus_tokens import (EOS_SPEECH, PAD, SNAC_SR, FrameWindow, codes_from_tokens,
                            custom_token_id, has_digits, prompt_text, wrap_prompt)
from pcm import StreamResampler, f32_to_pcm16

log = logging.getLogger("orpheus")


class _Cancelled(Exception):
    pass


class OrpheusEngine:
    name = "orpheus-3b"

    def __init__(self, langs):
        self.langs = tuple(langs)
        self.device = pick_device(S.TTS_DEVICE)
        self.backend = S.ORPHEUS_BACKEND
        self._lock = threading.Lock()          # one adapter active at a time
        self._tok = self._model = self._snac = None
        self._loaded_adapters: set[str] = set()
        self._lora_ids = _parse_lora_ids(S.LLAMACPP_LORA)

    # ------------------------------------------------------------ interface
    def served_langs(self) -> tuple[str, ...]:
        return self.langs

    def info(self) -> dict:
        return {"engine": self.name, "backend": self.backend, "device": self.device,
                "base": S.ORPHEUS_BASE, "adapter": S.ORPHEUS_ADAPTER,
                "codec": S.SNAC_REPO, "sample_rate": S.SR}

    def warm(self, lang: str) -> None:
        self._ensure_loaded()
        if self.backend == "transformers":
            self._ensure_adapter(lang)
        # a real forward pass pays kernel selection once, not inside a request
        t0 = time.perf_counter()
        voice = _default_voice(lang)
        n = sum(len(c) for c in self.synth_stream("Kumusta po.", lang, voice, None))
        log.info("warm %s: %d bytes in %.1fs", lang, n, time.perf_counter() - t0)

    def synth_stream(self, text: str, lang: str, voice: dict,
                     cancel: threading.Event | None) -> Iterator[bytes]:
        """Yield PCM S16LE 16 kHz chunks for one segment as they are decoded."""
        self._ensure_loaded()
        if has_digits(text):
            log.warning("digits in %r: the model never saw numerals; write them as words", text[:60])
        prompt = prompt_text(voice["id"], text)
        ids = wrap_prompt(self._tok(prompt, add_special_tokens=False).input_ids)
        if S.LOG_TEXT:
            log.debug("prompt %r (%d tokens)", prompt, len(ids))

        window = FrameWindow()
        resampler = StreamResampler(SNAC_SR, S.SR)
        tokens_q: queue.Queue = queue.Queue()
        err: list[BaseException] = []

        def produce():
            try:
                if self.backend == "llamacpp":
                    self._generate_llamacpp(ids, lang, cancel, tokens_q.put)
                else:
                    self._generate_transformers(ids, lang, cancel, tokens_q.put)
            except _Cancelled:
                pass
            except BaseException as exc:            # noqa: BLE001 - reported to the caller
                err.append(exc)
            finally:
                tokens_q.put(None)

        th = threading.Thread(target=produce, name="orpheus-gen", daemon=True)
        th.start()
        try:
            while True:
                tok = tokens_q.get()
                if tok is None:
                    break
                for win, sl in window.push(tok):
                    pcm = self._decode_slice(win, sl, resampler)
                    if pcm:
                        yield pcm
            if err:
                raise err[0]
            if cancel is not None and cancel.is_set():
                return
            for win, sl in window.flush():
                pcm = self._decode_slice(win, sl, resampler)
                if pcm:
                    yield pcm
            tail = resampler.flush()
            if tail.size:
                yield f32_to_pcm16(tail)
        finally:
            th.join(timeout=0.1)

    # -------------------------------------------------------------- loading
    def _ensure_loaded(self):
        if self._tok is not None:
            return
        with self._lock:
            if self._tok is not None:
                return
            import torch
            from transformers import AutoTokenizer
            torch.set_num_threads(S.TORCH_THREADS)
            self._torch = torch
            log.info("loading tokenizer %s", S.ORPHEUS_BASE)
            self._tok = AutoTokenizer.from_pretrained(S.ORPHEUS_BASE)
            from snac import SNAC
            log.info("loading codec %s on %s", S.SNAC_REPO, self.device)
            self._snac = SNAC.from_pretrained(S.SNAC_REPO).eval().to(self.device)
            if self.backend == "transformers":
                from transformers import AutoModelForCausalLM
                log.info("loading %s (bf16) on %s", S.ORPHEUS_BASE, self.device)
                self._model = AutoModelForCausalLM.from_pretrained(
                    S.ORPHEUS_BASE, dtype=torch.bfloat16, device_map=self.device).eval()

    def _ensure_adapter(self, lang: str):
        if lang in self._loaded_adapters:
            return
        with self._lock:
            if lang in self._loaded_adapters:
                return
            from peft import PeftModel
            repo = S.ORPHEUS_ADAPTER.format(lang=lang)
            log.info("loading adapter %s as %s", repo, lang)
            if not self._loaded_adapters:
                self._model = PeftModel.from_pretrained(self._model, repo, adapter_name=lang).eval()
            else:
                self._model.load_adapter(repo, adapter_name=lang)
            self._loaded_adapters.add(lang)

    # ------------------------------------------------------------ generators
    def _generate_transformers(self, ids, lang, cancel, emit):
        self._ensure_adapter(lang)
        torch = self._torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                flag = bool(cancel is not None and cancel.is_set())
                return torch.full((input_ids.shape[0],), flag, dtype=torch.bool,
                                  device=input_ids.device)

        class _Streamer:
            """Token-id streamer: generate() calls put() with the prompt once,
            then with each new token; the first call is the prompt and is skipped."""
            def __init__(self):
                self.first = True

            def put(self, value):
                if self.first:
                    self.first = False
                    return
                for t in value.reshape(-1).tolist():
                    if t == EOS_SPEECH:
                        return
                    emit(int(t))

            def end(self):
                pass

        with self._lock:                        # adapter switch + generate are one unit
            self._model.set_adapter(lang)
            prompt = torch.tensor([ids], device=self.device)
            with torch.inference_mode():
                self._model.generate(
                    prompt,
                    max_new_tokens=S.ORPHEUS_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=S.ORPHEUS_TEMPERATURE,
                    top_p=S.ORPHEUS_TOP_P,
                    repetition_penalty=S.ORPHEUS_REPETITION_PENALTY,
                    eos_token_id=EOS_SPEECH,
                    pad_token_id=self._tok.pad_token_id or PAD,
                    streamer=_Streamer(),
                    stopping_criteria=StoppingCriteriaList([_Stop()]),
                )
        if cancel is not None and cancel.is_set():
            raise _Cancelled()

    def _generate_llamacpp(self, ids, lang, cancel, emit):
        """Stream from a llama-server holding the base GGUF and the LoRA GGUFs.
        The prompt goes as token ids, so the frontend is identical; audio
        tokens come back as the tokenizer's <custom_token_N> strings."""
        import httpx
        body = {
            "prompt": ids, "stream": True, "n_predict": S.ORPHEUS_MAX_NEW_TOKENS,
            "temperature": S.ORPHEUS_TEMPERATURE, "top_p": S.ORPHEUS_TOP_P,
            "repeat_penalty": S.ORPHEUS_REPETITION_PENALTY,
            "stop": [f"<custom_token_{EOS_SPEECH - 128256}>"], "cache_prompt": False,
        }
        if lang in self._lora_ids:
            # one base, one adapter per language, scaled per request
            body["lora"] = [{"id": i, "scale": 1.0 if i == self._lora_ids[lang] else 0.0}
                            for i in sorted(set(self._lora_ids.values()))]
        headers = {}
        key = S.API_KEY
        if key:
            headers["Authorization"] = f"Bearer {key}"
        with httpx.Client(timeout=httpx.Timeout(10.0, read=120.0)) as client:
            with client.stream("POST", f"{S.LLAMACPP_URL.rstrip('/')}/completion",
                               json=body, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if cancel is not None and cancel.is_set():
                        raise _Cancelled()
                    if not line.startswith("data: "):
                        continue
                    try:
                        chunk = json.loads(line[6:])
                    except ValueError:
                        continue
                    for piece in _pieces(chunk.get("content", "")):
                        t = custom_token_id(piece)
                        if t is not None and t != EOS_SPEECH:
                            emit(t)
                    if chunk.get("stop"):
                        break

    # --------------------------------------------------------------- decoding
    def _decode_slice(self, window: list[int], sl: slice, resampler: StreamResampler) -> bytes:
        torch = self._torch
        c0, c1, c2 = codes_from_tokens(window)
        if not c0:
            return b""
        codes = [torch.tensor(c, dtype=torch.int32, device=self.device)[None] for c in (c0, c1, c2)]
        with torch.inference_mode():
            audio = self._snac.decode(codes)
        wav = audio[0, 0].float().cpu().numpy()[sl]
        if wav.size == 0:
            return b""
        peak = float(np.abs(wav).max())
        if peak > 1.0:                                # a few SNAC decodes overshoot
            wav = wav / peak * 0.95
        return f32_to_pcm16(resampler.process(wav))


def _pieces(content: str) -> list[str]:
    import re
    return re.findall(r"<custom_token_\d+>", content)


def _parse_lora_ids(spec: str) -> dict[str, int]:
    """"fil=0,ceb=1" -> {"fil": 0, "ceb": 1}: llama-server's --lora order."""
    out = {}
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        if "=" in part:
            lang, i = part.split("=", 1)
            if i.strip().isdigit():
                out[lang.strip()] = int(i)
    return out


def _default_voice(lang: str) -> dict:
    from voices import resolve_voice
    return resolve_voice(lang, None)
