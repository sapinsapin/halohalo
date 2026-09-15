"""
CPU SpeechT5 synthesis, ported from halohalo-dashboard/speech_demo.py.

Copied rather than imported: that module imports gradio at module scope, which
this Space does not install. The differences from the original are deliberate
and noted inline — thread count, a bounded decode, and PCM S16LE output.

Speaker x-vectors are precomputed .npy files (see assets.json), so speechbrain
is not a runtime dependency here either.
"""

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from pins import PINS, VOCODER, pinned_langs, tts_repo
from settings import (GAIN, MAXLENRATIO, MAX_SEG_SECONDS, SERVE_LANGS, SR,
                      TORCH_THREADS, TRIM_KEEP_MS, TRIM_TAIL_MS, TRIM_THRESHOLD,
                      VOCODER_REPO)

log = logging.getLogger("tts")

ROOT = Path(__file__).parent
ASSETS = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
LANGS = ASSETS["languages"]

# Accept a language as a code ("fil") or as a display name ("Filipino"),
# case-insensitively. Being liberal here costs nothing and firmware authors
# reasonably guess either.
LANG_ALIASES = {}
for _code, _meta in LANGS.items():
    LANG_ALIASES[_code.lower()] = _code
    LANG_ALIASES[_meta["name"].lower()] = _code


@lru_cache(maxsize=1)
def served_langs() -> tuple[str, ...]:
    """Languages this instance will serve: configured, known, and pinned.

    An unpinned language is dropped rather than served, because loading it
    would need a Hub round trip — a ~600 MB download inside whatever request
    first asked for it.
    """
    ok, dropped = [], []
    for code in SERVE_LANGS:
        if code not in LANGS:
            dropped.append(f"{code} (not in assets.json)")
        elif tts_repo(code) not in PINS:
            dropped.append(f"{code} (no pin in pins.py)")
        else:
            ok.append(code)
    if dropped:
        log.warning("not serving %s; pinned languages are %s",
                    ", ".join(dropped), ", ".join(pinned_langs()))
    return tuple(ok)


def resolve_lang(value: str | None, default: str) -> str | None:
    """Return a served language code, or None if the request named an unknown one."""
    if value is None or str(value).strip() == "":
        return default
    code = LANG_ALIASES.get(str(value).strip().lower())
    return code if code in served_langs() else None


def resolve_voice(lang: str, value: str | None) -> dict:
    """Pick a voice for a language. Unknown values fall back silently."""
    voices = LANGS[lang]["voices"]
    if not voices:
        raise RuntimeError(f"no voice preset for {lang}")
    if value:
        want = str(value).strip().lower()
        for v in voices:
            if want in (v["id"].lower(), v["gender"].lower()):
                return v
        if want.isdigit() and int(want) < len(voices):
            return voices[int(want)]
    return voices[0]


def voice_catalog() -> dict:
    return {c: {"name": LANGS[c]["name"],
                "voices": [{"id": v["id"], "gender": v["gender"]}
                           for v in LANGS[c]["voices"]]}
            for c in served_langs()}


# ---------------------------------------------------------------- model cache

@lru_cache(maxsize=1)
def _torch():
    import torch
    # speech_demo.py uses 4 threads; the free CPU Space has 2 vCPU, so 4 is
    # oversubscription. OMP_NUM_THREADS is also set in the Dockerfile, because
    # setting it from Python after torch has imported is ignored.
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass          # already initialized; harmless
    return torch


@lru_cache(maxsize=4)
def tts_model(lang: str):
    """Load one language's tokenizer and model.

    SpeechT5Tokenizer, not SpeechT5Processor. The processor also builds a
    feature extractor, which text-to-speech never uses — it is there for the
    audio-input paths (ASR, voice conversion) that speech_demo.py also serves.
    Loading it is not merely wasteful: these repos ship no
    preprocessor_config.json, and while transformers tolerates that when it can
    reach the Hub and confirm the file is absent, an offline container cannot
    tell "absent" from "unreachable" and raises OSError. The tokenizer alone
    produces identical input_ids.
    """
    from transformers import SpeechT5ForTextToSpeech, SpeechT5Tokenizer
    repo, rev = tts_repo(lang), PINS[tts_repo(lang)]
    log.info("loading %s@%s", repo, rev[:8])
    return (SpeechT5Tokenizer.from_pretrained(repo, revision=rev),
            SpeechT5ForTextToSpeech.from_pretrained(repo, revision=rev).eval())


@lru_cache(maxsize=1)
def vocoder():
    from transformers import SpeechT5HifiGan
    log.info("loading %s@%s", VOCODER_REPO, PINS[VOCODER_REPO][:8])
    # Emits a benign "using a model of type hifigan to instantiate
    # speecht5_hifigan" warning; expected, not a misconfiguration.
    return SpeechT5HifiGan.from_pretrained(
        VOCODER_REPO, revision=PINS[VOCODER_REPO]).eval()


@lru_cache(maxsize=32)
def xvector(rel: str):
    return _torch().tensor(np.load(ROOT / rel)).unsqueeze(0)


# ------------------------------------------------------------------ conversion

def trim_edges(wav: np.ndarray) -> np.ndarray:
    """Drop leading and trailing silence from one segment.

    These checkpoints emit seconds of low-level noise before the speech starts
    (3.9s of it on a 63-character Filipino sentence) plus a tail after it. Left
    in, the device plays hiss before the voice arrives and again between every
    pair of sentences, on top of the gap inserted deliberately.

    The threshold is relative to the segment's own peak frame RMS, so it does
    not care how loud the segment is: measured noise sits at 1-5% and speech at
    66-100%. Pre- and post-roll are kept so onsets and decays survive.
    """
    n = max(1, SR // 50)                          # 20 ms frames
    usable = wav.size - wav.size % n
    if usable < n * 2:
        return wav
    frames = wav[:usable].reshape(-1, n).astype(np.float32)
    rms = np.sqrt((frames * frames).mean(axis=1))
    peak = float(rms.max())
    if peak <= 0.0:
        return wav[:0]                            # silent segment -> caller errors
    voiced = np.nonzero(rms > peak * TRIM_THRESHOLD)[0]
    if voiced.size == 0:
        return wav[:0]
    pre = max(1, int(SR * TRIM_KEEP_MS / 1000) // n)
    post = max(1, int(SR * TRIM_TAIL_MS / 1000) // n)
    lo = max(0, int(voiced[0]) - pre) * n
    hi = min(wav.size, (int(voiced[-1]) + 1 + post) * n)
    return wav[lo:hi]


def f32_to_pcm16(x: np.ndarray, gain: float = GAIN) -> bytes:
    """float32 waveform in ~[-1, 1] -> PCM S16LE bytes.

    Three details here are the difference between working audio and audio that
    sounds broken in a way the partner cannot diagnose:

      * clip before the multiply, so a hot sample saturates instead of wrapping
      * scale by 32767, not 32768 — numpy does not raise on out-of-range
        float->int casts, so +1.0 * 32768 wraps to -32768: a full-scale
        positive sample becomes full-scale negative, i.e. a loud click
      * write '<i2' explicitly; byte-swapped S16 sounds like pulsing static

    No peak normalization: normalizing per segment makes loudness pump audibly
    between sentences, and normalizing the whole utterance would require
    synthesizing all of it first, which defeats streaming.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1).copy()
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    if gain != 1.0:
        x *= np.float32(gain)
    np.clip(x, -1.0, 1.0, out=x)
    y = np.rint(x * 32767.0).astype(np.int16)
    return y.astype("<i2", copy=False).tobytes()


def silence(ms: int) -> bytes:
    return b"\x00\x00" * int(SR * ms / 1000)


# ------------------------------------------------------------------- synthesis

def synth_segment(text: str, lang: str, voice: dict,
                  cancel: threading.Event | None = None) -> bytes | None:
    """Synthesize one segment to PCM S16LE. Returns None if already cancelled.

    Runs in a worker thread. It cannot be interrupted mid-forward-pass —
    generate_speech takes no stopping hook — so cancellation is checked on entry
    and the caller checks again on return. That is why segments are kept short.
    """
    if cancel is not None and cancel.is_set():
        return None
    torch = _torch()
    tokenizer, model = tts_model(lang)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            xvector(voice["file"]),
            vocoder=vocoder(),
            # Bounded on purpose: these checkpoints do not predict their stop
            # token reliably (speech_demo.py documents the same problem for the
            # voice-conversion model), and the 20.0 default lets a long segment
            # run on for the better part of a minute.
            maxlenratio=MAXLENRATIO,
        )
    wav = speech.cpu().numpy()
    cap = int(SR * MAX_SEG_SECONDS)
    if wav.size > cap:
        log.warning("segment hit the %.0fs cap (%d samples) — truncating",
                    MAX_SEG_SECONDS, wav.size)
        wav = wav[:cap]
    wav = trim_edges(wav)
    return f32_to_pcm16(wav)


def dropped_chars(text: str, lang: str) -> str:
    """Characters the tokenizer silently discards (e.g. digits, '₱').

    SpeechT5Tokenizer drops what it does not know without complaint, so '₱500'
    can simply vanish from the spoken output. Logged, not corrected — a
    Filipino number verbalizer is out of scope for this demo.
    """
    tokenizer, _ = tts_model(lang)
    kept = set(tokenizer.decode(
        tokenizer(text)["input_ids"], skip_special_tokens=True).lower())
    return "".join(sorted({c for c in text.lower()
                           if not c.isspace() and c not in kept}))


def warm(lang: str) -> None:
    """Load and run one real forward pass.

    from_pretrained alone is not enough: the first generate_speech call pays
    lazy oneDNN kernel selection, which dominates cold-start latency. Keep the
    text tiny so this costs seconds, not half a minute.
    """
    synth_segment("Kumusta po.", lang, resolve_voice(lang, None))
