"""
Every tunable limit for the demo TTS WebSocket server, in one place.

Each one is an env var so the endpoint can be retuned for the partner's
firmware without a code change — several of them exist specifically because we
cannot inspect that firmware and may have to adjust after the first
integration test (frame size, pacing, ping timeouts).
"""

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# ------------------------------------------------------------------- audio
SR = 16000                                  # every speecht5_tts-pld-* emits 16 kHz
BYTES_PER_SAMPLE = 2                        # PCM S16LE

# 3200 bytes = 100 ms of 16 kHz mono S16. Must stay even: an odd-length frame
# shifts the receiver's int16 stream by one byte and every later sample is
# garbage — permanently, and it sounds like static rather than like an error.
CHUNK_BYTES = _int("CHUNK_BYTES", 3200)

GAIN = _float("GAIN", 1.0)                  # fixed, never per-segment normalized
GAP_MS = _int("GAP_MS", 120)                # silence inserted between segments

# These checkpoints emit several seconds of low-level noise before (and a tail
# after) the actual speech: measured on speecht5_tts-pld-fil, the noise floor
# sits at 1-5% of the segment's peak frame RMS while speech sits at 66-100%, so
# 10% separates them cleanly. Trimmed per segment, otherwise the device plays
# seconds of hiss before the voice starts and again between every sentence.
TRIM_THRESHOLD = _float("TRIM_THRESHOLD", 0.10)
TRIM_KEEP_MS = _int("TRIM_KEEP_MS", 40)          # pre-roll, so onsets survive
TRIM_TAIL_MS = _int("TRIM_TAIL_MS", 120)         # post-roll, so decay survives

# 0 disables pacing: frames go out as fast as the transport drains them. Set to
# 1.0 to drip them at realtime if the device's buffer turns out to be small.
PACE_FACTOR = _float("PACE_FACTOR", 0.0)

# ------------------------------------------------------------------- model
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "fil")
# Not env-configurable: it has to match a pin in pins.py.
VOCODER_REPO = "microsoft/speecht5_hifigan"

# Only languages listed here are served. Anything else is an error rather than
# a lazy 600 MB download inside a live connection.
_langs = os.environ.get("SERVE_LANGS", DEFAULT_LANG).strip()
SERVE_LANGS = tuple(x for x in (s.strip() for s in _langs.split(",")) if x)

TORCH_THREADS = _int("TORCH_THREADS", 2)

# generate_speech defaults to maxlenratio=20.0, which permits roughly 250 ms of
# audio per input token. speech_demo.py already documents that these SpeechT5
# checkpoints do not predict their stop token reliably, so an unbounded decode
# can babble for ~50 s on a long segment. Bound it, and hard-cut the waveform.
MAXLENRATIO = _float("MAXLENRATIO", 12.0)
MAX_SEG_SECONDS = _float("MAX_SEG_SECONDS", 30.0)

# ---------------------------------------------------------------- text/segmenting
# Segment length is the only lever on time-to-first-audio: a segment is fully
# synthesized before any of its bytes are sent, so chunk size does not matter
# here. Keep the first segment deliberately short.
FIRST_SEG_CHARS = _int("FIRST_SEG_CHARS", 90)
MAX_SEG_CHARS = _int("MAX_SEG_CHARS", 200)
MAX_SEGMENTS = _int("MAX_SEGMENTS", 12)

# A clause longer than this is split on whitespace as damage control. Above
# MAX_SEG_CHARS but with no comma to cut at, a segment is left long on purpose
# — see the _soft_wrap docstring for why mid-clause cuts sound worse.
HARD_SEG_CHARS = _int("HARD_SEG_CHARS", 300)

# Rejected, never truncated. Silent truncation makes the partner conclude the
# model is broken instead of that the request was too long.
MAX_TEXT_CHARS = _int("MAX_TEXT_CHARS", 500)

# --------------------------------------------------------------- concurrency
MAX_CONNS = _int("MAX_CONNS", 8)
MAX_INFLIGHT = _int("MAX_INFLIGHT", 1)      # 2 vCPU: predictable p50 > parallelism
QUEUE_WAIT_S = _float("QUEUE_WAIT_S", 15.0)  # reject after this, don't queue forever

MAX_FRAME_BYTES = _int("MAX_FRAME_BYTES", 8192)   # checked before JSON parsing
MAX_TOTAL_BYTES = _int("MAX_TOTAL_BYTES", SR * BYTES_PER_SAMPLE * 120)

SEND_TIMEOUT_S = _float("SEND_TIMEOUT_S", 30.0)   # firmware that stops draining
FIRST_FRAME_TIMEOUT_S = _float("FIRST_FRAME_TIMEOUT_S", 30.0)
IDLE_TIMEOUT_S = _float("IDLE_TIMEOUT_S", 60.0)   # after end, awaiting next request
CONN_MAX_SECONDS = _float("CONN_MAX_SECONDS", 300.0)

# ---------------------------------------------------------------------- auth
# Off by default: demo #1 is explicitly unauthenticated. Present now so
# enabling it later is a config change on both sides, not a firmware change.
TTS_TOKEN = os.environ.get("TTS_TOKEN", "").strip()

LOG_TEXT = os.environ.get("LOG_TEXT", "0") == "1"   # log request text at DEBUG
