"""
Every tunable for the GB10 speech server, in one place, each one an env var.

The TTS side keeps serving/tts-ws's wire contract (docs/AGORA_INTEGRATION.md
there): the names and defaults below that it shares with tts-ws are the same
on purpose, so a device tuned against the demo host behaves identically here.
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


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _list(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default).strip()
    return tuple(x for x in (s.strip() for s in raw.split(",")) if x)


# ----------------------------------------------------------------- identity
# What NYO calls this server. nyo-node passes --served-name; the HTTP
# /v1/models list and the OpenAI-shaped endpoints answer to it.
SERVED_NAME = os.environ.get("SPEECH_SERVED_NAME", "sapin-speech")

# ------------------------------------------------------------------- audio
SR = 16000                                  # the wire rate: PCM S16LE 16 kHz mono
BYTES_PER_SAMPLE = 2
# 3200 bytes = 100 ms. Must stay even (see tts-ws/settings.py for why).
CHUNK_BYTES = _int("CHUNK_BYTES", 3200)
GAIN = _float("GAIN", 1.0)
GAP_MS = _int("GAP_MS", 120)                # silence inserted between segments
PACE_FACTOR = _float("PACE_FACTOR", 0.0)    # 0 = as fast as the socket drains

# SpeechT5 only: its checkpoints emit a noise floor before and after speech.
TRIM_THRESHOLD = _float("TRIM_THRESHOLD", 0.10)
TRIM_KEEP_MS = _int("TRIM_KEEP_MS", 40)
TRIM_TAIL_MS = _int("TRIM_TAIL_MS", 120)

# --------------------------------------------------------------------- TTS
# orpheus: the published best (sapinsapin/orpheus-3b-0.1-pretrained-char-pld-*,
#          LoRA on a 3B codec LM, GPU). speecht5: the CPU baseline tts-ws serves.
TTS_ENGINE = os.environ.get("TTS_ENGINE", "orpheus")
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "fil")
# Languages to load. Orpheus adapters exist for nine (no Pangasinan yet);
# SpeechT5 has all ten. An unlisted language is an error, never a download
# inside a live connection.
TTS_LANGS = _list("TTS_LANGS", "fil,ceb,hil")
ORPHEUS_BASE = os.environ.get("ORPHEUS_BASE", "unsloth/orpheus-3b-0.1-pretrained")
ORPHEUS_ADAPTER = os.environ.get(
    "ORPHEUS_ADAPTER", "sapinsapin/orpheus-3b-0.1-pretrained-char-pld-{lang}")
SNAC_REPO = os.environ.get("SNAC_REPO", "hubertsiuzdak/snac_24khz")
# Sampling, as finetune_orpheus.synthesize_samples and scripts/tts_eval.py.
ORPHEUS_TEMPERATURE = _float("ORPHEUS_TEMPERATURE", 0.6)
ORPHEUS_TOP_P = _float("ORPHEUS_TOP_P", 0.9)
ORPHEUS_REPETITION_PENALTY = _float("ORPHEUS_REPETITION_PENALTY", 1.1)
# 7 tokens a frame, ~12 frames a second: 1400 tokens is about 16 s of speech,
# the cap finetune_orpheus.py uses. A segment is at most MAX_SEG_CHARS long,
# so this is the runaway guard, not the working range.
ORPHEUS_MAX_NEW_TOKENS = _int("ORPHEUS_MAX_NEW_TOKENS", 1400)
# Where the generator runs: "transformers" (HF generate + PEFT, the published
# recipe) or "llamacpp" (a llama-server with the merged GGUF and LoRA GGUFs;
# faster on a GB10, see docs). Both feed the same SNAC decoder.
ORPHEUS_BACKEND = os.environ.get("ORPHEUS_BACKEND", "transformers")
LLAMACPP_URL = os.environ.get("LLAMACPP_URL", "http://127.0.0.1:8081")
LLAMACPP_LORA = os.environ.get("LLAMACPP_LORA", "")   # "fil=0,ceb=1,..." adapter ids
TTS_DEVICE = os.environ.get("TTS_DEVICE", "")        # "" = cuda if available
SPEECHT5_REPO = os.environ.get("SPEECHT5_REPO", "sapinsapin/speecht5_tts-pld-{lang}")
VOCODER_REPO = "microsoft/speecht5_hifigan"
MAXLENRATIO = _float("MAXLENRATIO", 12.0)
MAX_SEG_SECONDS = _float("MAX_SEG_SECONDS", 30.0)
TORCH_THREADS = _int("TORCH_THREADS", 4)

# ------------------------------------------------------------- text/segments
FIRST_SEG_CHARS = _int("FIRST_SEG_CHARS", 90)
MAX_SEG_CHARS = _int("MAX_SEG_CHARS", 200)
MAX_SEGMENTS = _int("MAX_SEGMENTS", 12)
HARD_SEG_CHARS = _int("HARD_SEG_CHARS", 300)
MAX_TEXT_CHARS = _int("MAX_TEXT_CHARS", 500)

# --------------------------------------------------------------------- ASR
# whisper (the published best), or none; fake exists for tests and for
# proving a deployment before the weights are down.
ASR_ENGINE = os.environ.get("ASR_ENGINE", "whisper")
ASR_LANGS = _list("ASR_LANGS", "fil,ceb,hil")
# The honest-split best per language (halolib.finetune._BEST_ASR); English has
# no large-v3 finetune yet, so it falls back to the whisper-small baseline.
ASR_REPO = os.environ.get("ASR_REPO", "sapinsapin/whisper-large-v3-pld-{lang}-norm")
ASR_REPO_ENG = os.environ.get("ASR_REPO_ENG", "sapinsapin/whisper-small-pld-eng")
# How many ASR models stay resident at once (large-v3 is ~3 GB in fp16 each;
# a 128 GB GB10 can hold all nine, a smaller card cannot).
ASR_MAX_RESIDENT = _int("ASR_MAX_RESIDENT", 3)
ASR_DEVICE = os.environ.get("ASR_DEVICE", "")
# Streaming endpointing (energy VAD; silero is used instead when installed).
VAD_FRAME_MS = _int("VAD_FRAME_MS", 20)
VAD_THRESHOLD = _float("VAD_THRESHOLD", 0.015)      # RMS of float PCM in [-1, 1]
VAD_SILENCE_MS = _int("VAD_SILENCE_MS", 700)        # silence that ends a segment
VAD_MIN_SPEECH_MS = _int("VAD_MIN_SPEECH_MS", 250)  # shorter bursts are noise
ASR_PARTIAL_EVERY_S = _float("ASR_PARTIAL_EVERY_S", 1.5)
ASR_MAX_SEGMENT_S = _float("ASR_MAX_SEGMENT_S", 25.0)   # Whisper's window is 30 s
ASR_MAX_AUDIO_S = _float("ASR_MAX_AUDIO_S", 600.0)      # per connection
ASR_MAX_UPLOAD_BYTES = _int("ASR_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)

# --------------------------------------------------------------- concurrency
MAX_CONNS = _int("MAX_CONNS", 16)
MAX_INFLIGHT = _int("MAX_INFLIGHT", 2)       # synthesis jobs at once on the GPU
QUEUE_WAIT_S = _float("QUEUE_WAIT_S", 15.0)
MAX_FRAME_BYTES = _int("MAX_FRAME_BYTES", 8192)   # text frames; audio frames are separate
MAX_AUDIO_FRAME_BYTES = _int("MAX_AUDIO_FRAME_BYTES", 65536)
MAX_TOTAL_BYTES = _int("MAX_TOTAL_BYTES", SR * BYTES_PER_SAMPLE * 120)
SEND_TIMEOUT_S = _float("SEND_TIMEOUT_S", 30.0)
FIRST_FRAME_TIMEOUT_S = _float("FIRST_FRAME_TIMEOUT_S", 30.0)
IDLE_TIMEOUT_S = _float("IDLE_TIMEOUT_S", 60.0)
CONN_MAX_SECONDS = _float("CONN_MAX_SECONDS", 300.0)
ASR_CONN_MAX_SECONDS = _float("ASR_CONN_MAX_SECONDS", 900.0)

# ---------------------------------------------------------------------- auth
# The bearer every /v1 call must carry, and that mints WebSocket tickets.
# nyo-node hands it in as NYO_SERVING_KEY; a hand-run server may set
# SPEECH_API_KEY or pass --api-key. Empty means open, like the first demo.
API_KEY = (os.environ.get("SPEECH_API_KEY") or os.environ.get("NYO_SERVING_KEY") or "").strip()
# Whether a WebSocket may connect with no ticket and no key when a key is set.
WS_PUBLIC = _bool("SPEECH_WS_PUBLIC", False)
TICKET_TTL_S = _int("TICKET_TTL_S", 600)

LOG_TEXT = _bool("LOG_TEXT", False)
