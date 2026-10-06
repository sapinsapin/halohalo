"""
The wire contracts, isolated so they are hard to change by accident.

TTS (`/tts`) is serving/tts-ws's contract, byte for byte: embedded firmware
compares the terminal frames literally, so even whitespace is a breaking
change. The frames are asserted at import.

ASR (`/asr`) is new here, and shaped after the TTS one so a device that
already speaks it has nothing new to learn: JSON text frames carry control
and results, binary frames carry PCM S16LE 16 kHz mono, and every request
ends in exactly one `end` or `error`.

    client -> {"type":"start","lang":"ceb"}         optional; defaults apply
    client -> <binary frames>                       PCM S16LE, 16 kHz, mono
    server -> {"type":"partial","text":"..."}       while a phrase is being spoken
    server -> {"type":"final","text":"...","start":1.2,"end":3.4}
    client -> {"type":"stop"}                       or just close
    server -> {"type":"end"}                        after the last final
       or  -> {"type":"error","message":"..."}
"""

import json

from settings import MAX_FRAME_BYTES, MAX_TEXT_CHARS

# Frozen. Do not reformat, do not add fields.
END = b'{"type":"end"}'
assert END == json.dumps({"type": "end"}, separators=(",", ":")).encode()


def frame(obj: dict) -> bytes:
    """A compact JSON text frame; the only shape the server ever sends."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def error(message: str) -> bytes:
    return frame({"type": "error", "message": message})


# Every message the partner can receive, so their doc can be exhaustive.
ERR_BAD_JSON = "invalid JSON"
ERR_NOT_OBJECT = "expected a JSON object"
ERR_NO_TEXT = "missing 'text'"
ERR_EMPTY_TEXT = "'text' is empty"
ERR_TOO_LONG = f"'text' exceeds {MAX_TEXT_CHARS} characters"
ERR_FRAME_TOO_BIG = f"message exceeds {MAX_FRAME_BYTES} bytes"
ERR_BUSY = "server busy, retry"
ERR_NOT_READY = "model still loading, retry"
ERR_UNAUTHORIZED = "unauthorized"
ERR_NO_AUDIO = "synthesis produced no audio"
ERR_SYNTH_FAILED = "synthesis failed"
ERR_UNKNOWN_LANG = "unknown 'lang'"          # suffixed with the served list
# ASR
ERR_AUDIO_FRAME_TOO_BIG = "audio frame too large"
ERR_ODD_FRAME = "audio frame has an odd byte length"
ERR_TOO_MUCH_AUDIO = "connection audio limit reached"
ERR_UNKNOWN_TYPE = "unknown 'type'"
ERR_ASR_FAILED = "transcription failed"


class BadRequest(Exception):
    """Client-side problem: send one error frame, then stop."""


class Request:
    __slots__ = ("text", "lang", "voice", "meta")

    def __init__(self, text: str, lang, voice, meta: bool):
        self.text, self.lang, self.voice, self.meta = text, lang, voice, meta


def _load_object(raw: str | None) -> dict:
    if raw is None:
        raise BadRequest(ERR_BAD_JSON)
    if len(raw.encode("utf-8", "ignore")) > MAX_FRAME_BYTES:
        raise BadRequest(ERR_FRAME_TOO_BIG)
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        raise BadRequest(ERR_BAD_JSON) from None
    if not isinstance(obj, dict):
        raise BadRequest(ERR_NOT_OBJECT)
    return obj


def parse(raw: str | None) -> Request:
    """Validate one TTS request. Liberal about unknown fields, strict about shape."""
    obj = _load_object(raw)
    if "text" not in obj:
        raise BadRequest(ERR_NO_TEXT)
    text = obj["text"]
    if not isinstance(text, str):
        raise BadRequest(ERR_NO_TEXT)
    text = text.strip()
    if not text:
        raise BadRequest(ERR_EMPTY_TEXT)
    if len(text) > MAX_TEXT_CHARS:
        raise BadRequest(ERR_TOO_LONG)
    return Request(text=text, lang=obj.get("lang"), voice=obj.get("voice"),
                   meta=bool(obj.get("meta", False)))


class AsrControl:
    """One ASR control frame: start (with options) or stop."""

    __slots__ = ("kind", "lang", "sample_rate", "partials")

    def __init__(self, kind: str, lang=None, sample_rate: int = 16000,
                 partials: bool = True):
        self.kind, self.lang, self.sample_rate, self.partials = kind, lang, sample_rate, partials


def parse_asr_control(raw: str | None) -> AsrControl:
    """Validate one ASR text frame.

    `start` may name a `lang`, a `sample_rate` (16000 or 48000; anything else
    is an error so a device never gets a silently-wrong transcript) and
    `partials: false` to receive finals only. `stop` ends the request.
    """
    obj = _load_object(raw)
    kind = obj.get("type")
    if kind == "stop":
        return AsrControl("stop")
    if kind in ("start", None):
        sr = obj.get("sample_rate", 16000)
        if sr not in (16000, 48000, 8000):
            raise BadRequest("unsupported 'sample_rate'; use 16000, 48000 or 8000")
        return AsrControl("start", lang=obj.get("lang"), sample_rate=int(sr),
                          partials=bool(obj.get("partials", True)))
    raise BadRequest(f"{ERR_UNKNOWN_TYPE}: {kind!r}")
