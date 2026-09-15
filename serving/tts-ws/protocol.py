"""
The wire contract, isolated so it is hard to change by accident.

The two terminal frames are frozen byte strings. Embedded firmware compares
them literally, so adding a field — or even whitespace — is a breaking change
for the partner, not a cosmetic edit. They are asserted at import.
"""

import json

from settings import MAX_FRAME_BYTES, MAX_TEXT_CHARS

# Frozen. Do not reformat, do not add fields.
END = b'{"type":"end"}'
assert END == json.dumps({"type": "end"}, separators=(",", ":")).encode()


def error(message: str) -> bytes:
    return json.dumps({"type": "error", "message": message},
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")


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


class BadRequest(Exception):
    """Client-side problem: send one error frame, then stop."""


class Request:
    __slots__ = ("text", "lang", "voice", "meta")

    def __init__(self, text: str, lang, voice, meta: bool):
        self.text, self.lang, self.voice, self.meta = text, lang, voice, meta


def parse(raw: str | None) -> Request:
    """Validate one client message.

    Liberal about extra and unknown fields — a firmware author adding a field
    should not get an error — but strict about malformed JSON, because
    accepting a bare string would commit us to supporting a non-JSON client
    forever.
    """
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
    if "text" not in obj:
        raise BadRequest(ERR_NO_TEXT)

    text = obj["text"]
    if not isinstance(text, str):
        raise BadRequest(ERR_NO_TEXT)
    text = text.strip()
    if not text:
        raise BadRequest(ERR_EMPTY_TEXT)
    # Counted in characters, matching the message the partner is shown.
    if len(text) > MAX_TEXT_CHARS:
        raise BadRequest(ERR_TOO_LONG)

    return Request(text=text,
                   lang=obj.get("lang"),
                   voice=obj.get("voice"),
                   meta=bool(obj.get("meta", False)))
