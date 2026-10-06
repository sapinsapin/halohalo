"""
Short-lived WebSocket tickets, so NYO can hand a browser (or a device) a
socket URL without ever giving it the serving key.

The gateway holds the serving key and calls `POST /v1/tickets`; this module
mints `exp.nonce.sig` where sig = HMAC-SHA256(key, "exp.nonce.scope"). The
browser connects to `/tts?ticket=...`; the server checks the signature and
the expiry with the same key. Nothing is stored, so tickets survive a
restart and cost no database. A ticket is bearer for its lifetime (default
ten minutes); losing one lets someone synthesize speech on your GPU for ten
minutes, which is the price of not proxying every PCM frame through NYO.

Pure Python; tested without the server.
"""

import hashlib
import hmac
import secrets
import time

SCOPES = ("tts", "asr", "speech")   # speech = both


def mint(key: str, ttl_s: int, scope: str = "speech", now: float | None = None) -> str:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    exp = int((now if now is not None else time.time()) + ttl_s)
    nonce = secrets.token_urlsafe(12)
    return f"{exp}.{nonce}.{scope}.{_sign(key, exp, nonce, scope)}"


def verify(key: str, ticket: str | None, want: str, now: float | None = None) -> bool:
    """True when the ticket is well formed, unexpired, signed with `key`, and
    its scope covers `want` (tts or asr)."""
    if not key or not ticket:
        return False
    parts = ticket.split(".")
    if len(parts) != 4:
        return False
    exp_s, nonce, scope, sig = parts
    if not exp_s.isdigit() or scope not in SCOPES:
        return False
    if scope != "speech" and scope != want:
        return False
    exp = int(exp_s)
    if exp < (now if now is not None else time.time()):
        return False
    return hmac.compare_digest(sig, _sign(key, exp, nonce, scope))


def expires_at(ticket: str) -> int | None:
    head = ticket.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _sign(key: str, exp: int, nonce: str, scope: str) -> str:
    msg = f"{exp}.{nonce}.{scope}".encode()
    return hmac.new(key.encode(), msg, hashlib.sha256).hexdigest()[:32]
