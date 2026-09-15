"""
Demo WebSocket TTS endpoint for the Agora hardware integration.

  ws(s)://<host>/tts
    client -> {"text": "..."}                       one JSON frame
    server -> raw binary frames                     PCM S16LE, 16 kHz, mono, no header
    server -> {"type":"end"}                        after the last audio frame
       or  -> {"type":"error","message":"..."}       on failure

See docs/AGORA_INTEGRATION.md for the partner-facing contract and
settings.py for every limit.
"""

import asyncio
import contextlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

import protocol as P
import settings as S
import tts
from textseg import segment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("tts-ws")

# One worker on purpose. asyncio's default executor is min(32, cpu+4) threads,
# and several concurrent torch forward passes on 2 vCPU only thrash. A single
# worker also makes the first real request queue behind warmup for free.
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="synth")
SEM = asyncio.Semaphore(S.MAX_INFLIGHT)

_state = {"ready": False, "error": None, "conns": 0}


# --------------------------------------------------------------------- warmup

def _warm_all() -> None:
    t0 = time.perf_counter()
    try:
        for lang in tts.served_langs():
            tts.warm(lang)
            log.info("warm: %s ready", lang)
        _state["ready"] = True
        log.info("ready in %.1fs (langs=%s)",
                 time.perf_counter() - t0, ",".join(tts.served_langs()))
    except Exception as exc:
        _state["error"] = type(exc).__name__
        log.exception("warmup failed")


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    if not tts.served_langs():
        log.error("SERVE_LANGS=%s matches nothing in assets.json", S.SERVE_LANGS)
    # Warm in the background: the platform health check must be able to get a
    # 200 from GET / within seconds, long before the weights are resident.
    asyncio.get_running_loop().run_in_executor(POOL, _warm_all)
    yield
    POOL.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="SapinSapin TTS WebSocket demo", lifespan=lifespan)


# ----------------------------------------------------------------- http routes

@app.get("/healthz")
async def healthz():
    body = {"ready": _state["ready"], "error": _state["error"],
            "connections": _state["conns"], "langs": list(tts.served_langs())}
    return JSONResponse(body, status_code=200 if _state["ready"] else 503)


@app.get("/", response_class=PlainTextResponse)
async def index():
    return (
        "SapinSapin TTS WebSocket demo\n\n"
        f"  endpoint   /tts\n"
        f"  request    {{\"text\": \"...\"}}  (optional: lang, voice)\n"
        f"  audio      PCM S16LE, {S.SR} Hz, mono, no WAV header, "
        f"{S.CHUNK_BYTES}-byte frames\n"
        f"  terminal   {P.END.decode()}  or  "
        '{"type":"error","message":"..."}\n'
        f"  max text   {S.MAX_TEXT_CHARS} characters\n"
        f"  languages  {json.dumps(tts.voice_catalog(), ensure_ascii=False)}\n"
        f"  ready      {_state['ready']}\n\n"
        "See docs/AGORA_INTEGRATION.md in this repo.\n"
    )


# ------------------------------------------------------------- socket plumbing

async def _send_bytes(ws: WebSocket, payload: bytes) -> bool:
    """Send one frame. False means the peer is gone — stop, do not report."""
    if ws.client_state is not WebSocketState.CONNECTED:
        return False
    try:
        await asyncio.wait_for(ws.send_bytes(payload), S.SEND_TIMEOUT_S)
        return True
    except (WebSocketDisconnect, ConnectionError, asyncio.TimeoutError):
        return False
    except RuntimeError:
        # Starlette: 'Cannot call "send" once a close message has been sent.'
        return False


async def _send_json_frame(ws: WebSocket, payload: bytes) -> bool:
    if ws.client_state is not WebSocketState.CONNECTED:
        return False
    try:
        await asyncio.wait_for(ws.send_text(payload.decode("utf-8")),
                               S.SEND_TIMEOUT_S)
        return True
    except (WebSocketDisconnect, ConnectionError, asyncio.TimeoutError, RuntimeError):
        return False


async def _close(ws: WebSocket, code: int = 1000) -> None:
    # Always 1000. Embedded stacks surface close codes inconsistently, so
    # meaning lives in the JSON frame; a second close() would raise.
    if ws.application_state is not WebSocketState.DISCONNECTED:
        with contextlib.suppress(RuntimeError, ConnectionError, WebSocketDisconnect):
            await ws.close(code)


async def _reader(ws: WebSocket, queue: asyncio.Queue, cancel: threading.Event):
    """Keep a receive() outstanding for the whole connection.

    This task is the only way to notice a client close: send-only code learns
    about it late or never, because the kernel accepts writes long after the
    peer is gone. Starlette forbids two concurrent receive() calls, so
    everything funnels through here.
    """
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                cancel.set()
                await queue.put(None)
                return
            # Liberal about the opcode: some embedded stacks send a text
            # payload with the binary opcode. receive_text() would raise.
            payload = msg.get("text")
            if payload is None and msg.get("bytes") is not None:
                payload = msg["bytes"].decode("utf-8", "replace")
            await queue.put(payload)
    except (WebSocketDisconnect, RuntimeError, ConnectionError):
        cancel.set()
        await queue.put(None)
    except asyncio.CancelledError:
        raise


# -------------------------------------------------------------------- synthesis

async def _serve_one(ws: WebSocket, req: P.Request, cancel: threading.Event) -> bool:
    """Handle one request. False means: stop, the connection is finished."""
    lang = tts.resolve_lang(req.lang, S.DEFAULT_LANG)
    if lang is None:
        await _send_json_frame(ws, P.error(
            f"{P.ERR_UNKNOWN_LANG}; served: {', '.join(tts.served_langs())}"))
        return False
    if not _state["ready"]:
        await _send_json_frame(ws, P.error(
            P.ERR_SYNTH_FAILED if _state["error"] else P.ERR_NOT_READY))
        return False

    voice = tts.resolve_voice(lang, req.voice)
    segs = segment(req.text)[:S.MAX_SEGMENTS]
    if not segs:
        await _send_json_frame(ws, P.error(P.ERR_EMPTY_TEXT))
        return False

    try:
        await asyncio.wait_for(SEM.acquire(), S.QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        await _send_json_frame(ws, P.error(P.ERR_BUSY))
        return False

    loop = asyncio.get_running_loop()
    started, sent_bytes = time.perf_counter(), 0
    first_audio_at = None
    try:
        if req.meta:
            # Opt-in only: a conforming client that expects audio as the first
            # frame after its request must never receive anything else.
            await _send_json_frame(ws, json.dumps(
                {"type": "start", "sample_rate": S.SR, "channels": 1,
                 "format": "pcm_s16le", "lang": lang, "voice": voice["id"],
                 "segments": len(segs)}, separators=(",", ":")).encode())

        if S.LOG_TEXT:
            log.debug("synth lang=%s voice=%s text=%r",
                      lang, voice["id"], req.text[:120])
        dropped = tts.dropped_chars(req.text, lang)
        if dropped:
            # Not corrected: SpeechT5Tokenizer discards what it cannot encode,
            # so digits and '₱' vanish from the audio. Numbers must be written
            # as words. Surfaced in the log so a silent gap is explainable.
            log.warning("tokenizer dropped %r — write numbers as words", dropped)

        for i, seg in enumerate(segs):
            if cancel.is_set():
                log.info("cancelled before segment %d/%d", i + 1, len(segs))
                return False
            t_seg = time.perf_counter()
            pcm = await loop.run_in_executor(
                POOL, tts.synth_segment, seg, lang, voice, cancel)
            log.debug("segment %d/%d synthesized in %.2fs (%d bytes)",
                      i + 1, len(segs), time.perf_counter() - t_seg,
                      len(pcm or b""))
            if cancel.is_set():
                log.info("cancelled after segment %d/%d", i + 1, len(segs))
                return False
            if not pcm:
                continue
            if first_audio_at is None:
                first_audio_at = time.perf_counter() - started
            if i and not await _stream(ws, tts.silence(S.GAP_MS), cancel):
                return False
            t_send = time.perf_counter()
            if not await _stream(ws, pcm, cancel):
                return False
            log.debug("segment %d/%d streamed in %.2fs",
                      i + 1, len(segs), time.perf_counter() - t_send)
            sent_bytes += len(pcm)
            if sent_bytes >= S.MAX_TOTAL_BYTES:
                log.warning("hit MAX_TOTAL_BYTES after segment %d", i + 1)
                break

        if not sent_bytes:
            await _send_json_frame(ws, P.error(P.ERR_NO_AUDIO))
            return False
        # Exactly one terminal frame per request, and end implies audio.
        if not await _send_json_frame(ws, P.END):
            log.info("client gone before end (%d bytes sent)", sent_bytes)
            return False
        audio_s = sent_bytes / (S.SR * S.BYTES_PER_SAMPLE)
        elapsed = time.perf_counter() - started
        log.info("ok lang=%s segs=%d %.2fs audio in %.2fs (rtf %.2f) ttfa %.2fs",
                 lang, len(segs), audio_s, elapsed, elapsed / max(audio_s, 1e-9),
                 first_audio_at or 0.0)
        return True
    except asyncio.CancelledError:
        cancel.set()
        raise
    except Exception as exc:
        # Type name only: this endpoint is public, and the exception text would
        # carry filesystem paths and repo names.
        log.exception("synthesis failed")
        await _send_json_frame(ws, P.error(
            f"{P.ERR_SYNTH_FAILED}: {type(exc).__name__}"))
        return False
    finally:
        SEM.release()


async def _stream(ws: WebSocket, pcm: bytes, cancel: threading.Event) -> bool:
    """Chop PCM into frames and send them."""
    step = S.CHUNK_BYTES
    pace = step / (S.SR * S.BYTES_PER_SAMPLE) * S.PACE_FACTOR
    for off in range(0, len(pcm), step):
        if cancel.is_set():
            return False
        if not await _send_bytes(ws, pcm[off:off + step]):
            log.info("send failed at byte %d — peer gone", off)
            return False
        # Yield to the event loop after every frame. Without this the send
        # loop starves _reader: awaiting a send that does not block does not
        # suspend, so the client's CLOSE frame sits unread and cancellation
        # only fires when the peer gives up (measured: 10s late, which is the
        # client's close timeout, not anything we chose).
        await asyncio.sleep(pace or 0)
    return True


# ------------------------------------------------------------------- endpoint

@app.websocket("/tts")
async def tts_ws(ws: WebSocket):
    # Admission control accepts first, then reports. ASGI cannot send before
    # accept, and a pre-accept close surfaces as an HTTP handshake failure that
    # minimal firmware often cannot report at all.
    await ws.accept()

    if S.TTS_TOKEN and ws.query_params.get("key") != S.TTS_TOKEN:
        await _send_json_frame(ws, P.error(P.ERR_UNAUTHORIZED))
        await _close(ws)
        return
    if _state["conns"] >= S.MAX_CONNS:
        await _send_json_frame(ws, P.error(P.ERR_BUSY))
        await _close(ws)
        return

    _state["conns"] += 1
    peer = ws.headers.get("x-forwarded-for", "") or (
        ws.client.host if ws.client else "?")
    log.info("open %s (%d/%d)", peer, _state["conns"], S.MAX_CONNS)

    cancel = threading.Event()
    queue: asyncio.Queue = asyncio.Queue(maxsize=4)
    reader = asyncio.create_task(_reader(ws, queue, cancel))
    deadline = time.monotonic() + S.CONN_MAX_SECONDS
    served = 0
    try:
        while True:
            # Tighter window before the first request kills parked sockets.
            budget = (S.FIRST_FRAME_TIMEOUT_S if served == 0 else S.IDLE_TIMEOUT_S)
            budget = min(budget, max(0.0, deadline - time.monotonic()))
            if budget <= 0:
                break
            try:
                raw = await asyncio.wait_for(queue.get(), budget)
            except asyncio.TimeoutError:
                break
            if raw is None:            # client closed
                break
            try:
                req = P.parse(raw)
            except P.BadRequest as exc:
                await _send_json_frame(ws, P.error(str(exc)))
                break
            served += 1
            # One request in flight per connection: the contract has no
            # correlation id, so pipelined audio would be ambiguous.
            if not await _serve_one(ws, req, cancel):
                break
    except asyncio.CancelledError:
        cancel.set()
        raise
    finally:
        cancel.set()
        reader.cancel()
        with contextlib.suppress(BaseException):
            await reader
        await _close(ws)
        _state["conns"] -= 1
        log.info("close %s (served %d)", peer, served)
