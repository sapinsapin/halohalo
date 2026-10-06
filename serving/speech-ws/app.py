"""
The GB10 speech server: streaming text-to-speech and speech recognition for
the Philippine languages, on the wire contract the Agora hardware
integration uses, plus OpenAI-shaped HTTP for NYO's gateway.

  ws(s)://<host>/tts     serving/tts-ws's contract, unchanged (docs/AGORA_INTEGRATION.md there)
  ws(s)://<host>/asr     streaming recognition (docs/ASR_INTEGRATION.md here)
  POST /v1/audio/speech            {"input","voice","language","response_format":"pcm"|"wav"}
  POST /v1/audio/transcriptions    multipart file (+ language) -> {"text": ...}
  GET  /v1/models  /v1/voices      what is served
  POST /v1/tickets                 short-lived socket tickets, for the gateway
  GET  /health  /healthz  /

Auth: `/v1/*` needs the serving key as a bearer when one is configured
(nyo-node sets NYO_SERVING_KEY). Sockets take `?ticket=` from /v1/tickets,
or `?key=`, or are open when no key is set / SPEECH_WS_PUBLIC=1.
"""

import asyncio
import contextlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Iterator

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

import protocol as P
import settings as S
import tickets
from asr_stream import Endpointer
from audio_io import UnsupportedAudio, decode_to_16k
from engines import get_asr, get_tts
from pcm import f32_to_pcm16, frames_of, pcm16_to_f32, resample_to_16k, silence, wav_bytes
from textseg import segment
from voices import catalog, resolve_lang, resolve_voice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("speech-ws")

# GPU work runs here, off the event loop. Two workers: one synthesis and one
# transcription can overlap; the engines serialise their own model access.
POOL = ThreadPoolExecutor(max_workers=max(2, S.MAX_INFLIGHT + 1), thread_name_prefix="speech")
SEM = asyncio.Semaphore(S.MAX_INFLIGHT)

_state = {"ready": False, "error": None, "conns": 0, "started": time.time(),
          "tts": None, "asr": None}


# --------------------------------------------------------------------- warmup

def _warm_all() -> None:
    t0 = time.perf_counter()
    try:
        tts, asr = get_tts(), get_asr()
        if tts is not None:
            for lang in tts.served_langs():
                tts.warm(lang)
                log.info("tts warm: %s", lang)
            _state["tts"] = tts.info()
        if asr is not None:
            for lang in asr.served_langs()[:max(1, S.ASR_MAX_RESIDENT)]:
                asr.warm(lang)
                log.info("asr warm: %s", lang)
            _state["asr"] = asr.info()
        _state["ready"] = True
        log.info("ready in %.1fs", time.perf_counter() - t0)
    except Exception as exc:                 # noqa: BLE001 - surfaced on /health
        _state["error"] = type(exc).__name__
        log.exception("warmup failed")


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    asyncio.get_running_loop().run_in_executor(POOL, _warm_all)
    yield
    POOL.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="SapinSapin speech (GB10)", lifespan=lifespan)


def _tts_langs() -> tuple[str, ...]:
    tts = get_tts()
    return tts.served_langs() if tts else ()


def _asr_langs() -> tuple[str, ...]:
    asr = get_asr()
    return asr.served_langs() if asr else ()


# ----------------------------------------------------------------------- auth

def _bearer(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _http_authorized(request: Request) -> bool:
    return not S.API_KEY or _bearer(request) == S.API_KEY


def _ws_authorized(ws: WebSocket, want: str) -> bool:
    if not S.API_KEY or S.WS_PUBLIC:
        return True
    key = ws.query_params.get("key") or ""
    auth = ws.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        key = key or auth[7:].strip()
    if key and key == S.API_KEY:
        return True
    return tickets.verify(S.API_KEY, ws.query_params.get("ticket"), want)


def _unauthorized():
    return JSONResponse({"error": {"message": "unauthorized", "type": "auth"}}, status_code=401)


# ---------------------------------------------------------------- http: info

@app.get("/health")
async def health():
    """nyo-node's probes: 200 only once the models are resident."""
    body = {"ready": _state["ready"], "error": _state["error"], "model": S.SERVED_NAME}
    return JSONResponse(body, status_code=200 if _state["ready"] else 503)


@app.get("/healthz")
async def healthz():
    body = {"ready": _state["ready"], "error": _state["error"], "connections": _state["conns"],
            "langs": list(_tts_langs()), "asr_langs": list(_asr_langs()),
            "uptime_s": round(time.time() - _state["started"])}
    return JSONResponse(body, status_code=200 if _state["ready"] else 503)


@app.get("/", response_class=PlainTextResponse)
async def index():
    return (
        f"SapinSapin speech server ({S.SERVED_NAME})\n\n"
        f"  tts        ws /tts   {{\"text\": \"...\"}}  (optional: lang, voice)  -> PCM S16LE {S.SR} Hz mono, {S.CHUNK_BYTES}-byte frames, then {P.END.decode()}\n"
        f"  asr        ws /asr   PCM S16LE {S.SR} Hz mono frames in -> partial/final JSON, then {P.END.decode()}\n"
        f"  http       POST /v1/audio/speech · POST /v1/audio/transcriptions · GET /v1/models · GET /v1/voices\n"
        f"  languages  tts {','.join(_tts_langs()) or '-'} · asr {','.join(_asr_langs()) or '-'}\n"
        f"  ready      {_state['ready']}\n"
    )


@app.get("/v1/models")
async def models(request: Request):
    if not _http_authorized(request):
        return _unauthorized()
    tts, asr = get_tts(), get_asr()
    return {"object": "list", "data": [{
        "id": S.SERVED_NAME, "object": "model", "owned_by": "sapinsapin", "type": "speech",
        "created": int(_state["started"]),
        "capabilities": {"tts": list(_tts_langs()), "asr": list(_asr_langs()),
                         "streaming": True, "sample_rate": S.SR, "format": "pcm_s16le"},
        "engines": {"tts": tts.info() if tts else None, "asr": asr.info() if asr else None},
        "ready": _state["ready"],
    }]}


@app.get("/v1/voices")
async def voices_route(request: Request):
    if not _http_authorized(request):
        return _unauthorized()
    return {"default_lang": S.DEFAULT_LANG if S.DEFAULT_LANG in _tts_langs() else (_tts_langs() or [None])[0],
            "languages": catalog(_tts_langs()), "asr_languages": list(_asr_langs()),
            "max_text_chars": S.MAX_TEXT_CHARS, "sample_rate": S.SR}


@app.post("/v1/tickets")
async def mint_ticket(request: Request):
    """The gateway holds the serving key; a browser never does. It asks for a
    ticket here and hands the browser a socket URL carrying it."""
    if not S.API_KEY:
        return JSONResponse({"error": {"message": "no serving key configured; the sockets are open"}}, 400)
    if not _http_authorized(request):
        return _unauthorized()
    body = {}
    with contextlib.suppress(Exception):
        body = await request.json()
    scope = body.get("scope", "speech") if isinstance(body, dict) else "speech"
    ttl = body.get("ttl_s", S.TICKET_TTL_S) if isinstance(body, dict) else S.TICKET_TTL_S
    try:
        ttl = max(30, min(int(ttl), 3600))
        t = tickets.mint(S.API_KEY, ttl, scope)
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": {"message": str(exc)}}, 400)
    return {"ticket": t, "expires_at": tickets.expires_at(t), "scope": scope,
            "tts_path": f"/tts?ticket={t}", "asr_path": f"/asr?ticket={t}"}


# ------------------------------------------------------------ http: speech

def _tts_or_503():
    if get_tts() is None:
        return JSONResponse({"error": {"message": "text-to-speech is not enabled on this server"}}, 503)
    if not _state["ready"]:
        return JSONResponse({"error": {"message": P.ERR_SYNTH_FAILED if _state["error"] else P.ERR_NOT_READY}}, 503)
    return None


def _synth_segments(text: str, lang: str, voice: dict, cancel: threading.Event) -> Iterator[bytes]:
    """Synchronous: every PCM chunk of every segment, with the inter-segment gap."""
    tts = get_tts()
    segs = segment(text)[:S.MAX_SEGMENTS]
    for i, seg in enumerate(segs):
        if cancel.is_set():
            return
        if i:
            yield silence(S.GAP_MS)
        yield from tts.synth_stream(seg, lang, voice, cancel)


async def _aiter_in_thread(gen: Iterator[bytes]) -> AsyncIterator[bytes]:
    """Drive a blocking generator on the pool and hand its chunks to the loop."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    done = object()

    def pump():
        try:
            for chunk in gen:
                loop.call_soon_threadsafe(q.put_nowait, chunk)
            loop.call_soon_threadsafe(q.put_nowait, done)
        except BaseException as exc:          # noqa: BLE001 - re-raised on the loop
            loop.call_soon_threadsafe(q.put_nowait, exc)

    POOL.submit(pump)
    while True:
        item = await q.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


@app.post("/v1/audio/speech")
async def speech(request: Request):
    """OpenAI-shaped synthesis. `response_format` pcm streams raw PCM S16LE
    16 kHz frames as they are made (chunked); wav returns a file."""
    if not _http_authorized(request):
        return _unauthorized()
    bad = _tts_or_503()
    if bad:
        return bad
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": {"message": P.ERR_BAD_JSON}}, 400)
    if not isinstance(body, dict):
        return JSONResponse({"error": {"message": P.ERR_NOT_OBJECT}}, 400)
    text = body.get("input", body.get("text"))
    if not isinstance(text, str) or not text.strip():
        return JSONResponse({"error": {"message": "'input' is required"}}, 400)
    text = text.strip()
    if len(text) > S.MAX_TEXT_CHARS:
        return JSONResponse({"error": {"message": P.ERR_TOO_LONG}}, 400)
    lang = resolve_lang(body.get("language", body.get("lang")), _tts_langs(), S.DEFAULT_LANG)
    if lang is None:
        return JSONResponse({"error": {"message": f"{P.ERR_UNKNOWN_LANG}; served: {', '.join(_tts_langs())}"}}, 400)
    voice = resolve_voice(lang, body.get("voice"))
    fmt = str(body.get("response_format", "pcm")).lower()
    if fmt not in ("pcm", "wav"):
        return JSONResponse({"error": {"message": "response_format must be pcm or wav"}}, 400)

    try:
        await asyncio.wait_for(SEM.acquire(), S.QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        return JSONResponse({"error": {"message": P.ERR_BUSY}}, 503)

    cancel = threading.Event()
    started = time.perf_counter()
    headers = {"x-sample-rate": str(S.SR), "x-channels": "1", "x-format": "pcm_s16le",
               "x-lang": lang, "x-voice": voice["id"], "cache-control": "no-store"}

    if fmt == "wav":
        try:
            pcm = b"".join([c async for c in _aiter_in_thread(_synth_segments(text, lang, voice, cancel))])
        except Exception as exc:              # noqa: BLE001
            log.exception("synthesis failed")
            return JSONResponse({"error": {"message": f"{P.ERR_SYNTH_FAILED}: {type(exc).__name__}"}}, 500)
        finally:
            SEM.release()
        if not pcm:
            return JSONResponse({"error": {"message": P.ERR_NO_AUDIO}}, 500)
        headers["x-audio-seconds"] = f"{len(pcm) / (S.SR * S.BYTES_PER_SAMPLE):.2f}"
        headers["x-elapsed-ms"] = str(int((time.perf_counter() - started) * 1000))
        return Response(wav_bytes(pcm), media_type="audio/wav", headers=headers)

    async def stream():
        sent = 0
        try:
            async for chunk in _aiter_in_thread(_synth_segments(text, lang, voice, cancel)):
                for fr in frames_of(chunk, S.CHUNK_BYTES):
                    sent += len(fr)
                    yield fr
                    if S.PACE_FACTOR:
                        await asyncio.sleep(S.CHUNK_BYTES / (S.SR * S.BYTES_PER_SAMPLE) * S.PACE_FACTOR)
            log.info("http speech ok lang=%s %.2fs audio in %.2fs", lang,
                     sent / (S.SR * S.BYTES_PER_SAMPLE), time.perf_counter() - started)
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception:                     # noqa: BLE001 - the stream has started; log only
            log.exception("synthesis failed mid-stream")
        finally:
            cancel.set()
            SEM.release()

    return StreamingResponse(stream(), media_type="audio/pcm", headers=headers)


# --------------------------------------------------------- http: transcribe

@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request, file: UploadFile = File(...),
                         language: str | None = Form(None), lang: str | None = Form(None),
                         response_format: str = Form("json"), model: str | None = Form(None)):
    if not _http_authorized(request):
        return _unauthorized()
    asr = get_asr()
    if asr is None:
        return JSONResponse({"error": {"message": "speech recognition is not enabled on this server"}}, 503)
    if not _state["ready"]:
        return JSONResponse({"error": {"message": P.ERR_NOT_READY}}, 503)
    code = resolve_lang(language or lang, _asr_langs(), S.DEFAULT_LANG)
    if code is None:
        return JSONResponse({"error": {"message": f"{P.ERR_UNKNOWN_LANG}; served: {', '.join(_asr_langs())}"}}, 400)
    data = await file.read(S.ASR_MAX_UPLOAD_BYTES + 1)
    if len(data) > S.ASR_MAX_UPLOAD_BYTES:
        return JSONResponse({"error": {"message": "file too large"}}, 413)
    loop = asyncio.get_running_loop()
    try:
        audio = await loop.run_in_executor(POOL, decode_to_16k, data, file.content_type)
    except UnsupportedAudio as exc:
        return JSONResponse({"error": {"message": str(exc)}}, 400)
    if audio.size > S.ASR_MAX_AUDIO_S * S.SR:
        return JSONResponse({"error": {"message": f"audio longer than {S.ASR_MAX_AUDIO_S:.0f} s"}}, 413)
    started = time.perf_counter()
    try:
        text = await loop.run_in_executor(POOL, asr.transcribe, audio, code)
    except Exception as exc:                  # noqa: BLE001
        log.exception("transcription failed")
        return JSONResponse({"error": {"message": f"{P.ERR_ASR_FAILED}: {type(exc).__name__}"}}, 500)
    elapsed = time.perf_counter() - started
    seconds = audio.size / S.SR
    log.info("http transcribe ok lang=%s %.1fs audio in %.2fs", code, seconds, elapsed)
    if response_format == "text":
        return PlainTextResponse(text)
    body = {"text": text, "language": code, "model": S.SERVED_NAME}
    if response_format == "verbose_json":
        body.update({"duration": round(seconds, 3), "elapsed_ms": int(elapsed * 1000),
                     "engine": asr.info()})
    return body


# ------------------------------------------------------------- socket plumbing

async def _send_bytes(ws: WebSocket, payload: bytes) -> bool:
    if ws.client_state is not WebSocketState.CONNECTED:
        return False
    try:
        await asyncio.wait_for(ws.send_bytes(payload), S.SEND_TIMEOUT_S)
        return True
    except (WebSocketDisconnect, ConnectionError, asyncio.TimeoutError, RuntimeError):
        return False


async def _send_json_frame(ws: WebSocket, payload: bytes) -> bool:
    if ws.client_state is not WebSocketState.CONNECTED:
        return False
    try:
        await asyncio.wait_for(ws.send_text(payload.decode("utf-8")), S.SEND_TIMEOUT_S)
        return True
    except (WebSocketDisconnect, ConnectionError, asyncio.TimeoutError, RuntimeError):
        return False


async def _close(ws: WebSocket, code: int = 1000) -> None:
    # Always 1000: meaning lives in the JSON frame, not the close code.
    if ws.application_state is not WebSocketState.DISCONNECTED:
        with contextlib.suppress(RuntimeError, ConnectionError, WebSocketDisconnect):
            await ws.close(code)


async def _reader(ws: WebSocket, queue: asyncio.Queue, cancel: threading.Event):
    """The one outstanding receive(): the only way to notice a client close."""
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                cancel.set()
                await queue.put(None)
                return
            await queue.put((msg.get("text"), msg.get("bytes")))
    except (WebSocketDisconnect, RuntimeError, ConnectionError):
        cancel.set()
        await queue.put(None)


async def _admit(ws: WebSocket, want: str) -> str | None:
    """Accept, then authorise and count. Returns the peer label, or None when
    the connection was refused (an error frame has been sent)."""
    await ws.accept()
    if not _ws_authorized(ws, want):
        await _send_json_frame(ws, P.error(P.ERR_UNAUTHORIZED))
        await _close(ws)
        return None
    if _state["conns"] >= S.MAX_CONNS:
        await _send_json_frame(ws, P.error(P.ERR_BUSY))
        await _close(ws)
        return None
    _state["conns"] += 1
    peer = ws.headers.get("x-forwarded-for", "") or (ws.client.host if ws.client else "?")
    log.info("open %s %s (%d/%d)", want, peer, _state["conns"], S.MAX_CONNS)
    return peer


# ------------------------------------------------------------------ TTS socket

async def _stream_pcm(ws: WebSocket, pcm: bytes, cancel: threading.Event) -> bool:
    pace = S.CHUNK_BYTES / (S.SR * S.BYTES_PER_SAMPLE) * S.PACE_FACTOR
    for fr in frames_of(pcm, S.CHUNK_BYTES):
        if cancel.is_set():
            return False
        if not await _send_bytes(ws, fr):
            return False
        # yield after every frame so the reader sees a CLOSE promptly
        await asyncio.sleep(pace or 0)
    return True


async def _serve_tts(ws: WebSocket, req: P.Request, cancel: threading.Event) -> bool:
    """Handle one TTS request. False means: stop, the connection is finished."""
    lang = resolve_lang(req.lang, _tts_langs(), S.DEFAULT_LANG)
    if lang is None:
        await _send_json_frame(ws, P.error(f"{P.ERR_UNKNOWN_LANG}; served: {', '.join(_tts_langs())}"))
        return False
    if get_tts() is None or not _state["ready"]:
        await _send_json_frame(ws, P.error(P.ERR_SYNTH_FAILED if _state["error"] else P.ERR_NOT_READY))
        return False
    voice = resolve_voice(lang, req.voice)
    segs = segment(req.text)[:S.MAX_SEGMENTS]
    if not segs:
        await _send_json_frame(ws, P.error(P.ERR_EMPTY_TEXT))
        return False
    try:
        await asyncio.wait_for(SEM.acquire(), S.QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        await _send_json_frame(ws, P.error(P.ERR_BUSY))
        return False

    started, sent_bytes, first_audio_at, carry = time.perf_counter(), 0, None, b""
    try:
        if req.meta:
            await _send_json_frame(ws, P.frame(
                {"type": "start", "sample_rate": S.SR, "channels": 1, "format": "pcm_s16le",
                 "lang": lang, "voice": voice["id"], "segments": len(segs)}))
        if S.LOG_TEXT:
            log.debug("synth lang=%s voice=%s text=%r", lang, voice["id"], req.text[:120])
        async for chunk in _aiter_in_thread(_synth_segments(req.text, lang, voice, cancel)):
            if cancel.is_set():
                return False
            if not chunk:
                continue
            if first_audio_at is None:
                first_audio_at = time.perf_counter() - started
            # keep wire frames CHUNK_BYTES long across chunk boundaries
            data = carry + chunk
            whole = len(data) - len(data) % S.CHUNK_BYTES
            carry = data[whole:]
            if whole and not await _stream_pcm(ws, data[:whole], cancel):
                return False
            sent_bytes += whole
            if sent_bytes >= S.MAX_TOTAL_BYTES:
                log.warning("hit MAX_TOTAL_BYTES")
                carry = b""
                break
        if carry and not cancel.is_set():
            if not await _stream_pcm(ws, carry, cancel):
                return False
            sent_bytes += len(carry)
        if not sent_bytes:
            await _send_json_frame(ws, P.error(P.ERR_NO_AUDIO))
            return False
        if not await _send_json_frame(ws, P.END):
            log.info("client gone before end (%d bytes sent)", sent_bytes)
            return False
        audio_s = sent_bytes / (S.SR * S.BYTES_PER_SAMPLE)
        elapsed = time.perf_counter() - started
        log.info("ok lang=%s segs=%d %.2fs audio in %.2fs (rtf %.2f) ttfa %.2fs", lang, len(segs),
                 audio_s, elapsed, elapsed / max(audio_s, 1e-9), first_audio_at or 0.0)
        return True
    except asyncio.CancelledError:
        cancel.set()
        raise
    except Exception as exc:                  # noqa: BLE001 - type name only, the endpoint is public
        log.exception("synthesis failed")
        await _send_json_frame(ws, P.error(f"{P.ERR_SYNTH_FAILED}: {type(exc).__name__}"))
        return False
    finally:
        SEM.release()


@app.websocket("/tts")
async def tts_ws(ws: WebSocket):
    peer = await _admit(ws, "tts")
    if peer is None:
        return
    cancel = threading.Event()
    queue: asyncio.Queue = asyncio.Queue(maxsize=4)
    reader = asyncio.create_task(_reader(ws, queue, cancel))
    deadline = time.monotonic() + S.CONN_MAX_SECONDS
    served = 0
    try:
        while True:
            budget = S.FIRST_FRAME_TIMEOUT_S if served == 0 else S.IDLE_TIMEOUT_S
            budget = min(budget, max(0.0, deadline - time.monotonic()))
            if budget <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), budget)
            except asyncio.TimeoutError:
                break
            if item is None:
                break
            text, data = item
            if text is None and data is not None:
                text = data.decode("utf-8", "replace")      # text with the binary opcode
            try:
                req = P.parse(text)
            except P.BadRequest as exc:
                await _send_json_frame(ws, P.error(str(exc)))
                break
            served += 1
            if not await _serve_tts(ws, req, cancel):
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
        log.info("close tts %s (served %d)", peer, served)


# ------------------------------------------------------------------ ASR socket

@app.websocket("/asr")
async def asr_ws(ws: WebSocket):
    peer = await _admit(ws, "asr")
    if peer is None:
        return
    asr = get_asr()
    cancel = threading.Event()
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    reader = asyncio.create_task(_reader(ws, queue, cancel))
    deadline = time.monotonic() + S.ASR_CONN_MAX_SECONDS
    loop = asyncio.get_running_loop()
    lang = resolve_lang(None, _asr_langs(), S.DEFAULT_LANG)
    sample_rate, partials = S.SR, True
    ep = Endpointer()
    total_samples, finals, started = 0, 0, time.perf_counter()
    # transcriptions run one at a time per connection, in order; a partial that
    # is superseded before it ran is dropped, a final never is
    jobs: asyncio.Queue = asyncio.Queue()
    stop_requested = False

    async def worker():
        nonlocal finals
        while True:
            seg = await jobs.get()
            if seg is None:
                return
            if not seg.final and not jobs.empty():
                continue
            if cancel.is_set():
                continue
            try:
                t0 = time.perf_counter()
                text = await loop.run_in_executor(POOL, asr.transcribe, seg.audio, lang)
                kind = "final" if seg.final else "partial"
                log.info("asr %s lang=%s %.1fs audio in %.2fs: %r", kind, lang,
                         seg.audio.size / S.SR, time.perf_counter() - t0, text[:80])
                if cancel.is_set():
                    continue
                if seg.final:
                    finals += 1
                    await _send_json_frame(ws, P.frame({"type": "final", "text": text,
                                                         "start": seg.start_s, "end": seg.end_s}))
                elif partials:
                    await _send_json_frame(ws, P.frame({"type": "partial", "text": text,
                                                         "start": seg.start_s}))
            except Exception as exc:          # noqa: BLE001
                log.exception("transcription failed")
                await _send_json_frame(ws, P.error(f"{P.ERR_ASR_FAILED}: {type(exc).__name__}"))
                cancel.set()
                return

    runner = asyncio.create_task(worker())
    error_sent = False
    try:
        if asr is None or not _state["ready"]:
            await _send_json_frame(ws, P.error(P.ERR_NOT_READY if asr else "speech recognition is not enabled"))
            error_sent = True
            return
        while not cancel.is_set():
            budget = max(0.0, deadline - time.monotonic())
            if budget <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), min(budget, S.IDLE_TIMEOUT_S))
            except asyncio.TimeoutError:
                break
            if item is None:
                break
            text, data = item
            if text is not None:
                try:
                    ctl = P.parse_asr_control(text)
                except P.BadRequest as exc:
                    await _send_json_frame(ws, P.error(str(exc)))
                    error_sent = True
                    return
                if ctl.kind == "stop":
                    stop_requested = True
                    break
                code = resolve_lang(ctl.lang, _asr_langs(), S.DEFAULT_LANG)
                if code is None:
                    await _send_json_frame(ws, P.error(f"{P.ERR_UNKNOWN_LANG}; served: {', '.join(_asr_langs())}"))
                    error_sent = True
                    return
                lang, sample_rate, partials = code, ctl.sample_rate, ctl.partials
                continue
            if data is None:
                continue
            if len(data) > S.MAX_AUDIO_FRAME_BYTES:
                await _send_json_frame(ws, P.error(P.ERR_AUDIO_FRAME_TOO_BIG))
                error_sent = True
                return
            if len(data) % 2:
                await _send_json_frame(ws, P.error(P.ERR_ODD_FRAME))
                error_sent = True
                return
            audio = pcm16_to_f32(data)
            if sample_rate != S.SR:
                audio = resample_to_16k(audio, sample_rate)
            total_samples += audio.size
            if total_samples > S.ASR_MAX_AUDIO_S * S.SR:
                await _send_json_frame(ws, P.error(P.ERR_TOO_MUCH_AUDIO))
                error_sent = True
                return
            for seg in ep.feed(audio):
                await jobs.put(seg)
        if not cancel.is_set():
            for seg in ep.finish():
                await jobs.put(seg)
            await jobs.put(None)
            await runner
            if not error_sent and not cancel.is_set():
                await _send_json_frame(ws, P.END)
                log.info("asr ok lang=%s %.1fs audio, %d finals, in %.1fs%s", lang,
                         total_samples / S.SR, finals, time.perf_counter() - started,
                         "" if stop_requested else " (idle close)")
    except asyncio.CancelledError:
        cancel.set()
        raise
    finally:
        cancel.set()
        runner.cancel()
        reader.cancel()
        with contextlib.suppress(BaseException):
            await runner
        with contextlib.suppress(BaseException):
            await reader
        await _close(ws)
        _state["conns"] -= 1
        log.info("close asr %s", peer)
