#!/usr/bin/env python3
"""
Host viability probe — run this against a host BEFORE trusting it.

    python tools/ws_probe.py wss://sapinsapin-tts-ws.hf.space/tts

A reverse proxy can break this design in ways a working HTTP endpoint does not
reveal. Four separate things are checked, because only the first is the one
people think to test:

  1. handshake   the WebSocket upgrade survives the proxy at all
  2. streaming   binary frames arrive incrementally, not buffered to the end
  3. idle        the proxy does not close a quiet connection mid-synthesis
  4. size        frames of CHUNK_BYTES are not fragmented or rejected

A proxy that buffers the whole response still passes a naive test while
silently defeating streaming, which is the entire point of chunked synthesis.
"""

import argparse
import asyncio
import json
import sys
import time

import websockets


async def probe_handshake(url: str, timeout: float) -> tuple[bool, str]:
    try:
        async with websockets.connect(url, open_timeout=timeout,
                                      max_size=None) as ws:
            ext = [str(e) for e in (ws.protocol.extensions or [])]
            return True, f"101 upgrade ok; extensions={ext or 'none'}"
    except websockets.InvalidStatus as exc:
        return False, (f"handshake rejected: HTTP {exc.response.status_code} "
                       "(a 404 here means the proxy never routed the upgrade)")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def probe_stream(url: str, text: str, chunk: int,
                       timeout: float) -> tuple[bool, str]:
    """Send multi-clause text so the server must produce several segments."""
    try:
        async with websockets.connect(url, open_timeout=timeout,
                                      max_size=None) as ws:
            await ws.send(json.dumps({"text": text}))
            t0 = time.perf_counter()
            arrivals, total, oversize = [], 0, 0
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout)
                if isinstance(msg, bytes):
                    arrivals.append(time.perf_counter() - t0)
                    total += len(msg)
                    oversize += len(msg) > chunk
                    continue
                obj = json.loads(msg)
                if obj.get("type") == "error":
                    return False, f"server error: {obj['message']}"
                break
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if not arrivals:
        return False, "no audio frames arrived"
    secs = total / 32000
    spread = arrivals[-1] - arrivals[0]
    detail = (f"{len(arrivals)} frames, {total} bytes ({secs:.1f}s audio); "
              f"first at {arrivals[0]:.2f}s, spread {spread:.2f}s")
    if oversize:
        return False, f"{detail}; {oversize} frames exceeded {chunk} bytes"
    if len(arrivals) > 8 and spread < 0.2:
        return False, (f"{detail}; frames arrived in one burst — something in "
                       "the path buffered the whole stream")
    return True, detail


async def probe_idle(url: str, seconds: float, server_close: float,
                     timeout: float) -> tuple[bool, str]:
    """Hold a connection open without traffic and see who closes it.

    Our own server closes a socket that has not sent a request within
    FIRST_FRAME_TIMEOUT_S (30 s by default), with code 1000. That is deliberate
    anti-slowloris behaviour, not a proxy problem, so it has to be attributed
    correctly or this check cries wolf on a perfectly good host. Raise the
    server's timeout if you need to probe past it:

        FIRST_FRAME_TIMEOUT_S=120 uvicorn app:app ...
    """
    try:
        async with websockets.connect(url, open_timeout=timeout,
                                      max_size=None,
                                      ping_interval=None) as ws:
            t0 = time.perf_counter()
            try:
                await asyncio.wait_for(ws.recv(), seconds)
            except asyncio.TimeoutError:
                return True, f"survived {seconds:.0f}s idle"
            except websockets.ConnectionClosed as exc:
                held = time.perf_counter() - t0
                code = exc.rcvd.code if exc.rcvd else None
                if code == 1000 and held >= server_close - 3:
                    return True, (f"closed after {held:.0f}s by our own server "
                                  f"(FIRST_FRAME_TIMEOUT_S={server_close:.0f}), "
                                  "not by the proxy")
                return False, (f"closed after {held:.0f}s with code {code} — "
                               "earlier than our own timeout, so something in "
                               "the path dropped it")
            return True, "received an unexpected frame but stayed open"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--chunk", type=int, default=3200)
    p.add_argument("--idle", type=float, default=60.0)
    p.add_argument("--server-idle-close", type=float, default=30.0,
                   help="the server's own FIRST_FRAME_TIMEOUT_S")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--skip-idle", action="store_true")
    p.add_argument("--text", default=(
        "Napansin niyang hugis puso ang mga dahon nito, at sinabi niya na ito "
        "ay ginagamit na panggamot ng maraming pamilya sa probinsya."))
    args = p.parse_args()

    print(f"probing {args.url}\n")
    checks = [("handshake", probe_handshake(args.url, args.timeout)),
              ("streaming", probe_stream(args.url, args.text, args.chunk,
                                         args.timeout))]
    if not args.skip_idle:
        checks.append(("idle", probe_idle(args.url, args.idle,
                                          args.server_idle_close,
                                          args.timeout)))

    failed = 0
    for name, coro in checks:
        good, detail = await coro
        print(f"{'ok  ' if good else 'FAIL'}  {name:10} {detail}")
        failed += not good

    print()
    if failed:
        print(f"{failed} check(s) failed — this host is not viable as is. "
              "Fall back to a Cloudflare named tunnel or a plain-ws VM.")
    else:
        print("all checks passed — this host can carry the endpoint.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
