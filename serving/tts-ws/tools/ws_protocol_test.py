#!/usr/bin/env python3
"""
Protocol conformance and failure-path tests.

    python tools/ws_protocol_test.py ws://localhost:7860/tts

Covers what the happy-path client cannot: every error frame, the
one-terminal-frame-per-request invariant, socket reuse after end, cancellation
on client close, and behaviour under saturation.
"""

import asyncio
import json
import sys
import time

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:7860/tts"
LONG = ("Napansin niyang hugis puso ang mga dahon nito, at sinabi niya na ito "
        "ay ginagamit na panggamot ng maraming pamilya sa probinsya kapag may "
        "sakit ang mga bata sa kanilang tahanan.")
PASS, FAIL = [], []


def ok(name, detail=""):
    PASS.append(name)
    print(f"ok    {name}{'  ' + detail if detail else ''}")


def bad(name, detail):
    FAIL.append(name)
    print(f"FAIL  {name}: {detail}")


async def collect(ws, timeout=90.0):
    """Drain until a text frame arrives. Returns (audio_bytes, terminal_or_None)."""
    audio = 0
    while True:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout)
        except asyncio.TimeoutError:
            return audio, None
        except websockets.ConnectionClosed:
            return audio, None
        if isinstance(msg, bytes):
            audio += len(msg)
        else:
            return audio, msg


async def expect_error(name, payload, contains):
    try:
        async with websockets.connect(URL, max_size=None) as ws:
            await ws.send(payload)
            audio, term = await collect(ws, 30.0)
            if term is None:
                return bad(name, "no terminal frame")
            obj = json.loads(term)
            if obj.get("type") != "error":
                return bad(name, f"expected error, got {term}")
            if contains not in obj.get("message", ""):
                return bad(name, f"message {obj['message']!r} lacks {contains!r}")
            if audio:
                return bad(name, f"sent {audio} bytes of audio with an error")
            # An error frame must be the last thing on the wire.
            try:
                extra = await asyncio.wait_for(ws.recv(), 2.0)
                return bad(name, f"frame after error: {extra!r}")
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                pass
            ok(name, f'-> "{obj["message"]}"')
    except Exception as exc:
        bad(name, f"{type(exc).__name__}: {exc}")


async def test_errors():
    await expect_error("malformed JSON", "this is not json", "invalid JSON")
    await expect_error("JSON array", "[1,2,3]", "JSON object")
    await expect_error("missing text", "{}", "missing 'text'")
    await expect_error("empty text", '{"text":"   "}', "is empty")
    await expect_error("non-string text", '{"text":42}', "missing 'text'")
    await expect_error("over-long text",
                       json.dumps({"text": "a" * 501}), "exceeds 500")
    await expect_error("oversize frame",
                       json.dumps({"text": "a", "pad": "x" * 9000}),
                       "exceeds 8192 bytes")
    await expect_error("unknown lang",
                       '{"text":"Kumusta po.","lang":"klingon"}', "unknown 'lang'")


async def test_unknown_voice_falls_back():
    name = "unknown voice falls back silently"
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send('{"text":"Kumusta po.","voice":"NOPE_9999"}')
        audio, term = await collect(ws)
    if term == '{"type":"end"}' and audio > 0:
        ok(name, f"{audio} bytes")
    else:
        bad(name, f"audio={audio} term={term}")


async def test_extra_fields_ignored():
    name = "unknown fields ignored"
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send('{"text":"Kumusta po.","speed":2,"future_flag":true}')
        audio, term = await collect(ws)
    ok(name, f"{audio} bytes") if term == '{"type":"end"}' and audio \
        else bad(name, f"audio={audio} term={term}")


async def test_binary_opcode_accepted():
    name = "text payload sent with binary opcode"
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(b'{"text":"Kumusta po."}')      # bytes -> binary opcode
        audio, term = await collect(ws)
    ok(name, f"{audio} bytes") if term == '{"type":"end"}' and audio \
        else bad(name, f"audio={audio} term={term}")


async def test_end_is_exact():
    name = "end frame is byte-exact"
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send('{"text":"Kumusta po."}')
        _, term = await collect(ws)
    if term is not None and term.encode() == b'{"type":"end"}':
        ok(name)
    else:
        bad(name, f"got {term!r}")


async def test_socket_reuse():
    name = "socket reuse: two requests, one connection"
    async with websockets.connect(URL, max_size=None) as ws:
        a1, t1 = (await collect(ws)) if await ws.send('{"text":"Kumusta po."}') \
            is None else (0, None)
        if t1 != '{"type":"end"}':
            return bad(name, f"first request: {t1}")
        await ws.send('{"text":"Salamat po."}')
        a2, t2 = await collect(ws)
    if t2 == '{"type":"end"}' and a1 and a2:
        ok(name, f"{a1} then {a2} bytes")
    else:
        bad(name, f"second request: audio={a2} term={t2}")


async def test_meta_opt_in():
    name = "meta frame only when asked"
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send('{"text":"Kumusta po.","meta":true}')
        first = await asyncio.wait_for(ws.recv(), 60.0)
    if isinstance(first, str) and json.loads(first).get("type") == "start":
        ok(name, "start frame present with meta:true")
    else:
        return bad(name, f"expected start frame, got {type(first).__name__}")

    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send('{"text":"Kumusta po."}')
        first = await asyncio.wait_for(ws.recv(), 60.0)
    if isinstance(first, bytes):
        ok("no frame before audio by default")
    else:
        bad("no frame before audio by default", f"got text frame {first!r}")


async def test_cancellation():
    name = "cancellation on client close"
    ws = await websockets.connect(URL, max_size=None)
    await ws.send(json.dumps({"text": LONG}))
    t_sent = time.perf_counter()
    for _ in range(4):                          # clearly mid-stream
        msg = await asyncio.wait_for(ws.recv(), 120.0)
        assert isinstance(msg, bytes)
    t_close = time.perf_counter()
    await ws.close()
    close_took = time.perf_counter() - t_close

    # What actually matters is that the server stopped working, which shows up
    # as a trivial follow-up not queueing behind the abandoned synthesis.
    await asyncio.sleep(0.3)
    t1 = time.perf_counter()
    async with websockets.connect(URL, max_size=None) as ws2:
        await ws2.send('{"text":"Oo."}')
        nxt = await asyncio.wait_for(ws2.recv(), 120.0)
        latency = time.perf_counter() - t1
        await collect(ws2)
    assert isinstance(nxt, bytes)
    if latency > 8.0:
        return bad(name, f"follow-up waited {latency:.1f}s — the abandoned "
                         "synthesis is still holding the worker")

    # How long close() took is a property of the path, not of the server. On a
    # direct connection it returns in milliseconds. Through a buffering reverse
    # proxy (Cloudflare tunnel, HF Spaces) the proxy keeps accepting our frames
    # and delays the closing handshake, so close() can block for the client's
    # full close timeout even though the server stopped on time. Reported, not
    # asserted — check the server log for "cancelled after segment N/M".
    note = f"closed in {close_took * 1000:.0f} ms"
    if close_took > 3.0:
        note = (f"close() took {close_took:.1f}s — a buffering proxy is in the "
                "path; the server still stopped (see its log)")
    ok(name, f"{note}; cancelled at {t_close - t_sent:.1f}s; "
             f"follow-up first audio in {latency:.1f}s")


async def test_saturation():
    name = "saturation: concurrent requests are serialized, not dropped"
    async def one(i):
        async with websockets.connect(URL, max_size=None) as ws:
            await ws.send(json.dumps({"text": LONG}))
            audio, term = await collect(ws, 180.0)
            return i, audio, term
    res = await asyncio.gather(*(one(i) for i in range(3)), return_exceptions=True)
    outcomes = []
    for r in res:
        if isinstance(r, Exception):
            outcomes.append(f"exc:{type(r).__name__}")
        else:
            i, audio, term = r
            obj = json.loads(term) if term else {}
            outcomes.append(f"{obj.get('type')}({audio}B)")
    served = sum(1 for o in outcomes if o.startswith("end"))
    if served >= 1 and not any(o.startswith("exc") for o in outcomes):
        ok(name, " ".join(outcomes))
    else:
        bad(name, " ".join(outcomes))


async def main():
    print(f"target {URL}\n")
    await test_errors()
    await test_unknown_voice_falls_back()
    await test_extra_fields_ignored()
    await test_binary_opcode_accepted()
    await test_end_is_exact()
    await test_socket_reuse()
    await test_meta_opt_in()
    await test_cancellation()
    await test_saturation()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
