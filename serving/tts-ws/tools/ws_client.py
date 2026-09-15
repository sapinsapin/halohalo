#!/usr/bin/env python3
"""
Reference client and verification harness for the demo TTS endpoint.

    python tools/ws_client.py ws://localhost:7860/tts "Magandang umaga po."
    python tools/ws_client.py wss://sapinsapin-tts-ws.hf.space/tts --lang ceb

Collects the stream, runs the checks below, and writes out.wav so the result
can be listened to. The signal checks exist because the failure modes here are
silent: byte-swapped or misaligned PCM decodes into loud static rather than an
error, and a runaway stop token produces plausible-looking but far-too-long
audio. Zero-crossing rate catches the first, duration-per-character the second.
"""

import argparse
import asyncio
import json
import sys
import time

import numpy as np
import websockets

SR = 16000


async def run(args) -> int:
    payload = {"text": args.text}
    if args.lang:
        payload["lang"] = args.lang
    if args.voice:
        payload["voice"] = args.voice
    if args.meta:
        payload["meta"] = True

    audio, terminals, arrivals = [], [], []
    t0 = time.perf_counter()
    first = None

    async with websockets.connect(args.url, max_size=None,
                                  open_timeout=args.timeout) as ws:
        connected = time.perf_counter() - t0
        await ws.send(json.dumps(payload))
        t_sent = time.perf_counter()
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), args.timeout)
            except asyncio.TimeoutError:
                print(f"FAIL: no frame for {args.timeout}s", file=sys.stderr)
                return 2
            except websockets.ConnectionClosed:
                break
            if isinstance(msg, bytes):
                if first is None:
                    first = time.perf_counter() - t_sent
                audio.append(msg)
                arrivals.append(time.perf_counter() - t_sent)
            else:
                terminals.append(msg)
                break

    print(f"connect            {connected * 1000:.0f} ms")
    print(f"frames             {len(audio)}")
    print(f"terminal           {terminals}")

    if terminals and json.loads(terminals[0]).get("type") == "error":
        print(f"\nserver returned an error: {terminals[0]}")
        return 1

    # --- protocol shape
    assert audio, "no audio received"
    assert len(terminals) == 1, f"expected exactly one terminal frame, got {terminals}"
    assert json.loads(terminals[0]) == {"type": "end"}, terminals[0]
    assert terminals[0].encode() == b'{"type":"end"}', \
        f"end frame is not byte-identical to the contract: {terminals[0]!r}"
    # An odd-length frame shifts every later int16 sample by one byte.
    odd = [i for i, f in enumerate(audio) if len(f) % 2]
    assert not odd, f"odd-length frames at {odd[:5]}"
    big = [len(f) for f in audio if len(f) > args.chunk]
    assert not big, f"frames larger than {args.chunk} bytes: {big[:5]}"

    # --- signal sanity
    pcm = np.frombuffer(b"".join(audio), dtype="<i2")
    secs = pcm.size / SR
    peak = int(np.abs(pcm).max())
    clipped = float((np.abs(pcm) >= 32700).mean())
    dc = float(pcm.astype(np.float64).mean())
    zcr = float(np.mean(np.diff(np.signbit(pcm)) != 0))
    rate = len(args.text) / secs

    print(f"audio              {secs:.2f}s  ({pcm.size} samples, "
          f"{len(b''.join(audio))} bytes)")
    print(f"time to first      {first * 1000:.0f} ms")
    print(f"wall clock         {arrivals[-1]:.2f}s  "
          f"(realtime factor {arrivals[-1] / secs:.2f})")
    print(f"peak / clipped     {peak} / {clipped:.5f}")
    print(f"dc offset          {dc:.1f}")
    print(f"zero crossing rate {zcr:.3f}")
    print(f"speech rate        {rate:.1f} chars/s")

    assert peak > 2000, f"near silence (peak {peak})"
    assert clipped < 0.001, f"hard clipped ({clipped:.4f} of samples)"
    assert abs(dc) < 200, f"DC offset {dc:.1f}"
    # Byte-swapped S16 lands near 0.5; real speech at 16 kHz sits well under.
    assert zcr < 0.25, f"noise-like (zcr {zcr:.3f}) — check endianness/alignment"
    # Filipino read speech measured at 16-19 chars/s; a wide band still catches
    # a stop-token runaway or a truncated utterance.
    assert 6.0 < rate < 60.0, f"implausible speech rate ({rate:.1f} chars/s)"

    # A single segment is synthesized whole and then sent, so its frames
    # legitimately arrive in one burst. Only multi-segment text can distinguish
    # real streaming from a proxy that buffered the whole response, which is
    # what --expect-streaming asserts.
    spread = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0.0
    print(f"arrival spread     {spread:.2f}s over {len(arrivals)} frames")
    if args.expect_streaming:
        assert spread > 0.2, (
            f"frames arrived in one burst (spread {spread:.2f}s) — either the "
            "text was one segment or something in the path buffered the stream")
        print("streaming          confirmed incremental")

    if args.out:
        import soundfile as sf
        sf.write(args.out, pcm, SR, subtype="PCM_16")
        print(f"\nwrote {args.out} — listen to it, "
              f"the checks above cannot hear intelligibility")
    print("\nALL CHECKS PASSED")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("url", nargs="?", default="ws://localhost:7860/tts")
    p.add_argument("text", nargs="?", default="Magandang umaga po sa inyong lahat.")
    p.add_argument("--lang")
    p.add_argument("--voice")
    p.add_argument("--meta", action="store_true")
    p.add_argument("--expect-streaming", action="store_true",
                   help="require frames to arrive incrementally; "
                        "only meaningful for multi-segment text")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--chunk", type=int, default=3200)
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()
    try:
        return asyncio.run(run(args))
    except AssertionError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
