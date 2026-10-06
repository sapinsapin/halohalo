#!/usr/bin/env python3
"""
Reference streaming-ASR client: plays a WAV file into /asr at realtime (or
faster) in 100 ms PCM frames and prints partials and finals as they arrive.

    python tools/asr_client.py ws://localhost:8000/asr clip.wav --lang ceb
    python tools/asr_client.py ws://localhost:8000/asr clip.wav --speed 0   # as fast as possible

Checks the protocol invariants: every text frame is JSON, finals carry
start/end, exactly one terminal frame, and it is `end` or `error`.
"""

import argparse
import asyncio
import json
import sys
import time

import numpy as np
import websockets

SR = 16000
FRAME = 3200       # bytes: 100 ms


def load_pcm16(path: str) -> bytes:
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        from fractions import Fraction
        r = Fraction(SR, int(sr)).limit_denominator(1000)
        wav = resample_poly(wav, r.numerator, r.denominator).astype(np.float32)
    return np.clip(wav, -1, 1).astype("<f4").__mul__(32767).astype("<i2").tobytes()


async def run(args) -> int:
    pcm = load_pcm16(args.wav)
    t0 = time.perf_counter()
    finals, partials, terminal = [], 0, None
    async with websockets.connect(args.url, max_size=None, open_timeout=args.timeout) as ws:
        start = {"type": "start", "lang": args.lang} if args.lang else {"type": "start"}
        await ws.send(json.dumps(start))

        async def send():
            for i in range(0, len(pcm), FRAME):
                await ws.send(pcm[i:i + FRAME])
                if args.speed > 0:
                    await asyncio.sleep((FRAME / 2 / SR) / args.speed)
            await ws.send(json.dumps({"type": "stop"}))

        sender = asyncio.create_task(send())
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), args.timeout)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            assert isinstance(msg, str), f"server sent a binary frame on /asr: {msg[:20]!r}"
            obj = json.loads(msg)
            t = time.perf_counter() - t0
            if obj["type"] == "partial":
                partials += 1
                print(f"{t:6.2f}s  partial  {obj['text']}")
            elif obj["type"] == "final":
                finals.append(obj)
                print(f"{t:6.2f}s  FINAL    [{obj['start']:.2f}-{obj['end']:.2f}] {obj['text']}")
            else:
                terminal = obj
                print(f"{t:6.2f}s  {msg}")
                break
        await sender
    assert terminal is not None, "no terminal frame"
    assert terminal["type"] in ("end", "error"), terminal
    print(f"\n{len(finals)} finals, {partials} partials, {len(pcm) / 2 / SR:.1f}s of audio "
          f"in {time.perf_counter() - t0:.1f}s")
    return 0 if terminal["type"] == "end" and finals else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("url", nargs="?", default="ws://localhost:8000/asr")
    p.add_argument("wav")
    p.add_argument("--lang")
    p.add_argument("--speed", type=float, default=1.0, help="1 = realtime, 0 = no pacing")
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()
    try:
        return asyncio.run(run(args))
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
