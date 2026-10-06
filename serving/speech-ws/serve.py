#!/usr/bin/env python3
"""
Entrypoint: the flags nyo-node's catalog recipe passes, turned into settings
and a uvicorn run with the WebSocket options the Agora contract needs.

    python serve.py --served-name sapin-speech --port 8000 --api-key "$NYO_SERVING_KEY"
    python serve.py --tts-langs fil,ceb,hil --asr-langs fil,ceb,hil --tts-engine orpheus

Every flag has an env-var twin (settings.py); flags win.
"""

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--served-name", default=None, help="the model name NYO routes to this server")
    ap.add_argument("--api-key", default=None, help="serving key for /v1/* and socket tickets")
    ap.add_argument("--tts-engine", default=None, choices=["orpheus", "speecht5", "none"])
    ap.add_argument("--tts-langs", default=None, help="comma-separated codes, e.g. fil,ceb,hil")
    ap.add_argument("--asr-langs", default=None)
    ap.add_argument("--orpheus-backend", default=None, choices=["transformers", "llamacpp"])
    ap.add_argument("--ws-public", action="store_true", help="sockets need no ticket even with a key")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    env = {
        "SPEECH_SERVED_NAME": args.served_name, "SPEECH_API_KEY": args.api_key,
        "TTS_ENGINE": args.tts_engine, "TTS_LANGS": args.tts_langs, "ASR_LANGS": args.asr_langs,
        "ORPHEUS_BACKEND": args.orpheus_backend,
        "SPEECH_WS_PUBLIC": "1" if args.ws_public else None,
    }
    for k, v in env.items():
        if v is not None:
            os.environ[k] = str(v)

    import uvicorn
    # --workers 1 is required: a second worker would hold its own copy of the
    # weights and break the in-flight semaphore. The ping timeout is 60 s so a
    # minimal firmware that does not auto-pong is not cut off mid-synthesis.
    uvicorn.run("app:app", host=args.host, port=args.port, workers=1, ws="websockets",
                ws_ping_interval=20, ws_ping_timeout=60, timeout_keep_alive=75,
                log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
