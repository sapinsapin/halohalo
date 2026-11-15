"""ASR judge server for a second GPU (the RTX 3070 on reth5et), so judging overlaps GB10 work.

    HF_HUB_OFFLINE=1 python -m training.fil_moss.judge_server --host 100.92.72.53 --port 9310 \
        --judges whisper --token-file ~/.config/fil-moss/judge-token \
        --omni-port-dir ~/fil-moss-judges/omniASR-CTC-1B-hf --omni-port-code ~/fil-moss-judges/code

Serves the pinned judges from ``recipe.json`` (``judges.fil``), so the remote and a local judge are the same
model and decoding. Binds to one address (use the tailnet address, never 0.0.0.0), requires a bearer token,
and handles one request at a time. Before loading it checks free VRAM: the 3070 also hosts other services, and
the server refuses to start rather than squeeze them.

Only synthesised audio is sent here (rollouts and evaluation generations). Real recordings and their speaker
embeddings stay on the GB10.

POST /transcribe  {"judges": [...], "audio_b64": [little-endian int16 PCM at 16 kHz, ...]}
               -> {"hyps": [{judge: text}, ...], "seconds": float}
GET  /healthz     -> {"judges": {...}, "free_gib": float}
"""
import argparse
import base64
import hmac
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAX_BODY = 64 * 2 ** 20
NEED_GIB = {"whisper": 3.6, "omni": 2.4, "mms": 4.4}


def free_gib():
    """Free GPU memory, the lower of CUDA's and nvidia-smi's view. Under WSL2, ``torch.cuda.mem_get_info``
    ignores other processes (measured on reth5et 2026-11-15: 6.93 GiB "free" while services held 6.2 GiB), so
    nvidia-smi is the one that sees them."""
    import shutil
    import subprocess
    import torch
    values = [torch.cuda.mem_get_info()[0] / 2 ** 30]
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else None)
    if smi:
        out = subprocess.run([smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.split("\n")[0]
        used, total = (float(x) for x in out.split(","))
        values.append((total - used) / 1024)
    return min(values)


def decode_audio(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<i2").astype(np.float32) / 32768.0


def encode_audio(wav16):
    pcm = np.clip(np.asarray(wav16, dtype=np.float32), -1.0, 1.0)
    return base64.b64encode((pcm * 32767.0).astype("<i2").tobytes()).decode("ascii")


def make_handler(judges, meta, token):
    from . import judges as J

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self):
            got = self.headers.get("Authorization", "")
            return hmac.compare_digest(got.encode(), ("Bearer " + token).encode())

        def do_GET(self):
            if not self._authorised():
                return self._send(401, {"error": "unauthorised"})
            if self.path != "/healthz":
                return self._send(404, {"error": "not found"})
            self._send(200, {"judges": meta, "free_gib": free_gib()})

        def do_POST(self):
            if not self._authorised():
                return self._send(401, {"error": "unauthorised"})
            if self.path != "/transcribe":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n <= MAX_BODY:
                return self._send(413, {"error": "body size"})
            try:
                req = json.loads(self.rfile.read(n))
                names = req["judges"]
                missing = [x for x in names if x not in judges]
                if missing:
                    return self._send(400, {"error": "judges not served: %s" % missing})
                start = time.monotonic()
                wavs = [decode_audio(a) for a in req["audio_b64"]]
                hyps = J.transcribe_batch({x: judges[x] for x in names}, wavs)
                self._send(200, {"hyps": hyps, "seconds": time.monotonic() - start})
            except Exception as exc:
                self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc)})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=9310)
    p.add_argument("--judges", nargs="+", default=["whisper"])
    p.add_argument("--token-file", required=True)
    p.add_argument("--recipe", default=str(HERE / "recipe.json"))
    p.add_argument("--omni-port-dir")
    p.add_argument("--omni-port-code")
    p.add_argument("--headroom-gib", type=float, default=1.0)
    a = p.parse_args(argv)
    if a.host in ("0.0.0.0", "::"):
        raise SystemExit("bind to one address (the tailnet address), not every interface")
    token = Path(a.token_file).expanduser().read_text().strip()
    if len(token) < 32:
        raise SystemExit("token too short")
    import torch
    from . import judges as J
    need = sum(NEED_GIB[j] for j in a.judges) + a.headroom_gib
    free = free_gib()
    if free < need:
        raise SystemExit("only %.2f GiB free, need %.2f GiB for %s; refusing to start" % (free, need, a.judges))
    cfg = json.loads(Path(a.recipe).read_text())["judges"]["fil"]
    cfg = {k: dict(v) for k, v in cfg.items() if k in a.judges}
    if "omni" in cfg:
        cfg["omni"].update({k: v for k, v in (("port_dir", a.omni_port_dir), ("port_code", a.omni_port_code)) if v})
    judges, failed = J.load(cfg)
    if failed:
        raise SystemExit("judges failed to load: %s" % failed)
    meta = {k: {kk: vv for kk, vv in v.items() if kk in ("model", "revision", "family")} for k, v in cfg.items()}
    print(json.dumps({"serving": meta, "host": a.host, "port": a.port,
                      "free_gib_after_load": free_gib()}), flush=True)
    HTTPServer((a.host, a.port), make_handler(judges, meta, token)).serve_forever()


if __name__ == "__main__":
    main()
