"""Serve the browser check locally with cross-origin isolation (so
onnxruntime-web can use wasm threads). Only the porting folders are exposed —
never the repo root, which holds .env.

  python3 porting/web/serve.py            # http://localhost:8765/web/
"""

import http.server
import os
import posixpath
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PORT = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs")) / "port"
MOUNTS = {"web": ROOT / "porting" / "web", "artefacts": PORT / "artefacts",
          "results": PORT / "results", "evalpack": PORT / "evalpack" / "wav"}


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        parts = [p for p in posixpath.normpath(urllib.parse.unquote(urllib.parse.urlsplit(path).path)).split("/") if p]
        if not parts or parts[0] not in MOUNTS or ".." in parts:
            return str(MOUNTS["web"] / "__nothing__")
        return str(MOUNTS[parts[0]].joinpath(*parts[1:]))

    def do_POST(self):
        """POST /save {model, device, dtype, lang, hyps, rtf, load_ms, adapter}:
        score a browser run like porting.validate does and write
        results/<model>/<device>-<dtype>-<lang>.json, so the report picks it up."""
        import json
        import re
        sys.path.insert(0, str(ROOT))
        from porting.metrics import agreement, score
        if self.path != "/save":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model, lang = body["model"], body.get("lang", "ceb")
        dev, dtype = body["device"], body["dtype"]
        if not all(re.fullmatch(r"[\w.\-]+", s) for s in (model, lang, dev, dtype)) \
                or not (MOUNTS["artefacts"] / model).is_dir():
            self.send_error(400)
            return
        refs = json.loads((MOUNTS["evalpack"] / lang / "refs.json").read_text())[:len(body["hyps"])]
        sfx = {"fp16": "_fp16", "q8": "_quantized", "fp32": ""}.get(dtype, "")
        files = [p for p in (MOUNTS["artefacts"] / model / "onnx-web" / "onnx").iterdir()
                 if p.name.split(".onnx")[0].endswith(sfx) and (sfx or not p.name.split(".onnx")[0]
                                                                 .endswith(("_fp16", "_quantized")))
                 and not p.name.startswith(("decoder_model.", "decoder_with_past"))]
        res = {"repo": f"sapinsapin/{model}", "runtime": dev, "variant": dtype, "lang": lang,
               "clips": len(body["hyps"]), "threads": None, "rtf": round(body["rtf"], 3),
               "load_seconds": round(body.get("load_ms", 0) / 1000, 1), "adapter": body.get("adapter"),
               "size_mb": round(sum(p.stat().st_size for p in files) / 2**20, 1) or None,
               "accuracy": score(refs, body["hyps"]), "hyps": body["hyps"]}
        ref = MOUNTS["results"] / model / f"torch-fp32-{lang}.json"
        if ref.exists():
            res["parity"] = agreement(json.loads(ref.read_text())["hyps"][:len(body["hyps"])], body["hyps"])
        out = MOUNTS["results"] / model / f"{dev}-{dtype}-{lang}.json"
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"saved": out.name, "cer": res["accuracy"]["cer"]}).encode())

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def guess_type(self, path):
        if str(path).endswith((".mjs", ".js")):
            return "text/javascript"
        if str(path).endswith(".wasm"):
            return "application/wasm"
        return super().guess_type(path)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
