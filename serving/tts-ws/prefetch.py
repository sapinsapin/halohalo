"""
Download every weight the server needs, at image build time.

Run from the Dockerfile, not at runtime. A lazy download inside a live
WebSocket connection means ~600 MB fetched while the device waits, which no
firmware timeout will tolerate.

Revisions come from pins.py, which the server loads from too, so the image and
the running process cannot disagree about what is being served.
"""

import os
import sys

from huggingface_hub import snapshot_download

from pins import PINS, tts_repo

# The .bin duplicates of .safetensors would double the image for nothing —
# except speecht5_hifigan, which only ships pytorch_model.bin.
IGNORE = ["*.msgpack", "*.h5", "*.onnx", "*.ot"]


def main() -> int:
    langs = [x.strip() for x in
             os.environ.get("SERVE_LANGS", "fil").split(",") if x.strip()]
    wanted = [r for r in PINS
              if not r.startswith("sapinsapin/speecht5_tts-pld-")
              or r in {tts_repo(l) for l in langs}]
    for repo in wanted:
        ignore = IGNORE + ([] if "hifigan" in repo else ["*.bin"])
        path = snapshot_download(repo, revision=PINS[repo],
                                 ignore_patterns=ignore)
        print(f"prefetched {repo}@{PINS[repo][:8]} -> {path}", flush=True)
    missing = [l for l in langs if tts_repo(l) not in PINS]
    if missing:
        print(f"ERROR: SERVE_LANGS names {missing}, which have no pin in "
              f"prefetch.py — add them there first", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
