"""
Pinned model revisions, shared by the prefetcher and the server.

Both sides must agree: prefetch.py downloads these exact commits at build time,
and tts.py loads these exact commits at runtime. That is not only for
reproducibility — snapshot_download(revision=<sha>) does not create a refs/main
entry in the cache, so an offline container asked for the default "main"
revision finds nothing and fails. Loading by sha resolves the snapshot
directly, and makes the pin real rather than advisory.

To refresh one:

    curl -s https://huggingface.co/api/models/<repo> \\
      | python3 -c "import json,sys;print(json.load(sys.stdin)['sha'])"
"""

VOCODER = "microsoft/speecht5_hifigan"

PINS = {
    "sapinsapin/speecht5_tts-pld-fil": "b2eeb82edf3f9c1621e0f06eb284d321d716ff92",
    "sapinsapin/speecht5_tts-pld-ceb": "c5c0341c42aa456d6d9e2bd37b0eca9a6f85a2d6",
    "sapinsapin/speecht5_tts-pld-hil": "4614575016c8266ad49fe599fb876c60343ae3ed",
    VOCODER: "bb6f429406e86a9992357a972c0698b22043307d",
}


def tts_repo(lang: str) -> str:
    return f"sapinsapin/speecht5_tts-pld-{lang}"


def pinned_langs() -> tuple[str, ...]:
    prefix = "sapinsapin/speecht5_tts-pld-"
    return tuple(r[len(prefix):] for r in PINS if r.startswith(prefix))
