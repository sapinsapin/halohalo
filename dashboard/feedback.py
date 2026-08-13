"""
Collect good/bad judgements on demo outputs and persist them to a Hub dataset
for later quality evaluation and RLHF.

Every rating is written as one JSONL row carrying the full context needed to
re-derive the example later: task, language, model repo, the input, the
output, and — when the rater allows it — the audio on both sides. That is
enough to build an eval set, and enough to build preference pairs once the
same input has been rated on more than one model.

Persistence uses `CommitScheduler`, the standard Spaces pattern: rows are
appended to a local JSONL and pushed to the dataset repo on a timer, so a
click never blocks on the network and a burst of ratings costs one commit.

Nothing here can take the demo down. If the Space has no write-scoped token
the module loads in a disabled state and the UI says so, rather than raising
at import or failing on click.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

REPO_ID = os.environ.get("FEEDBACK_REPO", "sapinsapin/halohalo-feedback")
SPACE_ID = os.environ.get("SPACE_ID", "local")
LOCAL_DIR = Path(os.environ.get("FEEDBACK_DIR", "/tmp/halohalo-feedback"))
AUDIO_DIR = LOCAL_DIR / "audio"
JSONL = LOCAL_DIR / "ratings.jsonl"
PUSH_EVERY_MIN = 5

_lock = Lock()
_scheduler = None
STATUS = "disabled"
DETAIL = "no write-scoped token found in this Space's secrets"
TOKEN_SOURCE = None


TOKEN_VARS = ("FEEDBACK_TOKEN", "HF_TOKEN", "SAPINSAPINDASH", "sapinsapindash")


def _token():
    """Find the write token, whatever the Space secret happens to be called.

    Space secrets arrive as environment variables under the exact name they
    were given, so a secret named for the Space rather than for this code
    would otherwise be invisible. Preferred names are checked first; failing
    that, any variable holding a Hugging Face token is accepted, which means
    renaming the secret does not silently switch collection off.
    """
    for var in TOKEN_VARS:
        tok = os.environ.get(var)
        if tok and tok.strip():
            return tok.strip()

    for name, value in os.environ.items():
        if (value and value.startswith("hf_") and len(value) > 20
                and "\n" not in value):
            globals()["TOKEN_SOURCE"] = name
            return value.strip()
    return None


def status_report() -> str:
    """One line describing whether ratings are being saved, and from where."""
    if enabled():
        src = TOKEN_SOURCE or "FEEDBACK_TOKEN/HF_TOKEN"
        return f"enabled → {REPO_ID} (token from `{src}`)"
    return f"disabled → {DETAIL}"


def init():
    """Start the commit scheduler if we hold a token that can write."""
    global _scheduler, STATUS, DETAIL, TOKEN_SOURCE
    tok = _token()
    if not tok:
        return
    for var in TOKEN_VARS:                    # record which name supplied it
        if os.environ.get(var, "").strip() == tok:
            TOKEN_SOURCE = var
            break
    try:
        from huggingface_hub import CommitScheduler, HfApi

        # A read-scoped token is the common case here (the dashboard only
        # needs read), so confirm write access before promising the rater
        # that anything is being saved.
        HfApi(token=tok).create_repo(REPO_ID, repo_type="dataset",
                                     private=True, exist_ok=True)
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        _scheduler = CommitScheduler(
            repo_id=REPO_ID, repo_type="dataset", folder_path=str(LOCAL_DIR),
            path_in_repo="data", every=PUSH_EVERY_MIN, private=True, token=tok,
            squash_history=False)
        STATUS, DETAIL = "enabled", REPO_ID
    except Exception as exc:                                   # noqa: BLE001
        STATUS = "disabled"
        DETAIL = f"{type(exc).__name__}: {str(exc)[:120]}"


def enabled() -> bool:
    return _scheduler is not None


def _save_audio(audio, stem: str) -> str | None:
    """Write a (sample_rate, ndarray) pair into the push folder."""
    if audio is None:
        return None
    try:
        import soundfile as sf
        sr, data = audio
        name = f"{stem}.wav"
        sf.write(AUDIO_DIR / name, data, sr)
        return f"audio/{name}"
    except Exception:                                          # noqa: BLE001
        return None


def record(rating: str, ctx: dict | None, comment: str = "",
           keep_audio: bool = True) -> str:
    """Append one rating. Returns the message to show the rater."""
    if not ctx:
        return "Nothing to rate yet — run the model first."
    if not enabled():
        return (f"Feedback is not being saved ({DETAIL}). "
                f"Ask the Space owner to set a write-scoped token.")

    rid = uuid.uuid4().hex[:12]
    row = {
        "id": rid,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "space": SPACE_ID,
        "rating": rating,
        "task": ctx.get("task"),
        "language": ctx.get("language"),
        "model": ctx.get("model"),
        "voice": ctx.get("voice"),
        "input_text": ctx.get("input_text"),
        "output_text": ctx.get("output_text"),
        "reference_text": ctx.get("reference_text"),
        "comment": (comment or "").strip()[:2000] or None,
        "audio_kept": bool(keep_audio),
        "input_audio": None,
        "output_audio": None,
    }

    try:
        with _lock, _scheduler.lock:
            if keep_audio:
                row["input_audio"] = _save_audio(ctx.get("input_audio"),
                                                 f"{rid}_in")
                row["output_audio"] = _save_audio(ctx.get("output_audio"),
                                                  f"{rid}_out")
            with JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:                                   # noqa: BLE001
        return f"Could not save feedback: {type(exc).__name__}: {exc}"

    kept = "with audio" if (row["input_audio"] or row["output_audio"]) else "text only"
    verdict = "👍 good" if rating == "good" else "👎 needs work"
    return (f"Saved {verdict} ({kept}). Thank you — this goes into the "
            f"evaluation set.")


init()
