"""Resolve the halo-collector resource IDs and repo settings.

`load_ids()` reads `HALO_AGENT_ID` / `HALO_ENVIRONMENT_ID` / `HALO_VAULT_ID`
from the environment first and falls back to `claude-lock.json` at the repo
root (written by `ant apply`). Missing values come back as ``None`` so callers
can decide whether to fail or to run in demo mode.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = REPO_ROOT / "claude-lock.json"

REPO_URL = "https://github.com/sapinsapin/halohalo"
REPO_BRANCH = "main"

_ENV_KEYS = {
    "agent_id": "HALO_AGENT_ID",
    "environment_id": "HALO_ENVIRONMENT_ID",
    "vault_id": "HALO_VAULT_ID",
}
_LOCK_KEYS = {
    "agent_id": "./agents/halo-collector.md",
    "environment_id": "./environments/halo-collector.yaml",
    "vault_id": "./vaults/halo-collector.yaml",
}


def _read_lock(lock_file: Path) -> dict[str, Any]:
    try:
        with open(lock_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    resources = data.get("resources") if isinstance(data, dict) else None
    return resources if isinstance(resources, dict) else {}


def _lock_id(resources: dict[str, Any], key: str) -> str | None:
    entry = resources.get(key)
    if isinstance(entry, dict):
        value = entry.get("id")
        return value if isinstance(value, str) and value else None
    return None


def load_ids(lock_file: Path | str | None = None) -> dict[str, str | None]:
    """Return ``{"agent_id", "environment_id", "vault_id"}`` (None when unknown)."""
    resources = _read_lock(Path(lock_file) if lock_file else LOCK_FILE)
    out: dict[str, str | None] = {}
    for field, env_key in _ENV_KEYS.items():
        value = os.environ.get(env_key) or None
        if value is None:
            value = _lock_id(resources, _LOCK_KEYS[field])
        out[field] = value
    return out


def repo_settings() -> dict[str, str | None]:
    """Return ``{"url", "branch", "token"}``; token is ``GITHUB_TOKEN`` or None."""
    return {
        "url": REPO_URL,
        "branch": REPO_BRANCH,
        "token": os.environ.get("GITHUB_TOKEN") or None,
    }
