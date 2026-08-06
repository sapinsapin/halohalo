"""
Per-file manifests — the backbone of incremental processing.

Each source file gets:
  {manifest_dir}/{file_id}.jsonl        — one row per turn (segment)
  {manifest_dir}/{file_id}.status.json  — per-stage completion stamps

Expensive stages (align, qc) write their columns into the manifest exactly
once; exports are cheap re-runs over manifests. A stage stamp records the
config hash it ran with, so changing thresholds/models invalidates only the
affected stages on the next run.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def config_hash(config: dict) -> str:
    """Stable hash of the parameters a stage depends on."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def manifest_path(manifest_dir: Path, file_id: str) -> Path:
    return manifest_dir / f"{file_id}.jsonl"


def status_path(manifest_dir: Path, file_id: str) -> Path:
    return manifest_dir / f"{file_id}.status.json"


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _read_status(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def stage_done(path: Path, stage: str, cfg_hash: str) -> bool:
    """True if `stage` completed with the same config hash."""
    entry = _read_status(path).get(stage)
    return entry is not None and entry.get("config") == cfg_hash


def mark_stage(path: Path, stage: str, cfg_hash: str, **extra) -> None:
    status = _read_status(path)
    status[stage] = {
        "config": cfg_hash,
        "ts": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    tmp.replace(path)


def clear_stage(path: Path, stage: str) -> None:
    status = _read_status(path)
    if stage in status:
        del status[stage]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)


def get_stage(path: Path, stage: str) -> dict | None:
    """Return the full status entry for a stage (config, ts, extras)."""
    return _read_status(path).get(stage)
