"""
Parquet shard streaming to the Hugging Face Hub, generalized from
process_fsc_parquet.py.

Pattern: write shard → upload in background thread → unlink, with exactly one
upload overlapping the next shard's compute, so local disk never holds more
than ~2 shards. Resume via progress.jsonl keyed on "{prefix}/{split}/{idx}".
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from datasets import Audio, Dataset
from huggingface_hub import HfApi

log = logging.getLogger(__name__)


def rows_to_dataset(rows: list[dict], sr: int) -> Dataset:
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    ds = Dataset.from_dict(cols)
    return ds.cast_column("audio", Audio(sampling_rate=sr))


class ShardStreamer:
    """Streams row batches as parquet shards to a Hub dataset repo.

    prefix distinguishes configs within one repo (e.g. "data/asr", "data/tts").
    """

    def __init__(self, repo: str, token: str, work_dir: Path,
                 prefix: str = "data", shard_size: int = 2000,
                 private: bool = True):
        self.repo = repo
        self.token = token
        self.private = private
        self.work_dir = work_dir
        self.prefix = prefix.strip("/")
        self.shard_size = shard_size
        self.resume_log = work_dir / "progress.jsonl"
        self._upload_pool = None
        self._upload_future = None
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def completed(self) -> set[str]:
        """Set of '{prefix}/{split}/{shard_idx}' keys already uploaded."""
        if not self.resume_log.exists():
            return set()
        done = set()
        with self.resume_log.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                    done.add(f"{e['prefix']}/{e['split']}/{e['shard_idx']}")
                except (json.JSONDecodeError, KeyError):
                    pass
        return done

    def _log_done(self, split: str, shard_idx: int, shard_name: str, rows: int, mb: float):
        with self.resume_log.open("a") as f:
            f.write(json.dumps({
                "prefix":     self.prefix,
                "split":      split,
                "shard_idx":  shard_idx,
                "shard_name": shard_name,
                "rows":       rows,
                "mb":         round(mb, 2),
                "ts":         datetime.now(timezone.utc).isoformat(),
            }) + "\n")

    def __enter__(self):
        HfApi(token=self.token).create_repo(
            repo_id=self.repo, repo_type="dataset", exist_ok=True,
            private=self.private)
        self._upload_pool = ThreadPoolExecutor(max_workers=1)
        return self

    def __exit__(self, *exc):
        if self._upload_future is not None:
            self._upload_future.result()
        self._upload_pool.shutdown()
        return False

    def _upload_and_delete(self, path: Path, repo_path: str,
                           split: str, shard_idx: int, rows: int, mb: float):
        log.info(f"  uploading {path.name}...")
        HfApi(token=self.token).upload_file(
            path_or_fileobj=str(path),
            path_in_repo=repo_path,
            repo_id=self.repo,
            repo_type="dataset",
        )
        path.unlink()
        self._log_done(split, shard_idx, path.name, rows, mb)
        log.info(f"  uploaded + logged: {path.name}")

    def write_shard(self, split: str, shard_idx: int, n_shards: int,
                    rows: list[dict], sr: int) -> None:
        """Write one shard and queue its upload (blocks on the previous upload)."""
        if not rows:
            log.warning(f"  shard {shard_idx+1}/{n_shards} produced 0 rows, skipping")
            return

        ds = rows_to_dataset(rows, sr)
        safe_prefix = self.prefix.replace("/", "-")
        shard_name = f"{safe_prefix}-{split}-{shard_idx:05d}-of-{n_shards:05d}.parquet"
        shard_path = self.work_dir / shard_name
        ds.to_parquet(str(shard_path))
        mb = shard_path.stat().st_size / 1024 / 1024
        log.info(f"  shard {shard_idx+1}/{n_shards} → {shard_name} ({mb:.1f} MB, {len(rows)} rows)")

        if self._upload_future is not None:
            self._upload_future.result()
        self._upload_future = self._upload_pool.submit(
            self._upload_and_delete, shard_path,
            f"{self.prefix}/{split}/{shard_name}",
            split, shard_idx, len(rows), mb,
        )
