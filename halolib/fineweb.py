"""
FineWeb-compatible dataset utilities:
- add_fineweb_columns: adds source, language, token_count, content_hash
- dedup_against: filters a dataset against an existing repo's content_hash index
- push_with_retry: push_to_hub with exponential backoff
"""

import hashlib
import time

from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_dataset
from huggingface_hub import repo_exists

SPLIT_RATIO = 0.9  # 90% train, 10% test


def train_test_split(ds: DatasetDict, ratio: float = SPLIT_RATIO, seed: int = 42) -> DatasetDict:
    """
    Ensure a DatasetDict has train/test splits.
    If only train exists, splits it. If both exist, returns as-is.
    """
    if "test" in ds:
        return ds
    split = ds["train"].train_test_split(test_size=1 - ratio, seed=seed)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def add_fineweb_columns(
    ds: DatasetDict,
    source: str,
    language: str,
    num_proc: int = 1,
) -> DatasetDict:
    """Add source, language, token_count, content_hash columns to all splits."""

    def _add(batch):
        texts = batch["text"]
        batch["source"]       = [source] * len(texts)
        batch["language"]     = [language] * len(texts)
        batch["token_count"]  = [len(t.split()) for t in texts]
        batch["content_hash"] = [hashlib.md5((t or "").encode()).hexdigest() for t in texts]
        return batch

    return ds.map(_add, batched=True, batch_size=1000, num_proc=num_proc)


def dedup_against(
    ds: DatasetDict,
    target_repo: str,
    num_proc: int = 1,
) -> DatasetDict:
    """
    Filter ds to remove documents already present in target_repo.
    Uses content_hash column if available, otherwise recomputes from text.
    """
    if not repo_exists(target_repo, repo_type="dataset"):
        print(f"  Target {target_repo} does not exist — skipping dedup.")
        return ds

    print(f"  Loading existing dataset from {target_repo} for dedup...")
    existing = load_dataset(target_repo)

    seen: set[str] = set()
    for split in existing.keys():
        if "content_hash" in existing[split].column_names:
            seen.update(existing[split]["content_hash"])
        else:
            print(f"  No content_hash in {split}, recomputing from text...")
            hashes = existing[split].map(
                lambda batch: {"content_hash": [
                    hashlib.md5((t or "").encode()).hexdigest()
                    for t in batch["text"]
                ]},
                batched=True, batch_size=1000, num_proc=num_proc,
            )["content_hash"]
            seen.update(hashes)
    print(f"  {len(seen)} existing documents indexed")

    def is_new(batch):
        return [
            hashlib.md5((t or "").encode()).hexdigest() not in seen
            for t in batch["text"]
        ]

    deduped = {}
    for split in ds.keys():
        before = len(ds[split])
        deduped[split] = ds[split].filter(is_new, batched=True, batch_size=1000, num_proc=num_proc)
        after = len(deduped[split])
        print(f"  {split}: {before} -> {after} ({before - after} duplicates removed)")

    return DatasetDict(deduped)


def _align_columns(ds1: Dataset, ds2: Dataset) -> tuple:
    """Add missing columns as empty strings so two datasets have identical features."""
    all_cols = set(ds1.column_names) | set(ds2.column_names)
    for col in all_cols - set(ds1.column_names):
        ds1 = ds1.add_column(col, [""] * len(ds1))
    for col in all_cols - set(ds2.column_names):
        ds2 = ds2.add_column(col, [""] * len(ds2))
    return ds1, ds2


def append_to(
    ds: DatasetDict,
    target_repo: str,
    num_proc: int = 1,
) -> DatasetDict:
    """
    Deduplicate ds against target_repo then concatenate with existing splits.
    Returns the combined DatasetDict ready to push.
    """
    ds = dedup_against(ds, target_repo, num_proc=num_proc)

    if not repo_exists(target_repo, repo_type="dataset"):
        return ds

    existing = load_dataset(target_repo)
    combined = {}
    for split in ds.keys():
        if split in existing:
            old, new = _align_columns(existing[split], ds[split])
            combined[split] = concatenate_datasets([old, new])
        else:
            combined[split] = ds[split]
    for split in existing.keys():
        if split not in combined:
            combined[split] = existing[split]

    # align all splits against each other so the final DatasetDict is consistent
    all_cols = set()
    for split in combined.values():
        all_cols.update(split.column_names)
    for split_name in combined:
        for col in all_cols - set(combined[split_name].column_names):
            combined[split_name] = combined[split_name].add_column(col, [""] * len(combined[split_name]))

    return DatasetDict(combined)


def push_with_retry(
    ds: DatasetDict,
    repo: str,
    token: str,
    max_attempts: int = 5,
    wait: int = 30,
) -> None:
    """Push dataset to hub with retry on transient errors."""
    import sys
    for attempt in range(1, max_attempts + 1):
        try:
            ds.push_to_hub(repo, token=token)
            return
        except Exception as e:
            print(f"  attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                print(f"  retrying in {wait}s...")
                time.sleep(wait)
            else:
                print("  all retries exhausted.")
                sys.exit(1)
