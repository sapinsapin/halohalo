"""
Exact and near-duplicate detection.

Exact: md5 of the cleaned text — the same `content_hash` prep_halohalo.py and
add_fineweb_columns already write, so a scrape can be deduplicated against
what is on the Hub without recomputing anything.

Near: MinHash over 5-word shingles with LSH at Jaccard 0.75 (the FineWeb
near-dedup convention). Regional news
sites syndicate the same article under several URLs with different headers
and footers; exact hashing misses all of those, LSH catches them. Index is
in-memory per run and seeded from earlier shards on disk so a resumed run
does not re-admit yesterday's pages.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

WORD = re.compile(r"\w+", re.UNICODE)


def content_hash(text: str) -> str:
    return hashlib.md5((text or "").encode()).hexdigest()


class DedupIndex:
    def __init__(self, threshold: float = 0.75, num_perm: int = 128, shingle: int = 5):
        from datasketch import MinHash, MinHashLSH
        self._MinHash = MinHash
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.shingle = shingle
        self.hashes: set[str] = set()
        self.n_exact = self.n_near = 0

    def _minhash(self, text: str):
        words = WORD.findall(text.lower())
        m = self._MinHash(num_perm=self.num_perm)
        k = self.shingle
        if len(words) < k:
            m.update(" ".join(words).encode())
            return m
        for i in range(len(words) - k + 1):
            m.update(" ".join(words[i:i + k]).encode())
        return m

    def seed_from_parquet(self, paths: list[Path]) -> int:
        """Load content_hash + text of existing shards so a resumed run treats
        them as already present."""
        import pyarrow.parquet as pq
        n = 0
        for p in paths:
            t = pq.read_table(p, columns=["content_hash", "text"])
            for h, txt in zip(t.column("content_hash").to_pylist(), t.column("text").to_pylist()):
                self.hashes.add(h)
                key = f"seed:{h}"
                if key not in self.lsh:
                    self.lsh.insert(key, self._minhash(txt or ""))
                n += 1
        return n

    def check_and_add(self, key: str, text: str) -> str | None:
        """Return None if new (and index it), else the reason it's a dup."""
        h = content_hash(text)
        if h in self.hashes:
            self.n_exact += 1
            return "exact"
        m = self._minhash(text)
        if self.lsh.query(m):
            self.n_near += 1
            return "near"
        self.hashes.add(h)
        self.lsh.insert(key, m)
        return None
