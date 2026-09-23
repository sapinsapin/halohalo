"""
search -> (fetch) -> LID gate -> clean -> dedup -> FineWeb-schema shards.

Resumable by construction: every URL that was ever looked at is a line in
`{lang}/manifest.jsonl` with its outcome, and a re-run skips them. Accepted
rows go to parquet shards in the same column layout as sapinsapin/halohalo so
prep_halohalo.py / append_to can publish them unchanged. The LID verdict is
kept per row (both models, their agreement) — the point is not just to
filter, but to keep the evidence so the filter can be audited and so
disagreements can feed the next LID training round.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from halolib import clean_text, is_usable
from halolib.lid import LANGS, Ensemble

from .dedup import DedupIndex, content_hash
from .fetch import Fetcher
from .search import DEFAULT_BACKEND, get_backend
from .seeds import prepare_seeds

COLUMNS = ["id", "text", "url", "date", "dump", "file_path", "detected_lang",
           "word_count", "title", "source", "language", "token_count",
           "content_hash", "crawled_at",
           # scrape-specific provenance
           "backend", "query", "lid_model", "lid_score", "lid_agreement",
           "lid_models_agree", "lid_detail"]


@dataclass
class ScrapeConfig:
    out_dir: Path
    langs: tuple[str, ...] = LANGS
    backend: str = DEFAULT_BACKEND
    backend_kwargs: dict = field(default_factory=dict)
    queries_per_lang: int = 20
    max_results: int = 10
    max_docs_per_lang: int = 500
    min_words: int = 30
    lid_min_score: float = 0.6
    lid_min_agreement: float = 0.6
    shard_rows: int = 1000
    use_glotlid: bool = True
    refresh_seeds: bool = False
    # Cap accepted documents per host per language. Off by default: for the
    # smallest languages one site may be most of what exists, and the CPT mix
    # is the right place to rebalance. The summary always reports the top
    # hosts so the skew is visible either way (Bikol's first Tavily run was
    # 41 % jw.org).
    max_per_host: int | None = None


class ShardWriter:
    def __init__(self, lang_dir: Path, rows_per_shard: int):
        self.dir = lang_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard
        self.buf: list[dict] = []
        self.n_shards = len(list(self.dir.glob("shard-*.parquet")))
        self.n_rows = 0

    def existing_shards(self) -> list[Path]:
        return sorted(self.dir.glob("shard-*.parquet"))

    def add(self, row: dict) -> None:
        self.buf.append(row)
        self.n_rows += 1
        if len(self.buf) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(self.buf, schema=pa.schema(
            [(c, pa.string()) if c not in ("word_count", "token_count") else (c, pa.int64())
             for c in COLUMNS if c not in ("lid_score", "lid_agreement", "lid_models_agree")]
            + [("lid_score", pa.float64()), ("lid_agreement", pa.float64()),
               ("lid_models_agree", pa.bool_())]))
        path = self.dir / f"shard-{self.n_shards:05d}.parquet"
        pq.write_table(table, path)
        self.n_shards += 1
        self.buf = []


class Manifest:
    """One JSON line per URL examined; the resume set."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen: set[str] = set()
        if path.exists():
            for ln in path.read_text(encoding="utf-8").splitlines():
                try:
                    self.seen.add(json.loads(ln)["url"])
                except (json.JSONDecodeError, KeyError):
                    pass
        self._fh = open(path, "a", encoding="utf-8")

    def record(self, url: str, status: str, **extra) -> None:
        self.seen.add(url)
        self._fh.write(json.dumps({"url": url, "status": status,
                                   "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   **extra}, ensure_ascii=False) + "\n")
        self._fh.flush()


def make_row(lang: str, url: str, text: str, title: str, date: str | None,
             backend: str, query: str, verdict, extra: dict) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, url)),
        "text": text,
        "url": url,
        "date": date or extra.get("date") or "",
        "dump": extra.get("dump") or f"web-{backend}-{now:%Y%m}",
        "file_path": "",
        "detected_lang": verdict.lang,
        "word_count": len(text.split()),
        "title": title or "",
        "source": f"web:{backend}",
        "language": lang,
        "token_count": len(text.split()),
        "content_hash": content_hash(text),
        "crawled_at": now.isoformat(timespec="seconds"),
        "backend": backend,
        "query": query,
        "lid_model": "halolid+glotlid" if verdict.detail.get("halolid") and verdict.detail.get("glotlid")
                     else next(iter(verdict.detail)),
        "lid_score": float(verdict.score),
        "lid_agreement": float(verdict.agreement),
        "lid_models_agree": bool(verdict.models_agree),
        "lid_detail": json.dumps(verdict.detail, ensure_ascii=False),
    }


def run_text(cfg: ScrapeConfig, lid: Ensemble | None = None) -> dict[str, dict]:
    """Scrape every language in cfg. Returns per-language counters."""
    from halolib.lid import default_ensemble
    lid = lid or default_ensemble(cfg.use_glotlid)

    seeds = prepare_seeds(cfg.out_dir / "seeds", langs=cfg.langs,
                          n_queries=cfg.queries_per_lang, refresh=cfg.refresh_seeds,
                          token=os.environ.get("HF_TOKEN"))
    backend = get_backend(cfg.backend, **cfg.backend_kwargs)
    fetcher = Fetcher()
    summary = {}

    for lang in cfg.langs:
        lang_dir = cfg.out_dir / "text" / lang
        writer = ShardWriter(lang_dir, cfg.shard_rows)
        manifest = Manifest(lang_dir / "manifest.jsonl")
        dedup = DedupIndex()
        seeded = dedup.seed_from_parquet(writer.existing_shards())
        stats = {"queries": 0, "hits": 0, "skipped_seen": 0, "fetched": 0,
                 "raw_fallback": 0, "fetch_fail": 0, "too_short": 0,
                 "lid_reject": 0, "dup": 0, "host_capped": 0, "accepted": 0,
                 "resumed_rows": seeded}
        per_host: dict[str, int] = {}
        print(f"\n[{lang}] backend={backend.name} resumed={seeded} rows, "
              f"{len(manifest.seen)} urls in manifest")
        t0 = time.time()

        for query in seeds[lang]["queries"]:
            if stats["accepted"] >= cfg.max_docs_per_lang:
                break
            stats["queries"] += 1
            try:
                hits = backend.search(query, lang, cfg.max_results)
            except Exception as exc:
                print(f"  search failed for {query!r}: {type(exc).__name__}: {exc}")
                continue
            stats["hits"] += len(hits)

            for hit in hits:
                if stats["accepted"] >= cfg.max_docs_per_lang:
                    break
                if hit.url in manifest.seen:
                    stats["skipped_seen"] += 1
                    continue
                host = hit.url.split("/")[2].lower() if hit.url.count("/") >= 2 else "?"
                if cfg.max_per_host and per_host.get(host, 0) >= cfg.max_per_host:
                    stats["host_capped"] += 1
                    manifest.record(hit.url, "host_capped", lang=lang, host=host)
                    continue

                text, title, date = hit.raw_text, hit.title, hit.extra.get("date")
                # A search backend's raw_content is the whole page, menus and
                # all. Extract the main text from the live page instead, and
                # fall back to the dump only when the fetch gives us nothing.
                if not text or hit.extra.get("page_dump"):
                    page = fetcher.fetch(hit.url)
                    if page is not None and page.text and len(page.text.split()) >= cfg.min_words:
                        text, title, date = page.text, page.title or title, page.date or date
                        stats["fetched"] += 1
                    elif not text:
                        stats["fetch_fail"] += 1
                        manifest.record(hit.url, "fetch_fail", lang=lang,
                                        http=getattr(page, "status", 0))
                        continue
                    else:
                        stats["raw_fallback"] += 1

                cleaned = clean_text(text)
                if not is_usable(cleaned, min_words=cfg.min_words):
                    stats["too_short"] += 1
                    manifest.record(hit.url, "too_short", lang=lang)
                    continue

                verdict = lid.identify(cleaned)
                if not verdict.accept(lang, cfg.lid_min_score, cfg.lid_min_agreement):
                    stats["lid_reject"] += 1
                    manifest.record(hit.url, "lid_reject", lang=lang, got=verdict.lang,
                                    score=round(verdict.score, 3),
                                    agreement=round(verdict.agreement, 3))
                    continue

                why = dedup.check_and_add(hit.url, cleaned)
                if why:
                    stats["dup"] += 1
                    manifest.record(hit.url, f"dup_{why}", lang=lang)
                    continue

                row = make_row(lang, hit.url, cleaned, title, date, backend.name,
                               hit.query, verdict, hit.extra)
                writer.add(row)
                manifest.record(hit.url, "accepted", lang=lang, hash=row["content_hash"],
                                words=row["word_count"], lid=round(verdict.score, 3))
                stats["accepted"] += 1
                per_host[host] = per_host.get(host, 0) + 1

        writer.flush()
        stats["seconds"] = round(time.time() - t0, 1)
        top = sorted(per_host.items(), key=lambda kv: -kv[1])[:5]
        stats["top_hosts"] = {h: n for h, n in top}
        summary[lang] = stats
        print(f"[{lang}] " + "  ".join(f"{k}={v}" for k, v in stats.items() if k != "top_hosts"))
        if top and stats["accepted"]:
            print(f"[{lang}] top hosts: " + ", ".join(
                f"{h} {n} ({n / stats['accepted']:.0%})" for h, n in top))

    (cfg.out_dir / "text" / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_lang_dataset(out_dir: Path, lang: str):
    """All shards for one language as a datasets.Dataset (or None)."""
    from datasets import Dataset, concatenate_datasets
    shards = sorted((out_dir / "text" / lang).glob("shard-*.parquet"))
    if not shards:
        return None
    return concatenate_datasets([Dataset.from_parquet(str(p)) for p in shards])


def push_lang(out_dir: Path, lang: str, repo: str, num_proc: int = 1) -> int:
    """Append a language's shards to a Hub dataset, deduplicating by
    content_hash against what is already there (halolib.fineweb.append_to)."""
    from datasets import DatasetDict

    from halolib.fineweb import append_to, push_with_retry, train_test_split
    ds = load_lang_dataset(out_dir, lang)
    if ds is None or len(ds) == 0:
        print(f"[{lang}] nothing to push")
        return 0
    n = len(ds)
    dd = train_test_split(DatasetDict({"train": ds}))
    combined = append_to(dd, repo, num_proc=num_proc)
    push_with_retry(combined, repo, os.environ["HF_TOKEN"])
    print(f"[{lang}] pushed {n} new rows -> {repo}")
    return n
