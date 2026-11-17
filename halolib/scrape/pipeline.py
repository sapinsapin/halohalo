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
from .search import DEFAULT_BACKEND, fw2_excluded, get_backend
from .seeds import prepare_seeds

# Hosts whose content we must not redistribute regardless of language: song
# lyrics are copyrighted text, and lyrics sites turn up in Filipino search
# results (genius.com was in the first Tavily run's top five). Checked on
# every backend, FineWeb-2 included.
EXCLUDED_HOSTS = ("genius.com", "azlyrics.com", "lyrics.com", "musixmatch.com",
                  "lyricstranslate.com", "letras.com", "lyricsmode.com", "songlyrics.com",
                  "metrolyrics.com", "smule.com")


# Adult sites: not redistributable in a public research corpus. Matched by
# name fragments because the long tail is unbounded (the flywheel's first
# Filipino expansion pass selected filipinosexstories.com as a "known-good
# host" — it had produced accepted Filipino pages).
ADULT_HOST_FRAGMENTS = ("sexstories", "porn", "xxx", "hentai", "xvideo", "xnxx", "sextape",
                        "escort", "nsfw", "camgirl", "onlyfans", "adultfriend", "kantutan",
                        "sex-stories", "sexstory", "erotic",
                        # an audit of the shards on 2026-09-26 found these past
                        # the list above ("sex" alone would catch essex)
                        "sexgame", "sex-game", "sexspiel", "sexyhot", "incest", "kalibugan")

# Gambling: betting promotion is not corpus text, and online sabong
# (cockfight betting) was banned in the Philippines in 2022. The crawl found
# these by amplifying hosts FineWeb-2 had given us: todosabongga.blogspot.com
# went from 13 pages to 59 in two rounds.
GAMBLING_HOST_FRAGMENTS = ("casino", "sabong", "togel", "jackpot", "betting", "slots")
# Machine-translated game portals on a ph. subdomain: every page is a game
# blurb in template Filipino ("Maglaro nang Libre Online"); the crawl took
# ph.sgames.org from 12 pages to 71.
SPAM_HOSTS = ("sgames.org", "playgame24.com", "zxcgames.com", "igames7.com", "wsgames.ru",
              # a Dutch law firm machine-translated into Cebuano without a
              # language code in the URL (676 pages on Dutch administrative law)
              "lawandmore.services")


def excluded_host(url: str) -> bool:
    try:
        host = url.split("/")[2].lower()
    except IndexError:
        return False
    if any(frag in host for frag in ADULT_HOST_FRAGMENTS + GAMBLING_HOST_FRAGMENTS):
        return True
    return any(host == h or host.endswith("." + h) for h in EXCLUDED_HOSTS + SPAM_HOSTS)


# Machine-translation farms put the language code in the URL *path* —
# alltechbuzz.net/ceb/..., qc-solar.com/pag/..., pilotech.ai/pam/... — corporate
# and content sites machine-translating their copy into every language. Of
# ~450 accepted hosts matching this shape on 2026-09-23, nearly all were
# manufacturers, casino sites and Spanish tech blogs "in Cebuano". The rule
# is the subdomain rule one slash later. The exceptions are publishers whose
# translations are made by people, and they are listed by name.
MT_PATH_CODES = {"ceb", "pag", "pam", "bik", "bcl", "war", "ilo", "hil", "tl", "fil", "tgl", "tsg"}
HUMAN_TRANSLATION_HOSTS = (
    "jw.org", "ebible.org", "bible.com", "bible.is", "biblegateway.com", "biblia.chat",
    "mormon.org", "lds.org", "churchofjesuschrist.org", "egwwritings.org",
    "globalrecordings.net", "dawnbible.com", "christianscience.com",
    "digitallibrary.io", "familysearch.org", "fema.gov", "justice.gov", "wikimedia.org",
)


def mt_path_excluded(url: str) -> bool:
    try:
        _, _, host, seg = url.split("/", 3)[:4] if url.count("/") >= 3 else (None, None, url.split("/")[2], "")
    except (IndexError, ValueError):
        return False
    host = host.lower()
    first = seg.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].lower()
    if first not in MT_PATH_CODES:
        return False
    return not any(host == h or host.endswith("." + h) for h in HUMAN_TRANSLATION_HOSTS)


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
    # A whole Cebuano Bible (477k words, ccel.org) was 84 % of Cebuano's first
    # Tavily run by words. One page should not be a language's corpus, and
    # scripture is already over-represented in these languages' web text.
    max_words: int = 20000
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
        table = pa.Table.from_pylist(self.buf, schema=shard_schema())
        write_shard_durably(table, self.dir / f"shard-{self.n_shards:05d}.parquet")
        self.n_shards += 1
        self.buf = []


def shard_schema():
    import pyarrow as pa
    return pa.schema(
        [(c, pa.string()) if c not in ("word_count", "token_count") else (c, pa.int64())
         for c in COLUMNS if c not in ("lid_score", "lid_agreement", "lid_models_agree")]
        + [("lid_score", pa.float64()), ("lid_agreement", pa.float64()),
           ("lid_models_agree", pa.bool_())])


def write_shard_durably(table, path: Path) -> None:
    """Write to a temp file, fsync, then rename. A WSL VM killed after a plain
    write (OOM or a kernel panic) left ceb/shard-00033 with a valid footer and
    garbage pages: its data had not reached the Windows disk."""
    import pyarrow.parquet as pq
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp)
    with open(tmp, "rb+") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def quarantine_corrupt_shards(text_dir: Path) -> list[Path]:
    """Move unreadable shards to {lang}/quarantine/ and leave an empty, valid
    shard in their place: shard numbering counts files, so the slot must stay
    filled. Their rows are then missing from the shards while their URLs stay
    `accepted` in the manifest, so Manifest.forget_unsaved re-fetches them.
    One bad shard otherwise stops every reader (seeds, dedup, LID training)."""
    import pyarrow.parquet as pq
    bad = []
    for shard in sorted(text_dir.glob("*/shard-*.parquet")):
        try:
            pq.read_table(shard)
        except Exception as exc:                  # OSError, ArrowInvalid: any unreadable file
            qdir = shard.parent / "quarantine"
            qdir.mkdir(exist_ok=True)
            os.replace(shard, qdir / f"{shard.stem}.{int(time.time())}.parquet")
            write_shard_durably(shard_schema().empty_table(), shard)
            print(f"  quarantined {shard.parent.name}/{shard.name}: {type(exc).__name__}: {str(exc)[:60]}")
            bad.append(shard)
    return bad


class Manifest:
    """One JSON line per URL examined; the resume set."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen: set[str] = set()
        self.accepted: dict[str, str] = {}      # url -> content_hash, last record wins
        if path.exists():
            for ln in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(ln)
                    self.seen.add(r["url"])
                except (json.JSONDecodeError, KeyError):
                    continue
                if r.get("status") == "accepted" and r.get("hash"):
                    self.accepted[r["url"]] = r["hash"]
                else:
                    self.accepted.pop(r["url"], None)
        self._fh = open(path, "a", encoding="utf-8")

    def forget_unsaved(self, saved_hashes: set[str]) -> int:
        """Accepted rows wait in memory until the language's pass flushes
        them, but their manifest line is written at once. A process that dies
        in between (round 1 during the LID retrain, round 11 when WSL was
        restarted) leaves URLs marked accepted whose text was never saved;
        un-see them so this run fetches them again."""
        lost = [u for u, h in self.accepted.items() if h not in saved_hashes]
        for u in lost:
            self.seen.discard(u)
            del self.accepted[u]
        return len(lost)

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
        unsaved = manifest.forget_unsaved(dedup.hashes)
        stats = {"queries": 0, "hits": 0, "skipped_seen": 0, "fetched": 0,
                 "raw_fallback": 0, "fetch_fail": 0, "too_short": 0, "too_long": 0,
                 "lid_reject": 0, "dup": 0, "host_excluded": 0, "host_capped": 0,
                 "accepted": 0, "resumed_rows": seeded}
        per_host: dict[str, int] = {}
        print(f"\n[{lang}] backend={backend.name} resumed={seeded} rows, "
              f"{len(manifest.seen)} urls in manifest"
              + (f", {unsaved} accepted but never saved (will refetch)" if unsaved else ""))
        t0 = time.time()

        # The seed cache can hold more queries than this pass may spend (the
        # flywheel's expansion pass reuses the seed pass's cache): slice, or
        # an "expand 3 queries" pass silently runs all six.
        queries = seeds[lang]["queries"][:cfg.queries_per_lang]
        if getattr(backend, "ignores_query", False):
            queries = ["(direct)"]
        for query in queries:
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
                # lyrics sites (copyright) on every backend, and the ceb/war
                # bot-Wikipedia + MT-farm rule that FineWeb-2 ingestion already
                # applies — search backends find those same hosts
                if excluded_host(hit.url) or fw2_excluded(hit.url, lang) or mt_path_excluded(hit.url):
                    stats["host_excluded"] += 1
                    manifest.record(hit.url, "host_excluded", lang=lang)
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
                    try:
                        page = fetcher.fetch(hit.url)
                    except Exception as exc:        # one bad page must not end the run
                        print(f"  fetch error {hit.url[:80]}: {type(exc).__name__}: {exc}")
                        page = None
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
                n_words = len(cleaned.split())
                if cfg.max_words and n_words > cfg.max_words:
                    stats["too_long"] += 1
                    manifest.record(hit.url, "too_long", lang=lang, words=n_words)
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
