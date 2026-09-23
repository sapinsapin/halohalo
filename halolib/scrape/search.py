"""
Discovery backends. All of them yield `Hit`s; the pipeline does not care
where a page came from, only that it records which backend and query found it.

  tavily    — DEFAULT. Search API with `include_raw_content`, so most pages
              arrive with their main text already extracted and never need a
              fetch. Needs TAVILY_API_KEY.
  direct    — a file of URLs, one per line. No search; every URL is a hit.
              What you use for a curated domain list or a smoke test.
  fineweb2  — HuggingFaceFW/fineweb-2, streamed per language. Not a search
              engine: it is the CommonCrawl-derived corpus the halo-* sets
              were built from, kept here so the same LID gate, cleaner and
              dedup apply to it as to fresh scrapes. Applies ingest_fineweb2's
              bot-Wikipedia and MT-content-farm exclusions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterator, Protocol

DEFAULT_BACKEND = "tavily"


class MissingCredential(RuntimeError):
    pass


@dataclass
class Hit:
    url: str
    title: str = ""
    snippet: str = ""
    raw_text: str | None = None       # main text if the backend already has it
    backend: str = ""
    query: str = ""
    extra: dict = field(default_factory=dict)


class SearchBackend(Protocol):
    name: str

    def search(self, query: str, lang: str, max_results: int) -> list[Hit]: ...


# ----------------------------------------------------------------------------
# Tavily (default)
# ----------------------------------------------------------------------------

# Login-walled or video platforms: Tavily returns them, but there is no page
# text to extract, so each hit is a wasted credit and a wasted fetch. The
# first hil query returned three Facebook posts with 0 characters of text.
SOCIAL_DOMAINS = ["facebook.com", "instagram.com", "tiktok.com", "twitter.com",
                  "x.com", "youtube.com", "pinterest.com", "threads.net"]


class TavilyBackend:
    name = "tavily"

    def __init__(self, api_key: str | None = None, search_depth: str = "advanced",
                 client=None, exclude_domains: list[str] | None = None,
                 include_domains: list[str] | None = None):
        # include_domains restricts a search to sites we already know publish
        # in the language — the flywheel's host-expansion pass. Such searches
        # can use "basic" depth (1 credit instead of 2): the site does the
        # narrowing, the ranking has little left to do.
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if client is None and not key:
            raise MissingCredential(
                "TAVILY_API_KEY is not set. Add it to .env (https://app.tavily.com) "
                "or choose another backend with --backend direct|fineweb2.")
        if client is None:
            from tavily import TavilyClient
            client = TavilyClient(api_key=key)
        self.client = client
        self.search_depth = search_depth
        self.exclude_domains = SOCIAL_DOMAINS if exclude_domains is None else exclude_domains
        self.include_domains = include_domains or None

    def search(self, query: str, lang: str, max_results: int = 10) -> list[Hit]:
        kwargs = dict(
            query=query,
            search_depth=self.search_depth,
            max_results=max_results,
            include_raw_content=True,      # full page text, so no fetch needed
            exclude_domains=self.exclude_domains,
        )
        if self.include_domains:
            kwargs["include_domains"] = self.include_domains
        resp = self.client.search(**kwargs)
        hits = []
        for r in resp.get("results", []):
            url = r.get("url")
            if not url:
                continue
            hits.append(Hit(
                url=url, title=r.get("title") or "", snippet=r.get("content") or "",
                raw_text=r.get("raw_content") or None, backend=self.name, query=query,
                # raw_content is a whole-page dump (menus included), so the
                # pipeline re-extracts from the live page when it can
                extra={"score": r.get("score"), "page_dump": True},
            ))
        return hits

    def extract(self, urls: list[str]) -> dict[str, str]:
        """Tavily's extractor for hits that arrived without raw_content."""
        out = {}
        for i in range(0, len(urls), 20):
            resp = self.client.extract(urls=urls[i:i + 20])
            for r in resp.get("results", []):
                if r.get("raw_content"):
                    out[r["url"]] = r["raw_content"]
        return out


# ----------------------------------------------------------------------------
# Direct URL list
# ----------------------------------------------------------------------------

class DirectBackend:
    name = "direct"

    def __init__(self, urls_file: str | None = None, urls: list[str] | None = None):
        if urls is None:
            if not urls_file:
                raise ValueError("direct backend needs --urls <file>")
            with open(urls_file, encoding="utf-8") as f:
                urls = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        self.urls = urls
        self._served: set[str] = set()

    def search(self, query: str, lang: str, max_results: int = 10) -> list[Hit]:
        # ignore the query: serve the whole list once per language, and let
        # the LID gate decide which pages belong to which language
        if lang in self._served:
            return []
        self._served.add(lang)
        return [Hit(url=u, backend=self.name, query="(direct)") for u in self.urls]


# ----------------------------------------------------------------------------
# FineWeb-2 (retained corpus source)
# ----------------------------------------------------------------------------

FW2_CONFIG = {
    "bcl": "bcl_Latn", "ceb": "ceb_Latn", "eng": "eng_Latn", "fil": "fil_Latn",
    "hil": "hil_Latn", "ilo": "ilo_Latn", "pag": "pag_Latn", "pam": "pam_Latn",
    "tsg": "tsg_Latn", "war": "war_Latn",
}

# Same exclusions as scripts/ingest_fineweb2.py: Lsjbot encyclopedias for
# ceb/war, and machine-translated content farms on language-code subdomains.
BOT_WIKI_LANGS = {"ceb", "war"}
BOT_WIKI_HOST = re.compile(r"wikipedia|wikiwand|wiki2\.|wikizero|dbpedia|wikimedia|wikidata"
                           r"|wikiplanet|wikiwon|wikipedie|gpedia|wiko\.wiki", re.I)
MT_FARM_HOST = re.compile(r"^(ceb|war|ilo|pag|pam|tsg|hil|bcl|fil|tl|tgl)\.", re.I)
WIKIMEDIA = ("wikipedia.org", "wikimedia.org", "wiktionary.org", "wikibooks.org", "wikisource.org")


def _host(url: str) -> str:
    try:
        return url.split("/")[2].lower()
    except IndexError:
        return "?"


def fw2_excluded(url: str, lang: str) -> bool:
    h = _host(url)
    if lang in BOT_WIKI_LANGS and BOT_WIKI_HOST.search(h):
        return True
    return bool(MT_FARM_HOST.match(h)) and not h.endswith(WIKIMEDIA)


class FineWeb2Backend:
    name = "fineweb2"

    def __init__(self, max_docs: int = 2000, token: str | None = None):
        self.max_docs = max_docs
        self.token = token or os.environ.get("HF_TOKEN")
        self._done: set[str] = set()

    COLS = ("text", "url", "dump", "date", "language_score")

    def _stream(self, lang: str) -> Iterator[dict]:
        cfg = FW2_CONFIG[lang]
        try:
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceFW/fineweb-2", name=cfg, split="train",
                              streaming=True, token=self.token)
            it = iter(ds)
            first = next(it)
        except StopIteration:
            return
        except Exception as exc:
            # Some configs (pam_Latn) mix shard schemas and the datasets library refuses
            # to cast them. Read the parquet files directly, columns we need.
            print(f"  fineweb2 {cfg}: streaming failed ({type(exc).__name__}); reading shards directly")
            yield from self._read_shards(cfg)
            return
        yield first
        yield from it

    def _read_shards(self, cfg: str) -> Iterator[dict]:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download, list_repo_tree
        files = sorted(f.path for f in list_repo_tree(
            "HuggingFaceFW/fineweb-2", path_in_repo=f"data/{cfg}/train",
            repo_type="dataset", token=self.token) if f.path.endswith(".parquet"))
        for f in files:
            local = hf_hub_download("HuggingFaceFW/fineweb-2", f, repo_type="dataset", token=self.token)
            pf = pq.ParquetFile(local)
            cols = [c for c in self.COLS if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(columns=cols, batch_size=1000):
                yield from batch.to_pylist()

    def search(self, query: str, lang: str, max_results: int = 10) -> list[Hit]:
        # one pass per language regardless of how many queries the seed built
        if lang in self._done:
            return []
        self._done.add(lang)
        hits, n = [], 0
        for r in self._stream(lang):
            url = r.get("url") or ""
            if fw2_excluded(url, lang):
                continue
            hits.append(Hit(url=url, raw_text=r.get("text"), backend=self.name,
                            query=f"(fineweb2:{FW2_CONFIG[lang]})",
                            extra={"dump": r.get("dump"), "date": r.get("date"),
                                   "language_score": r.get("language_score")}))
            n += 1
            if n >= self.max_docs:
                break
        return hits


BACKENDS = {"tavily": TavilyBackend, "direct": DirectBackend, "fineweb2": FineWeb2Backend}


def get_backend(name: str = DEFAULT_BACKEND, **kwargs):
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
