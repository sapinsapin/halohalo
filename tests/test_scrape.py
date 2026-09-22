"""
Offline tests for the scrape package: the Tavily adapter against a stub client
(so the default backend is verified without a key or network), the direct
backend, FineWeb-2 exclusion rules, and dedup.

  venv/bin/python3 -m pytest tests/test_scrape.py -q
"""

import pytest

from halolib.scrape.dedup import DedupIndex, content_hash
from halolib.scrape.search import (DirectBackend, MissingCredential, TavilyBackend,
                                   fw2_excluded, get_backend)


class StubTavily:
    """Mimics tavily.TavilyClient.search / extract response shapes."""

    def __init__(self):
        self.calls = []

    def search(self, **kw):
        self.calls.append(kw)
        return {"results": [
            {"url": "https://example.ph/a", "title": "A", "content": "snippet a",
             "raw_content": "Maayong buntag sa tanan. " * 20, "score": 0.9},
            {"url": "https://example.ph/b", "title": "B", "content": "snippet b",
             "raw_content": None, "score": 0.5},
            {"title": "no url"},
        ]}

    def extract(self, urls):
        return {"results": [{"url": u, "raw_content": f"body of {u}"} for u in urls],
                "failed_results": []}


def test_tavily_requires_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(MissingCredential):
        TavilyBackend()


def test_tavily_search_maps_results():
    stub = StubTavily()
    b = TavilyBackend(client=stub)
    hits = b.search("kag sang", "hil", max_results=5)
    assert [h.url for h in hits] == ["https://example.ph/a", "https://example.ph/b"]
    assert hits[0].raw_text and hits[1].raw_text is None
    assert hits[0].backend == "tavily" and hits[0].query == "kag sang"
    kw = stub.calls[0]
    assert kw["include_raw_content"] is True and kw["max_results"] == 5
    assert kw["search_depth"] == "advanced"


def test_tavily_extract_fills_missing():
    b = TavilyBackend(client=StubTavily())
    got = b.extract(["https://example.ph/b"])
    assert got == {"https://example.ph/b": "body of https://example.ph/b"}


def test_default_backend_is_tavily():
    from halolib.scrape import DEFAULT_BACKEND
    assert DEFAULT_BACKEND == "tavily"


def test_direct_backend_serves_once(tmp_path):
    f = tmp_path / "urls.txt"
    f.write_text("# comment\nhttps://x.ph/1\n\nhttps://x.ph/2\n")
    b = get_backend("direct", urls_file=str(f))
    assert isinstance(b, DirectBackend)
    assert [h.url for h in b.search("q", "ceb", 10)] == ["https://x.ph/1", "https://x.ph/2"]
    assert b.search("q", "ceb", 10) == []
    assert len(b.search("q", "hil", 10)) == 2      # once per language


def test_fw2_exclusions():
    assert fw2_excluded("https://ceb.wikipedia.org/wiki/Foo", "ceb")        # Lsjbot wiki
    assert not fw2_excluded("https://ceb.wikipedia.org/wiki/Foo", "ilo")    # only ceb/war
    assert fw2_excluded("https://ceb.martech.zone/post", "ceb")             # MT farm
    assert not fw2_excluded("https://www.sunstar.com.ph/cebu/x", "ceb")


def test_dedup_exact_and_near():
    d = DedupIndex()
    words = ("ang balita karon adlawa mahitungod sa siyudad sa sugbo ug sa mga tawo nga "
             "nagpuyo didto nag-atubang og bag-ong hagit human sa bagyo nga miigo sa isla "
             "kagahapon samtang ang kagamhanan nagsaad og tabang alang sa mga biktima ug "
             "nag-awhag sa publiko nga magbantay sa dugang nga pag-ulan sa sunod nga adlaw "
             "ang mga eskwelahan gisirado ug ang mga klase gisuspenso sa tibuok probinsya").split()
    base = " ".join(words)
    assert d.check_and_add("u1", base) is None
    assert d.check_and_add("u2", base) == "exact"
    # syndicated copy: one word changed out of ~75 (two edits sits right at the
    # 0.75 LSH threshold on a text this short; real pages are far longer)
    near = words[:]
    near[40] = "gobyerno"
    assert d.check_and_add("u3", " ".join(near)) == "near"
    assert d.check_and_add("u4", "usa ka hingpit nga lahi nga teksto mahitungod sa "
                           "laing butang nga wala gayoy kalabotan sa una " * 4) is None
    assert content_hash("x") == content_hash("x")
