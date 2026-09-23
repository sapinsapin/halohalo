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
    assert "facebook.com" in kw["exclude_domains"]      # login-walled, no text


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


def test_nav_lines_are_boilerplate():
    from halolib.cleaner import clean_text, is_nav_line
    assert is_nav_line("* HOME * BOMBO TUGUEGARAO * BOMBO LAOAG * BOMBO VIGAN * BOMBO CAUAYAN")
    assert is_nav_line("Wednesday, September 23  * Top Stories * Statement  * Balita Hiligaynon * Digicast Negros")
    assert is_nav_line("Home | About Us | Contact | Privacy | Terms")
    # real Hiligaynon prose with asterisks or spacing must survive
    assert not is_nav_line("Presyo sang humay ang yara karon sa P18 pesos ang kilo, suno sa mga rice farmers.")
    assert not is_nav_line("Ginsiling ni Mayor nga ang programa * bag-o * kag * importante gid para sa siyudad.")
    out = clean_text("* HOME * NEWS * SPORTS * OPINION * CONTACT\nAng pondo para sa year-end "
                     "incentives nagalakip sang milyon para sa mga empleyado sang kapitolyo.")
    assert "HOME" not in out and "nagalakip" in out


def test_vertical_bullet_menu_is_dropped():
    from halolib.cleaner import clean_text
    # the philippinerevolution.nu dump: one menu item per line
    page = ("* HR/IHL\n* Languages\n* Subscribe\n* Contact\n\n* Home\n* About Us\n"
            "+ CPP Constitution and Program\n+ Great Achievements of the CPP\n"
            "Ginpahayag sang 62nd IBPA nga patay ang duha ka soldado sa engkwentro sa Capiz.\n"
            "* isa ka punto nga malawig kag detalyado nga ginhambal sang tagapamaba sa sini nga hitabo\n"
            "* ikaduha nga punto nga malawig man kag may yara sang dugang nga konteksto para sa mga mamumugon")
    out = clean_text(page)
    assert "Languages" not in out and "Constitution" not in out
    assert "Ginpahayag" in out
    # a short run of long bullet points is content, not a menu
    assert "malawig kag detalyado" in out


def test_lyrics_hosts_are_excluded_everywhere():
    from halolib.scrape.pipeline import excluded_host
    assert excluded_host("https://genius.com/Artist-song-lyrics")
    assert excluded_host("https://www.azlyrics.com/lyrics/x/y.html")      # subdomain
    assert not excluded_host("https://www.bomboradyo.com/balita")
    assert not excluded_host("https://notgenius.com/page")                  # suffix trap
    assert not excluded_host("nonsense")


def test_decode_body_survives_bogus_charset():
    from halolib.scrape.fetch import decode_body
    body = "Maayong buntag — ₱45 milyon".encode("utf-8")
    assert decode_body(body, "empty") == "Maayong buntag — ₱45 milyon"      # the crash case
    assert decode_body(body, None) == "Maayong buntag — ₱45 milyon"
    assert decode_body(body, "utf-8") == "Maayong buntag — ₱45 milyon"
    assert decode_body(b"\xff\xfe", "not-a-codec")                           # never raises


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
