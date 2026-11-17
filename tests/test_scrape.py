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


def test_bot_wiki_rule_reaches_search_backends():
    """The ceb/war Lsjbot exclusion must not depend on which backend found the
    page: Tavily's first Waray run was 19 % war.wikipedia.org."""
    import inspect
    from halolib.scrape import pipeline
    src = inspect.getsource(pipeline.run_text)
    assert "fw2_excluded(hit.url, lang)" in src


def test_mt_farms_by_language_code_path():
    from halolib.scrape.pipeline import mt_path_excluded
    assert mt_path_excluded("https://www.alltechbuzz.net/ceb/bill-gate-bahin-sa-pagkadato")
    assert mt_path_excluded("https://www.qc-solar.com/pag/news/cable-free-solar-pv-junction-box")
    assert mt_path_excluded("https://tecnobits.com/ceb/windows-detecta-el-disco-duro")
    assert mt_path_excluded("https://www.pilotech.ai/pam/blogs/how-to-resolve")
    # human-translation publishers keep their language-code paths
    assert not mt_path_excluded("https://www.jw.org/pag/library/")
    assert not mt_path_excluded("https://ebible.org/tsg/")
    assert not mt_path_excluded("https://www.fema.gov/fil/disaster")
    # ordinary regional sites have no language code in the path
    assert not mt_path_excluded("https://roxas.bomboradyo.com/soldado-nga-nakatiro")
    assert not mt_path_excluded("https://www.sunstar.com.ph/cebu/balita")
    assert not mt_path_excluded("https://ceb.wikipedia.org/wiki/Sugbo")     # subdomain rule's job
    assert not mt_path_excluded("nonsense")


def test_adult_hosts_are_excluded():
    from halolib.scrape.pipeline import excluded_host
    assert excluded_host("https://filipinosexstories.com/story/123")
    assert excluded_host("https://www.pinoy-porn-hub.example/x")
    assert not excluded_host("https://www.essexnews.example/")            # 'sex' inside a word is not enough
    assert not excluded_host("https://www.pinoyparazzi.com/balita")
    assert excluded_host("https://simpsonssexgames.com/x")                # found in the shards
    assert excluded_host("https://sexyhotfilipinagirls.blogspot.com/")


def test_gambling_and_game_portal_spam_is_excluded():
    from halolib.scrape.pipeline import excluded_host
    assert excluded_host("https://todosabongga.blogspot.com/2024/01/x.html")
    assert excluded_host("https://world-casino-ceb.com/")
    assert excluded_host("https://ph.sgames.org/delta-force")             # MT game portal
    assert not excluded_host("https://sc.judiciary.gov.ph/decision")      # 'judi' is not a fragment
    assert not excluded_host("https://gamereactor.ph/review")             # games journalism is fine


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


class FakeFetcher:
    """fetch_html over a dict of url -> body; everything else is a 404."""

    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def fetch_html(self, url):
        self.calls.append(url)
        return (self.pages[url], 200) if url in self.pages else (None, 404)


def _urlset(*urls):
    return ("<?xml version='1.0'?><urlset>"
            + "".join(f"<url><loc>{u}</loc></url>" for u in urls) + "</urlset>")


def test_sitemap_backend_follows_robots_and_index():
    from halolib.scrape.search import SitemapBackend
    f = FakeFetcher({
        "https://blog.ph/robots.txt": "User-agent: *\nSitemap: https://blog.ph/sm-index.xml\n",
        "https://blog.ph/sm-index.xml": ("<sitemapindex><sitemap><loc>https://blog.ph/tag-sitemap.xml</loc>"
                                         "</sitemap><sitemap><loc>https://blog.ph/post-sitemap.xml</loc>"
                                         "</sitemap></sitemapindex>"),
        "https://blog.ph/post-sitemap.xml": _urlset(
            "https://blog.ph/2024/01/istorya", "https://blog.ph/2024/02/balita",
            "https://blog.ph/tag/news/", "https://blog.ph/wp-content/uploads/a.jpg",
            "https://other.ph/2024/x", "https://blog.ph/"),
        "https://blog.ph/tag-sitemap.xml": _urlset("https://blog.ph/tag/a/"),
    })
    b = SitemapBackend({"tsg": ["blog.ph"]}, per_host=10, fetcher=f,
                       exclude={"https://blog.ph/2024/02/balita"})
    hits = b.search("ignored", "tsg", 10)
    assert [h.url for h in hits] == ["https://blog.ph/2024/01/istorya"]   # unseen articles only
    assert hits[0].query == "(sitemap:blog.ph)" and hits[0].backend == "sitemap"
    assert b.search("again", "tsg", 10) == []                            # once per language
    assert f.calls.index("https://blog.ph/post-sitemap.xml") < f.calls.index("https://blog.ph/tag-sitemap.xml")


def test_sitemap_backend_falls_back_to_home_page_links():
    from halolib.scrape.search import SitemapBackend
    f = FakeFetcher({"https://site.ph/": '<a href="https://site.ph/a/b">x</a>'
                                         '<a href="https://elsewhere.ph/c">y</a>'})
    b = SitemapBackend({"war": ["site.ph"]}, fetcher=f)
    assert [h.url for h in b.search("", "war", 10)] == ["https://site.ph/a/b"]


def test_run_text_spends_only_its_query_budget(tmp_path, monkeypatch):
    """The flywheel's expansion pass reuses a seed cache with more queries
    than it may spend; round 2 ran six expansion queries instead of three."""
    import json
    from halolib.scrape import pipeline
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "hil.json").write_text(json.dumps({"queries": [f"q{i}" for i in range(6)]}))

    class CountingBackend:
        name = "count"

        def __init__(self):
            self.queries = []

        def search(self, query, lang, max_results):
            self.queries.append(query)
            return []

    backend = CountingBackend()
    monkeypatch.setattr(pipeline, "get_backend", lambda *a, **k: backend)
    cfg = pipeline.ScrapeConfig(out_dir=tmp_path, langs=("hil",), queries_per_lang=3)
    pipeline.run_text(cfg, lid=object())
    assert backend.queries == ["q0", "q1", "q2"]


def test_lid_split_is_a_function_of_the_text():
    """Adding web sentences must not move a single PLD sentence between train
    and test (it did, and the promotion gate compared different test sets)."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "train_lid", Path(__file__).resolve().parent.parent / "scripts" / "train_lid.py")
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)
    pld = [(l, f"{l} pangungusap {i}", "pld") for l in ("ceb", "tsg") for i in range(400)]
    web = [(l, f"{l} web {i}", "web") for l in ("ceb", "tsg") for i in range(300)]
    more_web = web + [("ceb", f"ceb bag-o {i}", "web") for i in range(777)]
    _, test_a, _ = tl.cap_and_split(pld + web, max_per_lang=20000)
    train_b, test_b, _ = tl.cap_and_split(pld + more_web, max_per_lang=20000)
    pld_a = {t for _, t, s in test_a if s == "pld"}
    assert pld_a == {t for _, t, s in test_b if s == "pld"}
    assert pld_a and not pld_a & {t for _, t, _ in train_b}


def test_flywheel_expansion_hosts(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "flywheel", Path(__file__).resolve().parent.parent / "scripts" / "flywheel.py")
    fw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fw)
    # one blog on a platform is a site; the platform and scripture mirrors are not
    assert not fw._no_expand("tausugblog.blogspot.com")
    assert fw._no_expand("blogspot.com") and fw._no_expand("bibles.org")
    assert fw._no_expand("pam.wikipedia.org")
    assert fw._no_expand("anlibronimormon.blogspot.com") and fw._no_expand("speedbibleverse.com")
    assert not fw._no_expand("sanfabianparish.blogspot.com")

    monkeypatch.setattr(fw, "SCRAPE", tmp_path)
    monkeypatch.setattr(fw, "FLY", tmp_path / "fly")
    monkeypatch.setattr(fw, "HOST_STATS", tmp_path / "fly" / "host_stats.json")
    d = tmp_path / "text" / "tsg"
    d.mkdir(parents=True)
    recs = ([{"url": f"https://bibles.org/{i}", "status": "accepted", "ts": "2026-01-01"} for i in range(9)]
            + [{"url": f"https://www.tausugblog.blogspot.com/{i}", "status": "accepted", "ts": "2026-01-01"}
               for i in range(3)]
            + [{"url": "https://sufihub.com/a", "status": "accepted", "ts": "2026-01-01"}])
    (d / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    assert fw.accepted_hosts("tsg", 5) == ["tausugblog.blogspot.com", "sufihub.com"]

    stats = {}
    for _ in range(fw.DRY_PASSES):            # passes that find nothing new retire a host
        fw.update_host_stats(stats, "tavily", "tsg", ["sufihub.com"], "2027-01-01")
    assert fw.accepted_hosts("tsg", 5, "tavily", stats) == ["tausugblog.blogspot.com"]
    assert fw.accepted_hosts("tsg", 5, "sitemap", stats) == ["tausugblog.blogspot.com", "sufihub.com"]


def test_lid_gate_is_bounded_by_the_best_model(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "flywheel", Path(__file__).resolve().parent.parent / "scripts" / "flywheel.py")
    fw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fw)
    res = tmp_path / "results.json"

    def write(pld_new, pld_old, web_new=0.99, web_old=0.99):
        res.write_text(json.dumps({"results": [
            {"model": "halolid", "subset": "pld", "macro_f1": pld_new},
            {"model": "baseline", "subset": "pld", "macro_f1": pld_old},
            {"model": "halolid", "subset": "web", "macro_f1": web_new},
            {"model": "baseline", "subset": "web", "macro_f1": web_old}]}))

    write(0.8734, 0.8746)
    assert fw.gate(res, -0.002)[0]                        # within tolerance of the incumbent
    write(0.8722, 0.8734)
    assert fw.gate(res, -0.002)[0]                        # ...and again: this is the drift
    assert not fw.gate(res, -0.002, best_pld=0.8746)[0]   # 0.0024 below the best: rejected
    write(0.8760, 0.8746, web_new=0.985)
    assert not fw.gate(res, -0.002)[0]                    # web regressed


def test_purge_excluded_rewrites_shards_and_manifest(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq
    spec = importlib.util.spec_from_file_location(
        "purge", Path(__file__).resolve().parent.parent / "scripts" / "purge_excluded.py")
    pu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pu)
    d = tmp_path / "fil"
    d.mkdir()
    urls = ["https://pep.ph/a", "https://filipinosexstories.com/b", "https://ph.sgames.org/c"]
    pq.write_table(pa.table({"url": urls, "word_count": [10, 20, 30]}), d / "shard-00000.parquet")
    pq.write_table(pa.table({"url": urls[1:2], "word_count": [5]}), d / "shard-00001.parquet")
    (d / "manifest.jsonl").write_text("\n".join(json.dumps({"url": u, "status": "accepted"}) for u in urls))

    hosts, words = pu.purge_lang(d, dry_run=False, manifests=False)
    assert sum(hosts.values()) == 3 and words == 55
    assert pq.read_table(d / "shard-00000.parquet").column("url").to_pylist() == ["https://pep.ph/a"]
    assert (d / "shard-00001.parquet").exists()                    # emptied, never deleted
    assert pq.read_table(d / "shard-00001.parquet").num_rows == 0
    assert all(json.loads(l)["status"] == "accepted" for l in (d / "manifest.jsonl").read_text().splitlines())

    pu.purge_lang(d, dry_run=False)                                  # later full run marks them
    st = [json.loads(l)["status"] for l in (d / "manifest.jsonl").read_text().splitlines()]
    assert st == ["accepted", "host_excluded", "host_excluded"]


def test_manifest_forgets_accepted_rows_that_were_never_saved(tmp_path):
    import json
    from halolib.scrape.pipeline import Manifest
    mp = tmp_path / "manifest.jsonl"
    mp.write_text("\n".join(json.dumps(r) for r in [
        {"url": "https://a.ph/1", "status": "accepted", "hash": "h1"},
        {"url": "https://a.ph/2", "status": "accepted", "hash": "h2"},     # died before flush
        {"url": "https://a.ph/3", "status": "lid_reject"},
        {"url": "https://a.ph/4", "status": "accepted", "hash": "h4"},
        {"url": "https://a.ph/4", "status": "host_excluded"},               # purged later
    ]))
    m = Manifest(mp)
    assert m.forget_unsaved({"h1"}) == 1
    assert "https://a.ph/2" not in m.seen                                   # will be fetched again
    assert {"https://a.ph/1", "https://a.ph/3", "https://a.ph/4"} <= m.seen


def test_corrupt_shards_are_quarantined_and_their_slot_kept(tmp_path):
    import pyarrow.parquet as pq
    from halolib.scrape.pipeline import (quarantine_corrupt_shards, shard_schema,
                                         write_shard_durably)
    d = tmp_path / "ceb"
    d.mkdir()
    write_shard_durably(shard_schema().empty_table(), d / "shard-00000.parquet")
    (d / "shard-00001.parquet").write_bytes(b"PAR1" + b"\x00garbage" * 50 + b"PAR1")
    bad = quarantine_corrupt_shards(tmp_path)
    assert [p.name for p in bad] == ["shard-00001.parquet"]
    assert pq.read_table(d / "shard-00001.parquet").num_rows == 0      # slot still filled
    assert len(list((d / "quarantine").iterdir())) == 1
    assert not list(d.glob("*.tmp"))
    assert quarantine_corrupt_shards(tmp_path) == []                   # idempotent


def test_interrupted_round_resumes_without_redoing_work(tmp_path, monkeypatch):
    import argparse
    import importlib.util
    from pathlib import Path
    import halolib.lid
    import halolib.scrape.pipeline as pipeline
    import halolib.scrape.seeds as seeds
    spec = importlib.util.spec_from_file_location(
        "flywheel", Path(__file__).resolve().parent.parent / "scripts" / "flywheel.py")
    fw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fw)
    monkeypatch.setattr(fw, "SCRAPE", tmp_path / "scrape")
    monkeypatch.setattr(fw, "FLY", tmp_path / "fly")
    monkeypatch.setattr(fw, "HOST_STATS", tmp_path / "fly" / "host_stats.json")
    monkeypatch.setattr(fw, "LID_EVENTS", tmp_path / "fly" / "lid_events.jsonl")
    (tmp_path / "scrape" / "text").mkdir(parents=True)
    monkeypatch.setattr(fw, "plan_state", lambda: (100, 10000))
    monkeypatch.setattr(halolib.lid, "default_ensemble", lambda *a: None)
    seeded = []
    monkeypatch.setattr(seeds, "prepare_seeds", lambda *a, **k: seeded.append(1))
    calls, crash = [], {"at": ("seed", "ceb")}

    def fake_run_text(cfg, lid):
        key = ("seed" if cfg.backend == "tavily" and not cfg.backend_kwargs else cfg.backend, cfg.langs[0])
        if key == crash["at"]:
            raise SystemExit("WSL panic")
        calls.append(key)
        return {cfg.langs[0]: {"hits": 1, "accepted": 1}}

    monkeypatch.setattr(pipeline, "run_text", fake_run_text)
    args = argparse.Namespace(queries_per_lang=2, expand_queries=0, expand_hosts=5, crawl_hosts=0,
                              crawl_per_host=0, max_results=10, max_docs=10, no_lid=True,
                              retrain_every=1000)
    langs = ("bcl", "ceb", "fil")
    try:
        fw.run_round(11, args, langs, None)
    except SystemExit:
        pass
    assert calls == [("seed", "bcl")] and len(seeded) == 1
    assert (tmp_path / "fly" / "round_011.progress.json").exists()

    crash["at"] = None
    rec = fw.run_round(11, args, langs, None)
    assert calls == [("seed", "bcl"), ("seed", "ceb"), ("seed", "fil")]   # bcl not redone
    assert len(seeded) == 1                                               # queries not redrawn
    assert set(rec["seed_pass"]) == set(langs)
    assert not (tmp_path / "fly" / "round_011.progress.json").exists()


def test_latin_share_decides_script():
    from halolib.lid import latin_share
    assert latin_share("Maayong buntag sa inyong tanan") == 1.0
    assert latin_share("Ñino, ika'y mabuhay — Việt Nam") == 1.0      # accents are Latin
    assert latin_share("สวัสดีครับ ยินดีต้อนรับ") == 0.0
    assert latin_share("שלום עולם hello") < 0.5
    assert latin_share("12345 !!!") == 1.0                              # no letters: let the model decide
