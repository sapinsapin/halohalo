"""
The corpus flywheel: scrape -> accumulate -> improve LID -> iterate.

Each round uses what the previous rounds collected:

  seeds    keywords are re-mined from PLD + halohalo + every accepted scrape
           page so far, and the query sampler gets a new random seed, so a
           round searches for different combinations of the language's own
           words; queries already spent are never reused
  scrape   the seed queries run on the default backend (Tavily)
  expand   a second, cheaper pass restricted to the hosts that produced
           accepted pages in earlier rounds (Tavily include_domains, basic
           depth = 1 credit) — sites we already know publish in the language
  crawl    a free pass over the same kind of hosts: their sitemaps are read
           and a sample of unseen pages fetched directly (no search credits),
           so the loop keeps growing after the search allowance is spent
  lid      once enough new documents have accumulated (--retrain-every),
           HaloLID is retrained with the shards folded in and scored against
           the incumbent on the *same* held-out sentences in the same run; it
           replaces the incumbent only if it does not regress on the
           human-labelled PLD test set or the web test set (macro F1)
  log      one JSON line per round in $FINETUNE_DIR/flywheel/rounds.jsonl,
           one per LID attempt in lid_events.jsonl, per-host yield in
           host_stats.json

Hosts that stop paying are dropped from expansion and crawling: two passes
in a row with no new accepted page, or at least 20 pages tried at under 10 %
acceptance (multilingual sites whose pages are mostly another language).

Credits: every round asks Tavily's usage endpoint and scales its search
passes down to what is left within --credit-budget (a total for the whole
invocation; the plan's remaining allowance by default). A seed query costs
2 credits (advanced depth), an expansion query 1 (basic). The crawl is free.

  python scripts/flywheel.py --rounds 1
  python scripts/flywheel.py --rounds 3 --queries-per-lang 0 --expand-queries 0   # crawl only
  python scripts/flywheel.py --rounds 1 --langs tsg,war --no-lid
  python scripts/flywheel.py --lid-only                 # run the LID gate now
  python scripts/flywheel.py --lid-rebase 2026-09-23T18:05:13+00:00

--lid-rebase is for an incumbent trained before the fixed hash split
(train_lid.SPLIT): it cannot be scored fairly on today's test set because
some of those sentences were in its training data. The rebase retrains the
incumbent's recipe (scrape rows crawled before the cutoff) on today's split
as a control, then a candidate with everything, and promotes whichever wins.

Everything is resumable: URLs already examined are skipped by the manifest,
shards accumulate, and the LID promotion decision is recorded.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from halolib.lid import LANGS  # noqa: E402

FINETUNE = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs"))
SCRAPE = Path(os.environ.get("SCRAPE_DIR", FINETUNE / "scrape"))
LID_DIR = FINETUNE / "lid"
FLY = FINETUNE / "flywheel"
HOST_STATS = FLY / "host_stats.json"
LID_EVENTS = FLY / "lid_events.jsonl"
LID_FILES = ("model.ftz", "model.bin", "results.json")

# Hosts that must not drive expansion: bot encyclopedias, scripture mirrors
# that already dominate the small languages, and generic platforms.
NO_EXPAND = ("wikipedia.org", "wikisource.org", "wikimedia.org", "wiktionary.org",
             "jw.org", "jw-cdn.org", "bible.com", "bible.is", "ebible.org", "biblegateway.com",
             "bibles.org",   # 277 of Tausug's 387 pages; round 2 expanded into it and found 0 new
             "biblica.com", "biblia.chat", "speedbibleverse.com", "breakeveryyoke.com",
             "desiringgod.org", "gotquestions.org", "lds.org", "mormon.org",
             "churchofjesuschrist.org", "bibliamundi.com", "amazinggracebibleinstitute.com",
             "scribd.com", "pdfcoffee.com", "wattpad.com",
             "mymemory.translated.net", "translated.net",
             # English expansion is pointless (dictionary.com, goodreads...) and
             # English is not what the corpus is for
             "dictionary.com", "dictionary.cambridge.org", "goodreads.com", "rottentomatoes.com")
# Blog platforms: the platform itself spans every language, but one blog on
# it is one site — tausugblog.blogspot.com is exactly the kind of host the
# smallest languages need, and a suffix rule on blogspot.com threw it away.
PLATFORMS = ("blogspot.com", "wordpress.com", "tumblr.com", "weebly.com", "wixsite.com",
             "tripod.com", "webs.com")
# Scripture mirrors by name: the long tail (speedbibleverse, htmlbible,
# sacrednamebible...) is unbounded, and once single blogs became eligible the
# Bikol Book of Mormon blog (anlibronimormon.blogspot.com) was 23 of round 3's
# 33 crawled Bikol pages. Parish blogs are local news and stay eligible.
SCRIPTURE_FRAGMENTS = ("bible", "biblia", "biblica", "biblista", "mormon", "scriptur")
DRY_PASSES = 2
MIN_TRIED, MIN_RATE = 20, 0.10


def _norm_host(url_or_host: str) -> str:
    h = url_or_host.split("/")[2] if "//" in url_or_host else url_or_host
    h = h.lower().split(":")[0]
    return h[4:] if h.startswith("www.") else h


def _no_expand(host: str) -> bool:
    from halolib.scrape.pipeline import excluded_host
    return (host in PLATFORMS
            or any(f in host for f in SCRIPTURE_FRAGMENTS)
            or any(host == h or host.endswith("." + h) for h in NO_EXPAND)
            or excluded_host(f"https://{host}/"))


def _manifest(lang: str):
    mp = SCRAPE / "text" / lang / "manifest.jsonl"
    if not mp.exists():
        return
    for ln in mp.read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(ln)
        except json.JSONDecodeError:
            continue


# -- Tavily credits -----------------------------------------------------------

def tavily_usage() -> dict | None:
    import requests
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return None
    try:
        r = requests.get("https://api.tavily.com/usage",
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
        return r.json().get("account") if r.ok else None
    except requests.RequestException:
        return None


def plan_state() -> tuple[int | None, int | None]:
    """(credits used this period, credits left on the plan)."""
    u = tavily_usage()
    if not u or u.get("plan_usage") is None:
        return None, None
    used = int(u["plan_usage"])
    left = int(u["plan_limit"]) - used if u.get("plan_limit") is not None else None
    return used, left


# -- hosts --------------------------------------------------------------------

def load_host_stats() -> dict:
    return json.loads(HOST_STATS.read_text()) if HOST_STATS.exists() else {}


def exhausted(s: dict | None) -> bool:
    if not s:
        return False
    return s["dry"] >= DRY_PASSES or (s["tried"] >= MIN_TRIED and s["accepted"] / s["tried"] < MIN_RATE)


def accepted_hosts(lang: str, top_n: int, mode: str | None = None, stats: dict | None = None) -> list[str]:
    """Hosts that produced accepted pages, most productive first, minus the
    ones expansion must not amplify and (for a mode) the ones that stopped
    paying in that mode."""
    per = (stats or {}).get(mode, {}).get(lang, {}) if mode else {}
    c: Counter = Counter()
    for r in _manifest(lang):
        if r.get("status") == "accepted":
            host = _norm_host(r["url"])
            if not _no_expand(host) and not exhausted(per.get(host)):
                c[host] += 1
    return [h for h, _ in c.most_common(top_n)]


def update_host_stats(stats: dict, mode: str, lang: str, hosts: list[str], since_iso: str) -> None:
    """Credit each targeted host with what its pass produced (a record on a
    subdomain counts for the targeted parent: tacloban.bomboradyo.com)."""
    tried, acc = Counter(), Counter()
    for r in _manifest(lang):
        if r.get("ts", "") < since_iso:
            continue
        h = _norm_host(r["url"])
        target = next((t for t in hosts if h == t or h.endswith("." + t)), None)
        if target:
            tried[target] += 1
            acc[target] += r.get("status") == "accepted"
    per = stats.setdefault(mode, {}).setdefault(lang, {})
    for h in hosts:
        s = per.setdefault(h, {"passes": 0, "tried": 0, "accepted": 0, "dry": 0})
        s["passes"] += 1
        s["tried"] += tried[h]
        s["accepted"] += acc[h]
        s["dry"] = 0 if acc[h] else s["dry"] + 1
    FLY.mkdir(parents=True, exist_ok=True)
    HOST_STATS.write_text(json.dumps(stats, indent=1))


# -- accounting ---------------------------------------------------------------

def tally(langs) -> dict[str, dict]:
    import glob
    import pyarrow.parquet as pq
    out = {}
    for lang in langs:
        d = w = 0
        for f in glob.glob(str(SCRAPE / "text" / lang / "shard-*.parquet")):
            col = pq.read_table(f, columns=["word_count"]).column("word_count").to_pylist()
            d += len(col)
            w += sum(col)
        out[lang] = {"docs": d, "words": w}
    return out


def gained_since(langs, t0_iso: str) -> dict[str, dict]:
    """Documents and words accepted per language since an ISO timestamp,
    from the manifests. Unlike a shard tally this is unaffected by purges
    that run mid-round (the first round's tally showed Cebuano at -412
    documents because an MT-farm purge happened while it ran)."""
    out = {}
    for lang in langs:
        d = w = 0
        for r in _manifest(lang):
            if r.get("status") == "accepted" and r.get("ts", "") >= t0_iso:
                d += 1
                w += r.get("words", 0)
        out[lang] = {"docs": d, "words": w}
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -- LID gate -----------------------------------------------------------------

def _f1(results: dict, model: str, subset: str) -> float | None:
    for row in results.get("results", []):
        if row.get("model") == model and row.get("subset") == subset:
            return row.get("macro_f1")
    return None


def _train(out: Path, baseline: Path | None = None, extra_before: str | None = None,
           tag: str = "") -> tuple[int, Path]:
    cmd = [sys.executable, str(ROOT / "scripts" / "train_lid.py"),
           "--extra-parquet", str(SCRAPE / "text"), "--out", str(out)]
    if baseline:
        cmd += ["--baseline", str(baseline)]
    if extra_before:
        cmd += ["--extra-before", extra_before]
    # Log to a file, not a pipe: the run is long, its output is worth keeping,
    # and a parent that dies mid-way must not take the trainer's output with it.
    FLY.mkdir(parents=True, exist_ok=True)
    log_path = FLY / f"train_lid_{datetime.now(timezone.utc):%Y%m%dT%H%M}{tag}.log"
    print(f"  training -> {out}  (log {log_path})")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, log_path


def _copy_model(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in LID_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)


def _log_lid_event(ev: dict) -> dict:
    ev = {"ts": _now(), **ev}
    FLY.mkdir(parents=True, exist_ok=True)
    with open(LID_EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")
    return ev


def last_lid_ts() -> str:
    """When the gate last ran — lid_events.jsonl, else the last round that
    retrained, else the beginning of time."""
    if LID_EVENTS.exists():
        lines = LID_EVENTS.read_text(encoding="utf-8").splitlines()
        if lines:
            return json.loads(lines[-1])["ts"]
    ts = ""
    if (FLY / "rounds.jsonl").exists():
        for ln in open(FLY / "rounds.jsonl", encoding="utf-8"):
            r = json.loads(ln)
            if "after_f1" in r.get("lid", {}):
                ts = r["ts"]
    return ts


def best_pld_f1() -> float | None:
    """Best PLD macro F1 any promoted model has had on the hash split. The
    PLD test set is fixed, so these are comparable across runs; bounding a
    candidate by the best as well as by the incumbent stops a run of
    within-tolerance promotions from drifting downhill (round 4's retrain was
    promoted at -0.0012)."""
    best = None
    if LID_EVENTS.exists():
        for ln in LID_EVENTS.read_text(encoding="utf-8").splitlines():
            ev = json.loads(ln)
            pld = ev.get("gate", {}).get("pld", {})
            scores = [pld.get("baseline")] + ([pld.get("candidate")] if ev.get("promoted") else [])
            for v in scores:
                if v is not None:
                    best = v if best is None else max(best, v)
    return best


def gate(candidate_results: Path, min_delta: float, best_pld: float | None = None) -> tuple[bool, dict]:
    """Candidate vs the baseline scored in the same run on the same sentences,
    and on PLD also vs the best score any promoted model has had."""
    r = json.loads(candidate_results.read_text()) if candidate_results.exists() else {}
    deltas, ok = {}, bool(r)
    for subset in ("pld", "web"):
        new, old = _f1(r, "halolid", subset), _f1(r, "baseline", subset)
        deltas[subset] = {"baseline": old, "candidate": new,
                          "delta": round(new - old, 4) if None not in (new, old) else None}
        if new is None:
            ok = False
        elif old is not None and new - old < min_delta:
            ok = False
    new = _f1(r, "halolid", "pld")
    if best_pld is not None and new is not None:
        deltas["pld"]["best"] = best_pld
        if new - best_pld < min_delta:
            ok = False
    return ok, deltas


def retrain_lid(min_delta: float = -0.002) -> dict:
    """Retrain with the accumulated shards; promote only if neither the PLD
    nor the web macro F1 drops by more than min_delta against the incumbent
    scored on the same test set."""
    from scripts.train_lid import SPLIT
    inc = LID_DIR / "results.json"
    split = json.loads(inc.read_text()).get("split") if inc.exists() else None
    if split != SPLIT:
        print(f"LID: incumbent split is {split!r}, not {SPLIT!r}; run --lid-rebase first")
        return _log_lid_event({"status": "needs_rebase", "incumbent_split": split})

    backup = FLY / "lid_backup"
    _copy_model(LID_DIR, backup)
    t0 = time.time()
    rc, log_path = _train(LID_DIR, baseline=backup / "model.ftz")
    promoted, deltas = gate(inc, min_delta, best_pld_f1()) if rc == 0 else (False, {})
    if not promoted:
        _copy_model(backup, LID_DIR)
    print(f"LID: {deltas}  {'PROMOTED' if promoted else 'kept incumbent'}")
    return _log_lid_event({"status": "promoted" if promoted else "kept", "promoted": promoted,
                           "gate": deltas, "min_delta": min_delta, "returncode": rc,
                           "seconds": round(time.time() - t0), "log": str(log_path)})


def rebase_lid(cutoff: str, min_delta: float = -0.002) -> dict:
    """Move the incumbent onto the hash split. Control = the incumbent's data
    recipe (scrape rows crawled before `cutoff`) retrained on today's split;
    candidate = everything, scored against the control in the same run."""
    backup = FLY / "lid_backup"
    _copy_model(LID_DIR, backup)
    control = FLY / "lid_control"
    t0 = time.time()
    rc, clog = _train(control, extra_before=cutoff, tag="_control")
    if rc != 0:
        return _log_lid_event({"status": "rebase_failed", "stage": "control", "log": str(clog)})
    rc, log_path = _train(LID_DIR, baseline=control / "model.ftz", tag="_candidate")
    promoted, deltas = gate(LID_DIR / "results.json", min_delta) if rc == 0 else (False, {})
    if not promoted:
        # the control is the incumbent's recipe on the new split: it becomes
        # the incumbent either way, so the next gate is clean
        _copy_model(control, LID_DIR)
    winner = "candidate" if promoted else "control"
    print(f"LID rebase: {deltas}  -> {winner}")
    return _log_lid_event({"status": "rebased", "winner": winner, "promoted": promoted,
                           "cutoff": cutoff, "gate": deltas, "min_delta": min_delta,
                           "seconds": round(time.time() - t0),
                           "log": str(log_path), "control_log": str(clog)})


# -- a round ------------------------------------------------------------------

def _slim(summary):
    keep = ("hits", "skipped_seen", "host_excluded", "fetch_fail", "lid_reject", "dup", "accepted")
    return {l: {k: v for k, v in s.items() if k in keep} for l, s in summary.items()}


def run_round(rnd: int, args, langs, allowance: int | None) -> dict:
    from halolib.lid import default_ensemble
    from halolib.scrape.pipeline import ScrapeConfig, run_text
    from halolib.scrape.seeds import prepare_seeds

    from halolib.scrape.pipeline import quarantine_corrupt_shards

    print(f"\n================ round {rnd} ================")
    quarantine_corrupt_shards(SCRAPE / "text")
    # A round interrupted by a WSL panic resumes where it stopped: finished
    # (pass, language) pairs are skipped and the seed queries already drawn are
    # reused. Round 11 otherwise paid for four fresh seed passes (1,312 credits)
    # without completing once.
    prog_path = FLY / f"round_{rnd:03d}.progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else None
    if prog:
        print(f"resuming round {rnd} from {prog['round_start']}: done {prog['done']}")
        seed_q, exp_q, used_before = prog["seed_q"], prog["exp_q"], prog["used_before"]
    else:
        used_before, left = plan_state()
        if allowance is not None:
            left = allowance if left is None else min(left, allowance)
        seed_q, exp_q = args.queries_per_lang, args.expand_queries
        if not os.environ.get("TAVILY_API_KEY"):
            seed_q = exp_q = 0
        need = len(langs) * (seed_q * 2 + exp_q)
        print(f"credits left: {left}  | search passes need ~{need}")
        if left is not None and need > left:
            # scale down rather than refuse outright
            scale = max(0.0, left / need)
            seed_q, exp_q = int(seed_q * scale), int(exp_q * scale)
            print(f"  scaled to {seed_q} seed / {exp_q} expand queries per language")
    crawl = args.crawl_per_host > 0 and args.crawl_hosts > 0
    if not (seed_q or exp_q or crawl):
        return {"round": rnd, "skipped": "no credits and crawl disabled"}

    lid = default_ensemble(True)
    stats = load_host_stats()
    if not prog:
        prog = {"round_start": _now(), "seed_q": seed_q, "exp_q": exp_q, "used_before": used_before,
                "seeds_ready": False, "done": {"seed": [], "expand": [], "crawl": []},
                "summary": {"seed": {}, "expand": {}, "crawl": {}},
                "expand_hosts": {}, "crawl_hosts": {}}
    FLY.mkdir(parents=True, exist_ok=True)

    def save():
        prog_path.write_text(json.dumps(prog, ensure_ascii=False, indent=1))

    round_start = prog["round_start"]
    if (seed_q or exp_q) and not prog["seeds_ready"]:
        prepare_seeds(SCRAPE / "seeds", langs=langs, n_queries=max(seed_q, exp_q),
                      refresh=True, token=os.environ.get("HF_TOKEN"),
                      query_seed=1000 + rnd, include_scrape=True,
                      used_queries_path=SCRAPE / "seeds" / "used_queries.json")
        prog["seeds_ready"] = True
        save()

    def todo(pass_name):
        return [l for l in langs if l not in prog["done"][pass_name]]

    def finish(pass_name, lang, summary):
        prog["summary"][pass_name].update(summary)
        prog["done"][pass_name].append(lang)
        save()

    # pass 1: fresh seed queries, advanced depth
    if seed_q:
        for lang in todo("seed"):
            cfg = ScrapeConfig(out_dir=SCRAPE, langs=(lang,), backend="tavily",
                               queries_per_lang=seed_q, max_results=args.max_results,
                               max_docs_per_lang=args.max_docs)
            finish("seed", lang, run_text(cfg, lid))

    # pass 2: expansion inside known-good hosts, basic depth, per language
    if exp_q:
        for lang in todo("expand"):
            hosts = accepted_hosts(lang, args.expand_hosts, "tavily", stats)
            if hosts:
                prog["expand_hosts"][lang] = hosts
                print(f"[{lang}] expand within {hosts}")
                t0 = _now()
                cfg = ScrapeConfig(out_dir=SCRAPE, langs=(lang,), backend="tavily",
                                   backend_kwargs={"include_domains": hosts, "search_depth": "basic"},
                                   queries_per_lang=exp_q, max_results=args.max_results,
                                   max_docs_per_lang=args.max_docs)
                summary = run_text(cfg, lid)
                update_host_stats(stats, "tavily", lang, hosts, t0)
            else:
                summary = {}
            finish("expand", lang, summary)

    # pass 3: crawl known-good hosts' sitemaps, free
    if crawl:
        # every URL any language has examined: a page is fetched once
        seen = {r["url"] for l in LANGS for r in _manifest(l)}
        for lang in todo("crawl"):
            hosts = accepted_hosts(lang, args.crawl_hosts, "sitemap", stats)
            summary = {}
            if hosts:
                prog["crawl_hosts"][lang] = hosts
                t0 = _now()
                cfg = ScrapeConfig(out_dir=SCRAPE, langs=(lang,), backend="sitemap",
                                   backend_kwargs={"hosts_by_lang": {lang: hosts},
                                                   "per_host": args.crawl_per_host,
                                                   "exclude": seen, "seed": rnd},
                                   queries_per_lang=1, max_docs_per_lang=args.max_docs)
                summary = run_text(cfg, lid)
                update_host_stats(stats, "sitemap", lang, hosts, t0)
            finish("crawl", lang, summary)

    summary1, summary2, summary3 = (prog["summary"][p] for p in ("seed", "expand", "crawl"))
    expand_hosts, crawl_hosts = prog["expand_hosts"], prog["crawl_hosts"]
    gained = gained_since(langs, round_start)
    print("gained this round: " + "  ".join(f"{l}={g['docs']}d/{g['words']:,}w" for l, g in gained.items()))
    used_after, _ = plan_state()

    rec = {"round": rnd, "ts": _now(), "round_start": round_start,
           "queries_per_lang": seed_q, "expand_queries": exp_q,
           "crawl_hosts_per_lang": args.crawl_hosts if crawl else 0,
           "crawl_per_host": args.crawl_per_host if crawl else 0,
           "credits_used_before": used_before, "credits_used_after": used_after,
           "credits_spent": (used_after - used_before) if None not in (used_before, used_after) else None,
           "gained": gained, "totals": tally(langs), "lid": {"status": "pending"},
           "seed_pass": _slim(summary1), "expand_pass": _slim(summary2), "crawl_pass": _slim(summary3),
           "expand_hosts": expand_hosts, "crawl_hosts": crawl_hosts}
    FLY.mkdir(parents=True, exist_ok=True)
    # Write the scrape accounting now: the LID retrain takes ten minutes and a
    # parent killed during it must not lose the round (round 1 did).
    rec_path = FLY / f"round_{rnd:03d}.json"
    rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))

    since = last_lid_ts()
    pending = sum(g["docs"] for g in gained_since(LANGS, since).values())
    if args.no_lid:
        lid_info = {"status": "skipped"}
    elif pending < args.retrain_every:
        lid_info = {"status": "deferred", "docs_since_last_gate": pending,
                    "retrain_every": args.retrain_every}
        print(f"LID: {pending} new docs since the last gate (< {args.retrain_every}); not retraining")
    else:
        lid_info = retrain_lid()
    rec["lid"] = lid_info
    rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(FLY / "rounds.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    prog_path.unlink(missing_ok=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--start-round", type=int, default=None, help="default: continue from the log")
    ap.add_argument("--langs", default="all")
    ap.add_argument("--queries-per-lang", type=int, default=10)
    ap.add_argument("--expand-queries", type=int, default=4)
    ap.add_argument("--expand-hosts", type=int, default=6)
    ap.add_argument("--crawl-hosts", type=int, default=8, help="hosts per language to crawl (0 = off)")
    ap.add_argument("--crawl-per-host", type=int, default=30, help="unseen pages per host per round")
    ap.add_argument("--max-results", type=int, default=10)
    ap.add_argument("--max-docs", type=int, default=300, help="accepted docs per language per pass")
    ap.add_argument("--credit-budget", type=int, default=None,
                    help="max Tavily credits to spend in total (default: what the plan has left)")
    ap.add_argument("--retrain-every", type=int, default=1000,
                    help="retrain LID once this many new documents have accumulated since the last gate")
    ap.add_argument("--no-lid", action="store_true", help="skip the LID step entirely")
    ap.add_argument("--lid-only", action="store_true", help="run the LID gate now and exit")
    ap.add_argument("--lid-rebase", metavar="CUTOFF_ISO",
                    help="move the incumbent onto the hash split (see the module docstring)")
    args = ap.parse_args()

    if args.lid_rebase:
        print(json.dumps(rebase_lid(args.lid_rebase), indent=1))
        return
    if args.lid_only:
        print(json.dumps(retrain_lid(), indent=1))
        return

    langs = tuple(LANGS) if args.langs == "all" else tuple(args.langs.split(","))
    start = args.start_round
    if start is None:
        start = 1
        if (FLY / "rounds.jsonl").exists():
            start = 1 + sum(1 for _ in open(FLY / "rounds.jsonl", encoding="utf-8"))
    spent = 0
    for rnd in range(start, start + args.rounds):
        allowance = None if args.credit_budget is None else max(0, args.credit_budget - spent)
        rec = run_round(rnd, args, langs, allowance)
        if rec.get("skipped"):
            print(f"round {rnd} skipped: {rec['skipped']}")
            break
        spent += rec.get("credits_spent") or 0


if __name__ == "__main__":
    main()
