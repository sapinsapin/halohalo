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
  lid      HaloLID is retrained with the accumulated shards folded in and
           kept only if it does not regress on the human-labelled PLD test
           set (macro F1); otherwise the previous model is restored, so the
           gate for the next round is never worse than this one's
  log      one JSON line per round in $FINETUNE_DIR/flywheel/rounds.jsonl

Credits: every round asks Tavily's usage endpoint first and refuses to start
a pass it cannot afford within --credit-budget (the plan's remaining
allowance by default). A seed pass costs 2 credits per query at advanced
depth, an expansion pass 1 per query at basic depth.

  python scripts/flywheel.py --rounds 1
  python scripts/flywheel.py --rounds 3 --queries-per-lang 10 --expand-queries 4
  python scripts/flywheel.py --rounds 1 --langs tsg,war --no-lid

Everything is resumable: URLs already examined are skipped by the manifest,
shards accumulate, and the LID promotion decision is recorded per round.
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

# Hosts that must not drive expansion: bot encyclopedias, scripture mirrors
# that already dominate the small languages, and generic platforms.
NO_EXPAND = ("wikipedia.org", "wikisource.org", "wikimedia.org", "wiktionary.org",
             "jw.org", "jw-cdn.org", "bible.com", "bible.is", "ebible.org", "biblegateway.com",
             "biblica.com", "biblia.chat", "speedbibleverse.com", "breakeveryyoke.com",
             "desiringgod.org", "gotquestions.org", "lds.org", "mormon.org",
             "churchofjesuschrist.org", "bibliamundi.com", "amazinggracebibleinstitute.com",
             "scribd.com", "pdfcoffee.com", "blogspot.com", "wordpress.com", "wattpad.com",
             "mymemory.translated.net", "translated.net",
             # English expansion is pointless (dictionary.com, goodreads...) and
             # English is not what the corpus is for
             "dictionary.com", "dictionary.cambridge.org", "goodreads.com", "rottentomatoes.com")


def _no_expand(host: str) -> bool:
    from halolib.scrape.pipeline import excluded_host
    return (any(host == h or host.endswith("." + h) for h in NO_EXPAND)
            or excluded_host(f"https://{host}/"))


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


def credits_left(budget_override: int | None) -> int | None:
    u = tavily_usage()
    if u is None or u.get("plan_limit") is None:
        return budget_override
    left = int(u["plan_limit"]) - int(u.get("plan_usage", 0))
    return min(left, budget_override) if budget_override is not None else left


def accepted_hosts(lang: str, top_n: int) -> list[str]:
    """Hosts that produced accepted pages, most productive first, minus the
    ones expansion must not amplify."""
    mp = SCRAPE / "text" / lang / "manifest.jsonl"
    if not mp.exists():
        return []
    c: Counter = Counter()
    for ln in mp.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "accepted":
            try:
                host = r["url"].split("/")[2].lower()
            except IndexError:
                continue
            if host.startswith("www."):
                host = host[4:]
            if not _no_expand(host):
                c[host] += 1
    return [h for h, _ in c.most_common(top_n)]


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


def lid_pld_f1(results_path: Path) -> float | None:
    if not results_path.exists():
        return None
    r = json.loads(results_path.read_text())
    for row in r.get("results", []):
        if row.get("model") == "halolid" and row.get("subset") == "pld":
            return row.get("macro_f1")
    return None


def retrain_lid(min_delta: float = -0.002) -> dict:
    """Retrain with the accumulated shards; promote only if PLD macro F1 does
    not regress by more than min_delta."""
    LID_DIR.mkdir(parents=True, exist_ok=True)
    before = lid_pld_f1(LID_DIR / "results.json")
    backup = FLY / "lid_backup"
    backup.mkdir(parents=True, exist_ok=True)
    for f in ("model.ftz", "model.bin", "results.json"):
        if (LID_DIR / f).exists():
            shutil.copy2(LID_DIR / f, backup / f)

    cmd = [sys.executable, str(ROOT / "scripts" / "train_lid.py"),
           "--extra-parquet", str(SCRAPE / "text")]
    t0 = time.time()
    # Log to a file, not a pipe: the run is long, its output is worth keeping,
    # and a parent that dies mid-way must not take the trainer's output with it.
    log_path = FLY / f"train_lid_{datetime.now(timezone.utc):%Y%m%dT%H%M}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    after = lid_pld_f1(LID_DIR / "results.json") if proc.returncode == 0 else None

    promoted = (after is not None) and (before is None or after - before >= min_delta)
    if not promoted:
        for f in ("model.ftz", "model.bin", "results.json"):
            if (backup / f).exists():
                shutil.copy2(backup / f, LID_DIR / f)
    return {"before_f1": before, "after_f1": after, "promoted": promoted,
            "seconds": round(time.time() - t0), "returncode": proc.returncode,
            "log": str(log_path)}


def gained_since(langs, t0_iso: str) -> dict[str, dict]:
    """Documents and words accepted per language since an ISO timestamp,
    from the manifests. Unlike a shard tally this is unaffected by purges
    that run mid-round (the first round's tally showed Cebuano at -412
    documents because an MT-farm purge happened while it ran)."""
    out = {}
    for lang in langs:
        d = w = 0
        mp = SCRAPE / "text" / lang / "manifest.jsonl"
        if mp.exists():
            for ln in mp.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "accepted" and r.get("ts", "") >= t0_iso:
                    d += 1
                    w += r.get("words", 0)
        out[lang] = {"docs": d, "words": w}
    return out


def run_round(rnd: int, args, langs) -> dict:
    from halolib.lid import default_ensemble
    from halolib.scrape.pipeline import ScrapeConfig, run_text
    from halolib.scrape.seeds import prepare_seeds

    print(f"\n================ round {rnd} ================")
    left = credits_left(args.credit_budget)
    seed_cost = len(langs) * args.queries_per_lang * 2
    expand_cost = len(langs) * args.expand_queries * 1
    print(f"credits left: {left}  | this round needs ~{seed_cost} (seed) + ~{expand_cost} (expand)")
    if left is not None and left < seed_cost + expand_cost:
        # scale down rather than refuse outright
        scale = max(0.0, left / max(seed_cost + expand_cost, 1))
        args.queries_per_lang = max(0, int(args.queries_per_lang * scale))
        args.expand_queries = max(0, int(args.expand_queries * scale))
        print(f"  scaled to {args.queries_per_lang} seed / {args.expand_queries} expand queries per language")
        if args.queries_per_lang == 0 and args.expand_queries == 0:
            return {"round": rnd, "skipped": "no credits"}

    lid = default_ensemble(True)
    round_start = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seeds = prepare_seeds(SCRAPE / "seeds", langs=langs, n_queries=args.queries_per_lang,
                          refresh=True, token=os.environ.get("HF_TOKEN"),
                          query_seed=1000 + rnd, include_scrape=True,
                          used_queries_path=SCRAPE / "seeds" / "used_queries.json")

    # pass 1: fresh seed queries, advanced depth
    summary1 = {}
    if args.queries_per_lang:
        cfg = ScrapeConfig(out_dir=SCRAPE, langs=langs, backend="tavily",
                           queries_per_lang=args.queries_per_lang, max_results=args.max_results,
                           max_docs_per_lang=args.max_docs)
        summary1 = run_text(cfg, lid)

    # pass 2: expansion inside known-good hosts, basic depth, per language
    summary2 = {}
    if args.expand_queries:
        for lang in langs:
            hosts = accepted_hosts(lang, args.expand_hosts)
            if not hosts:
                continue
            print(f"[{lang}] expand within {hosts}")
            cfg = ScrapeConfig(out_dir=SCRAPE, langs=(lang,), backend="tavily",
                               backend_kwargs={"include_domains": hosts, "search_depth": "basic"},
                               queries_per_lang=args.expand_queries, max_results=args.max_results,
                               max_docs_per_lang=args.max_docs)
            summary2.update(run_text(cfg, lid))

    gained = gained_since(langs, round_start)
    print("gained this round: " + "  ".join(f"{l}={g['docs']}d/{g['words']:,}w" for l, g in gained.items()))

    def _slim(summary):
        return {l: {k: v for k, v in s.items() if k in ("hits", "accepted", "lid_reject", "fetch_fail")}
                for l, s in summary.items()}

    rec = {"round": rnd, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "round_start": round_start,
           "queries_per_lang": args.queries_per_lang, "expand_queries": args.expand_queries,
           "credits_before": left, "credits_after": credits_left(None),
           "gained": gained, "totals": tally(langs), "lid": {"status": "pending"},
           "seed_pass": _slim(summary1), "expand_pass": _slim(summary2)}
    FLY.mkdir(parents=True, exist_ok=True)
    # Write the scrape accounting now: the LID retrain takes ten minutes and a
    # parent killed during it must not lose the round (round 1 did).
    rec_path = FLY / f"round_{rnd:03d}.json"
    rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))

    lid_info = {"status": "skipped"} if args.no_lid else retrain_lid()
    if "before_f1" in lid_info:
        print(f"LID: pld macro-F1 {lid_info['before_f1']} -> {lid_info['after_f1']}  "
              f"{'PROMOTED' if lid_info['promoted'] else 'kept previous'}")
    rec["lid"] = lid_info
    rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(FLY / "rounds.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--start-round", type=int, default=None, help="default: continue from the log")
    ap.add_argument("--langs", default="all")
    ap.add_argument("--queries-per-lang", type=int, default=10)
    ap.add_argument("--expand-queries", type=int, default=4)
    ap.add_argument("--expand-hosts", type=int, default=6)
    ap.add_argument("--max-results", type=int, default=10)
    ap.add_argument("--max-docs", type=int, default=300, help="accepted docs per language per pass")
    ap.add_argument("--credit-budget", type=int, default=None,
                    help="max Tavily credits to spend in total (default: what the plan has left)")
    ap.add_argument("--no-lid", action="store_true", help="skip the LID retrain step")
    args = ap.parse_args()

    langs = tuple(LANGS) if args.langs == "all" else tuple(args.langs.split(","))
    start = args.start_round
    if start is None:
        start = 1
        if (FLY / "rounds.jsonl").exists():
            start = 1 + sum(1 for _ in open(FLY / "rounds.jsonl", encoding="utf-8"))
    for rnd in range(start, start + args.rounds):
        rec = run_round(rnd, args, langs)
        if rec.get("skipped"):
            print(f"round {rnd} skipped: {rec['skipped']}")
            break


if __name__ == "__main__":
    main()
