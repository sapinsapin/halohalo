"""
Seeded web scrape for the ten PLD languages — text, and voice discovery.

Seeds come from the corpora we hold (PLD, halohalo, BantayWika): distinctive
words per language become search queries, hits are fetched, gated by
language identification (HaloLID + GlotLID), cleaned, deduplicated, and
written as FineWeb-schema parquet shards with per-URL provenance. Resumable.

Backends (halolib/scrape/search.py): tavily (default), direct, fineweb2.

  python scrape_web.py                                  # all langs, Tavily
  python scrape_web.py --langs ceb,war --max-docs 200
  python scrape_web.py --backend direct --urls urls.txt --langs ceb
  python scrape_web.py --backend fineweb2 --max-docs 2000
  python scrape_web.py --voice --langs ilo --download-audio   # CC-only by default
  python scrape_web.py --push                           # append shards to sapinsapin/halo-{lang}

Output: $SCRAPE_DIR/{seeds,text/{lang},voice/{lang}} (default finetune_runs/scrape).
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from halolib.lid import LANGS  # noqa: E402
from halolib.scrape import DEFAULT_BACKEND, BACKENDS, MissingCredential  # noqa: E402

DEFAULT_OUT = Path(os.environ.get("SCRAPE_DIR",
                                  Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs")) / "scrape"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--langs", default="all", help="comma list of PLD codes, or all")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default=DEFAULT_BACKEND)
    ap.add_argument("--urls", help="URL list file for --backend direct")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--queries-per-lang", type=int, default=20)
    ap.add_argument("--max-results", type=int, default=10, help="hits per query")
    ap.add_argument("--max-docs", type=int, default=500, help="accepted docs per language")
    ap.add_argument("--min-words", type=int, default=30)
    ap.add_argument("--lid-min-score", type=float, default=0.6)
    ap.add_argument("--lid-min-agreement", type=float, default=0.6)
    ap.add_argument("--no-glotlid", action="store_true", help="HaloLID only (faster, no 1.7 GB model)")
    ap.add_argument("--max-per-host", type=int, default=None,
                    help="cap accepted documents per host per language (off by default; "
                         "the summary always reports the top hosts)")
    ap.add_argument("--refresh-seeds", action="store_true")
    ap.add_argument("--no-text", action="store_true", help="skip the text scrape")
    ap.add_argument("--voice", action="store_true", help="YouTube discovery with the same seeds")
    ap.add_argument("--download-audio", action="store_true")
    ap.add_argument("--all-licenses", action="store_true",
                    help="download non-CC audio too (default: Creative Commons only)")
    ap.add_argument("--max-downloads", type=int, default=20)
    ap.add_argument("--push", action="store_true", help="append text shards to the Hub")
    ap.add_argument("--repo-prefix", default="sapinsapin/halo-")
    args = ap.parse_args()

    langs = tuple(LANGS) if args.langs == "all" else tuple(args.langs.split(","))
    bad = [l for l in langs if l not in LANGS]
    if bad:
        sys.exit(f"unknown language code(s) {bad}; valid: {LANGS}")

    from halolib.lid import default_ensemble
    from halolib.scrape.pipeline import ScrapeConfig, push_lang, run_text
    from halolib.scrape.seeds import prepare_seeds

    backend_kwargs = {}
    if args.backend == "direct":
        backend_kwargs["urls_file"] = args.urls
    elif args.backend == "fineweb2":
        backend_kwargs["max_docs"] = args.max_docs * 3      # headroom for the LID gate

    cfg = ScrapeConfig(
        out_dir=args.out, langs=langs, backend=args.backend, backend_kwargs=backend_kwargs,
        queries_per_lang=args.queries_per_lang, max_results=args.max_results,
        max_docs_per_lang=args.max_docs, min_words=args.min_words,
        lid_min_score=args.lid_min_score, lid_min_agreement=args.lid_min_agreement,
        use_glotlid=not args.no_glotlid, refresh_seeds=args.refresh_seeds,
        max_per_host=args.max_per_host,
    )

    print("loading language ID ...")
    lid = default_ensemble(not args.no_glotlid)
    print(f"  models: {'halolid ' if lid.halo else ''}{'glotlid' if lid.glot else ''}".rstrip())

    if not args.no_text:
        try:
            run_text(cfg, lid)
        except MissingCredential as exc:
            sys.exit(f"\n{exc}")

    if args.voice:
        from halolib.scrape.voice import run_voice
        seeds = prepare_seeds(args.out / "seeds", langs=langs, n_queries=args.queries_per_lang,
                              token=os.environ.get("HF_TOKEN"))
        run_voice(args.out, langs, seeds, lid, do_download=args.download_audio,
                  cc_only=not args.all_licenses, max_downloads=args.max_downloads)

    if args.push:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
        for lang in langs:
            push_lang(args.out, lang, f"{args.repo_prefix}{lang}")


if __name__ == "__main__":
    main()
