"""
Apply today's host exclusions to documents already accepted.

The exclusion rules (halolib.scrape.pipeline: excluded_host, mt_path_excluded,
and fw2_excluded for the language) run at scrape time, so a rule added later
leaves everything it would have caught sitting in the shards. The adult-host
rule of 2026-09-23 was one: filipinosexstories.com, the site its own comment
names, still had 64 Filipino rows three days later. This removes such rows
and marks their URLs `host_excluded` in the manifest, so no later round
fetches them again and the flywheel no longer counts their hosts as
productive.

Shards are rewritten in place and never deleted, even when emptied: the
shard writer names new shards by counting existing files, so a missing file
would make it overwrite a later shard. The writer only ever creates new
shard files, so --shards-only is safe while the flywheel runs; the
manifests, which it holds open for appending, are rewritten only by a run
without that flag, once nothing is scraping.

  python scripts/purge_excluded.py --dry-run
  python scripts/purge_excluded.py --shards-only     # while the flywheel runs
  python scripts/purge_excluded.py                   # afterwards: marks the manifests
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from halolib.scrape.pipeline import excluded_host, mt_path_excluded  # noqa: E402
from halolib.scrape.search import fw2_excluded  # noqa: E402

FINETUNE = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs"))
SCRAPE = Path(os.environ.get("SCRAPE_DIR", FINETUNE / "scrape"))


def is_excluded(url: str, lang: str) -> bool:
    return excluded_host(url) or mt_path_excluded(url) or fw2_excluded(url, lang)


def _host(url: str) -> str:
    h = url.split("/")[2].lower() if url.count("/") >= 2 else "?"
    return h[4:] if h.startswith("www.") else h


def purge_lang(lang_dir: Path, dry_run: bool, manifests: bool = True) -> tuple[Counter, int]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    lang = lang_dir.name
    hosts, words, purged = Counter(), 0, set()
    for shard in sorted(lang_dir.glob("shard-*.parquet")):
        t = pq.read_table(shard)
        urls = t.column("url").to_pylist()
        drop = [is_excluded(u, lang) for u in urls]
        if not any(drop):
            continue
        wc = t.column("word_count").to_pylist()
        for u, d, w in zip(urls, drop, wc):
            if d:
                hosts[_host(u)] += 1
                words += w
                purged.add(u)
        if not dry_run:
            kept = t.filter(pc.invert(pa.array(drop)))
            tmp = shard.with_suffix(".tmp")
            pq.write_table(kept, tmp)
            os.replace(tmp, shard)

    mp = lang_dir / "manifest.jsonl"
    if manifests and not dry_run and mp.exists():
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out, changed = [], False
        for ln in mp.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                out.append(ln)
                continue
            # this run's purges, and any earlier --shards-only run's
            if r.get("status") == "accepted" and (r.get("url") in purged
                                                  or is_excluded(r.get("url", ""), lang)):
                changed = True
                r["status"], r["purged"] = "host_excluded", stamp
            out.append(json.dumps(r, ensure_ascii=False))
        if not changed:
            return hosts, words
        tmp = mp.with_suffix(".tmp")
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        os.replace(tmp, mp)
    return hosts, words


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shards-only", action="store_true",
                    help="leave the manifests alone: safe while the flywheel is appending to "
                         "them; a later run without it marks the URLs")
    args = ap.parse_args()
    total_docs = total_words = 0
    for lang_dir in sorted(p for p in (SCRAPE / "text").iterdir() if p.is_dir()):
        hosts, words = purge_lang(lang_dir, args.dry_run, manifests=not args.shards_only)
        n = sum(hosts.values())
        if n:
            total_docs += n
            total_words += words
            top = ", ".join(f"{h} {c}" for h, c in hosts.most_common(6))
            print(f"[{lang_dir.name}] {n} docs / {words:,} words: {top}")
    verb = "would purge" if args.dry_run else "purged"
    print(f"{verb} {total_docs} documents / {total_words:,} words")


if __name__ == "__main__":
    main()
