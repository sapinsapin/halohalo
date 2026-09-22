"""
Re-gate a published halo-* text corpus by language, and make it
pretraining-ready.

Why this exists: `sapinsapin/halo-hil` carried a `hil` label that came from the
crawler's own language detection and was never verified. An audit on
2026-09-22 found GlotLID calls only ~12 % of its sentences Hiligaynon — most
of it is English and Tagalog news copy. A label nobody checked became training
data, and a language identifier trained on it learned that Tagalog news is
Hiligaynon.

What this does, per document:

  1. clean  — the same cleaner the rest of the repo uses (halolib.cleaner)
  2. split  — into sentences
  3. LID    — every sentence, with HaloLID (decider) and GlotLID (second
              opinion), giving a character-weighted share per language
  4. decide — keep whole documents that are mostly the target language;
              *salvage* the target-language sentences out of mixed documents
              when enough of them survive; drop the rest
  5. dedup  — exact (md5) then near-duplicate (MinHash LSH, Jaccard 0.75)
  6. emit   — FineWeb-compatible columns + the LID evidence, split train/test

Salvage matters here: bilingual Philippine news sites run the same article in
English and in the local language on one page. Dropping the whole page loses
real Hiligaynon; keeping the whole page poisons the corpus. Taking the
target-language sentences keeps what is real and labels how it was obtained.

  python scripts/refilter_corpus.py --repo sapinsapin/halo-hil --lang hil
  python scripts/refilter_corpus.py --repo sapinsapin/halo-hil --lang hil --push

Writes $FINETUNE_DIR/refilter/{lang}/ and, with --push, replaces the dataset
on the Hub (the previous revision stays in the repo's git history, and the new
card links to it).
"""

import argparse
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from halolib import clean_text, is_usable  # noqa: E402
from halolib.lid import LANGS, split_sentences  # noqa: E402
from halolib.scrape.dedup import DedupIndex, content_hash  # noqa: E402

OUT = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs")) / "refilter"


def analyse(text: str, lang: str, lid) -> dict:
    """Character-weighted language shares over a document's sentences."""
    sents = split_sentences(text)
    if not sents:
        return {"share": {}, "target_share": 0.0, "kept_text": "", "n_sent": 0}
    share: Counter = Counter()
    kept = []
    for s in sents:
        p = lid.halo.predict(s) if lid.halo else lid.glot.predict(s)
        share[p.lang] += len(s)
        if p.lang == lang and p.score >= 0.5:
            kept.append(s)
    total = sum(share.values()) or 1
    return {
        "share": {k: round(v / total, 4) for k, v in share.most_common(5)},
        "target_share": share[lang] / total,
        "kept_text": " ".join(kept),
        "n_sent": len(sents),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--repo", required=True, help="source dataset, e.g. sapinsapin/halo-hil")
    ap.add_argument("--lang", required=True, choices=LANGS)
    ap.add_argument("--target-repo", default=None, help="defaults to --repo (in-place)")
    ap.add_argument("--min-ratio", type=float, default=0.5,
                    help="keep a whole document when the target language is at least this share")
    ap.add_argument("--salvage-ratio", type=float, default=0.15,
                    help="below --min-ratio but above this, keep the target-language sentences")
    ap.add_argument("--min-words", type=int, default=30)
    ap.add_argument("--no-salvage", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="debug: only this many documents")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    target_repo = args.target_repo or args.repo
    token = os.environ.get("HF_TOKEN")
    out_dir = OUT / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset, DatasetDict, load_dataset
    from huggingface_hub import HfApi

    from halolib.lid import default_ensemble

    print(f"loading {args.repo} ...")
    ds = load_dataset(args.repo, split="train", token=token)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"  {len(ds)} documents")

    api = HfApi(token=token)
    src_sha = api.dataset_info(args.repo).sha
    print(f"  source revision {src_sha[:12]}")

    lid = default_ensemble(use_glotlid=True)
    dedup = DedupIndex()
    rows, stats = [], Counter()
    glot_share_sum = Counter()
    now = datetime.now(timezone.utc)

    for i, r in enumerate(ds):
        if i and i % 1000 == 0:
            print(f"  {i}/{len(ds)}  kept={stats['keep_whole']}+{stats['keep_salvaged']} "
                  f"dropped={stats['drop_offlang'] + stats['drop_short'] + stats['dup_exact'] + stats['dup_near']}")
        raw = r.get("text_cleaned") or r.get("text") or ""
        text = clean_text(raw)
        if not text:
            stats["drop_empty"] += 1
            continue

        a = analyse(text, args.lang, lid)
        # second opinion on the document as a whole, for the record
        gv, gagree = lid.glot.predict_document(text) if lid.glot else (None, 0.0)
        if gv:
            glot_share_sum[gv.lang] += 1

        if a["target_share"] >= args.min_ratio:
            final, how = text, "whole"
        elif (not args.no_salvage) and a["target_share"] >= args.salvage_ratio and a["kept_text"]:
            final, how = a["kept_text"], "salvaged"
        else:
            stats["drop_offlang"] += 1
            continue

        if not is_usable(final, min_words=args.min_words):
            stats["drop_short"] += 1
            continue

        why = dedup.check_and_add(r.get("url") or f"row{i}", final)
        if why:
            stats[f"dup_{why}"] += 1
            continue

        stats[f"keep_{how}"] += 1
        rows.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, (r.get("url") or "") + final[:200])),
            "text": final,
            "url": r.get("url") or "",
            "date": r.get("date") or "",
            "dump": r.get("dump") or "",
            "language": args.lang,
            "source": f"{args.repo.split('/')[-1]}:refiltered",
            "extraction": how,
            "word_count": len(final.split()),
            "token_count": len(final.split()),
            "content_hash": content_hash(final),
            "lid_target_share": round(a["target_share"], 4),
            "lid_share": json.dumps(a["share"], ensure_ascii=False),
            "lid_glotlid_doc": gv.lang if gv else "",
            "crawled_at": now.isoformat(timespec="seconds"),
        })

    n_in, n_out = len(ds), len(rows)
    words = sum(r["word_count"] for r in rows)
    report = {
        "source_repo": args.repo, "source_revision": src_sha, "language": args.lang,
        "documents_in": n_in, "documents_out": n_out,
        "retention": round(n_out / max(n_in, 1), 4),
        "words_out": words,
        "min_ratio": args.min_ratio, "salvage_ratio": args.salvage_ratio,
        "counts": dict(stats),
        "glotlid_document_verdicts": dict(glot_share_sum.most_common(6)),
        "generated": now.isoformat(timespec="seconds"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print("\n" + json.dumps(report, indent=1))

    if not rows:
        sys.exit("nothing survived the filter; not writing anything")

    dd = DatasetDict({"train": Dataset.from_list(rows)})
    from halolib.fineweb import train_test_split
    dd = train_test_split(dd)
    for split, d in dd.items():
        d.to_parquet(str(out_dir / f"{split}.parquet"))
    print(f"wrote {out_dir}/train.parquet ({len(dd['train'])}) + test.parquet ({len(dd['test'])})")

    if args.push:
        card = build_card(args, report, dd)
        (out_dir / "README.md").write_text(card, encoding="utf-8")
        print(f"pushing to {target_repo} ...")
        dd.push_to_hub(target_repo, token=token,
                       commit_message=f"Re-filter to verified {args.lang} (was {report['retention']:.0%} retained)")
        api.upload_file(path_or_fileobj=str(out_dir / "README.md"), path_in_repo="README.md",
                        repo_id=target_repo, repo_type="dataset",
                        commit_message="Rewrite card after language re-filtering")
        api.upload_file(path_or_fileobj=str(out_dir / "report.json"), path_in_repo="refilter_report.json",
                        repo_id=target_repo, repo_type="dataset",
                        commit_message="Add the re-filtering report")
        print(f"pushed https://huggingface.co/datasets/{target_repo}")


def build_card(args, rep, dd) -> str:
    name = (args.target_repo or args.repo).split("/")[-1]
    c = rep["counts"]
    kept_whole = c.get("keep_whole", 0)
    kept_salv = c.get("keep_salvaged", 0)
    return f"""---
license: mit
language:
- {args.lang}
pretty_name: {name}
task_categories:
- text-generation
tags: [philippine-languages, low-resource, pretraining, language-filtered]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

# {name}

Web text in **{args.lang}**, re-filtered by language and prepared for
pretraining.

## What changed, and why it had to

The earlier version of this dataset was labelled `{args.lang}` by the
crawler's own language detection, and that label was never verified. An audit
on 2026-09-22 found that **most of it was not {args.lang}**: over a random
sample of 1,499 sentences, GlotLID v3 called 44 % English, 22 %
Filipino/Tagalog and only 12 % Hiligaynon — much of the corpus was Tagalog
news copy and site boilerplate.

That mattered beyond this dataset. A language identifier trained on these
labels learned that Tagalog news is Hiligaynon, and then rejected 5,747
genuinely Filipino pages as Hiligaynon in a downstream scrape.

This revision re-gates every document by language:

| step | what it does |
|---|---|
| clean | the repo's shared boilerplate cleaner (`halolib.cleaner`) |
| split | documents into sentences |
| identify | every sentence, with [halo-lid](https://huggingface.co/sapinsapin/halo-lid) deciding and [GlotLID v3](https://huggingface.co/cis-lmu/glotlid) as a second opinion, giving a character-weighted share per language |
| decide | keep a document whole when ≥{args.min_ratio:.0%} of it is {args.lang}; **salvage** the {args.lang} sentences when the document is ≥{args.salvage_ratio:.0%} but mixed; drop the rest |
| deduplicate | md5 exact, then MinHash LSH near-duplicates at Jaccard 0.75 |

Salvage exists because bilingual Philippine news sites publish the same
article in English and in the local language on one page. Dropping the page
loses real {args.lang}; keeping it poisons the corpus. The `extraction`
column records which documents were salvaged so you can exclude them.

## Size

| | documents | |
|---|---|---|
| before | {rep['documents_in']:,} | unverified label |
| **after** | **{rep['documents_out']:,}** | {rep['retention']:.1%} retained, {rep['words_out']:,} words |
| ├ kept whole | {kept_whole:,} | ≥{args.min_ratio:.0%} {args.lang} |
| └ salvaged | {kept_salv:,} | {args.lang} sentences taken from mixed pages |

Dropped: {c.get('drop_offlang', 0):,} off-language, {c.get('drop_short', 0):,}
too short after filtering, {c.get('dup_exact', 0) + c.get('dup_near', 0):,}
duplicates.

Splits: {len(dd['train']):,} train / {len(dd['test']):,} test.

## Columns

| column | meaning |
|---|---|
| `text` | cleaned text, language-filtered — use this for pretraining |
| `language` | `{args.lang}`, now verified rather than assumed |
| `extraction` | `whole` or `salvaged` |
| `lid_target_share` | share of the original document's characters identified as `{args.lang}` |
| `lid_share` | the full per-language share, as JSON |
| `lid_glotlid_doc` | GlotLID's verdict on the whole original document |
| `url`, `date`, `dump`, `source` | provenance |
| `word_count`, `token_count`, `content_hash` | FineWeb-compatible fields |

## Honest limitations

- The filter is only as good as the identifier. `halo-lid` scores 0.853 on
  Hiligaynon against human-labelled read prompts, so a few percent of these
  sentences are probably still misfiled, and some genuine {args.lang} was
  certainly thrown away.
- Salvaged documents are sentence sequences from a page, not continuous prose.
  They are fine for language modelling and poor for anything needing document
  structure.
- This is web text: the domain skews to news and blogs, and it has had no
  quality scoring beyond length and language.
- The previous, unfiltered revision is still in this repository's git history
  at `{rep['source_revision'][:12]}` if you need it for reproducibility.

Produced by [`scripts/refilter_corpus.py`](https://github.com/sapinsapin/halohalo)
in the halohalo pipeline. Report: `refilter_report.json`.
"""


if __name__ == "__main__":
    main()
