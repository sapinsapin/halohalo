"""
Build the `other` class for HaloLID from FineWeb-2: sentences in ~180
languages that are not one of our ten.

Why: HaloLID's `other` was trained on Indonesian and Malay only, so on an
open benchmark (WiLI-2018, 235 languages) text in any other language had
nowhere to go: English absorbed 44,701 European paragraphs and scripts the
model had never seen landed on Philippine classes. Precision on our
languages was 0.01-0.22 (see docs/web_scrape_pipeline.md, "Benchmarks").

Which languages, chosen without looking at any benchmark's test set:
  - the largest Latin-script FineWeb-2 languages (default 150)
  - neighbours, weighted up: Indonesian, Malay, Javanese, Sundanese, ...
    and the *other* Philippine languages (Kinaray-a, Chavacano, Maranao,
    ...), which a scraper for our ten must reject, not absorb
  - a few large non-Latin-script languages (HaloLID also applies a script
    rule, so these are a backstop)

One row group (1,000 documents) per language is read over HTTP ranges, so
the whole build downloads a few hundred MB. Output:
$FINETUNE_DIR/lid/negatives.jsonl, one {"lang", "text"} per sentence, which
train_lid.py uses for `other` when it exists.

  python scripts/build_lid_negatives.py
  python scripts/build_lid_negatives.py --top-latin 200 --per-lang 150
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from halolib.lid import LANGS, normalize, split_sentences  # noqa: E402

OUT = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs")) / "lid" / "negatives.jsonl"
REPO = "HuggingFaceFW/fineweb-2"

# Hardest confusions: weighted up (per_lang * NEIGHBOUR_WEIGHT sentences each)
NEIGHBOURS = ["ind", "zsm", "jav", "sun", "min", "ban", "bjn", "ace", "mad", "tet", "bug", "mak", "nij",
              "krj", "cbk", "mrw", "msb", "ivb", "kne", "ifk", "ify", "itv", "agn", "tbw", "sml", "yka",
              "mri", "haw", "smo", "ton", "fij", "spa", "por"]
NEIGHBOUR_WEIGHT = 4
NON_LATIN = ["rus_Cyrl", "arb_Arab", "hin_Deva", "cmn_Hani", "jpn_Jpan", "kor_Hang", "tha_Thai", "heb_Hebr",
             "ell_Grek", "ben_Beng", "tam_Taml", "ukr_Cyrl", "fas_Arab", "mya_Mymr", "khm_Khmr"]
# ours, and what FineWeb-2 calls them (Tagalog is fil_Latn there)
EXCLUDE = set(LANGS) | {"tgl", "bik", "bcl"}


def list_configs(api) -> dict[str, tuple[str, int]]:
    """{config: (first train parquet, total train bytes)}"""
    out: dict[str, list] = {}
    for f in api.list_repo_tree(REPO, path_in_repo="data", repo_type="dataset", recursive=True):
        p = f.path.split("/")
        if len(p) >= 4 and p[2] == "train" and f.path.endswith(".parquet") and not p[1].endswith("_removed"):
            e = out.setdefault(p[1], [f.path, 0])
            e[0] = min(e[0], f.path)
            e[1] += getattr(f, "size", 0) or 0
    return {k: (v[0], v[1]) for k, v in out.items()}


def sample(fs, path: str, n: int, rng: random.Random, per_doc: int = 3) -> list[str]:
    import pyarrow.parquet as pq
    with fs.open(f"datasets/{REPO}/{path}") as fh:
        pf = pq.ParquetFile(fh)
        texts = pf.read_row_group(0, columns=["text"]).column("text").to_pylist()
    rng.shuffle(texts)
    out = []
    for t in texts:
        sents = [normalize(s) for s in split_sentences(t or "")]
        sents = [s for s in sents if len(s) >= 20]
        rng.shuffle(sents)
        out += sents[:per_doc]
        if len(out) >= n:
            break
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--top-latin", type=int, default=150)
    ap.add_argument("--per-lang", type=int, default=150)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from huggingface_hub import HfApi, HfFileSystem
    token = os.environ.get("HF_TOKEN")
    api, fs = HfApi(token=token), HfFileSystem(token=token)
    cfgs = list_configs(api)
    latin = sorted((c for c in cfgs if c.endswith("_Latn") and c.split("_")[0] not in EXCLUDE),
                   key=lambda c: -cfgs[c][1])
    chosen = {c: args.per_lang for c in latin[:args.top_latin]}
    for code in NEIGHBOURS:
        c = f"{code}_Latn"
        if c in cfgs:
            chosen[c] = args.per_lang * NEIGHBOUR_WEIGHT
    for c in NON_LATIN:
        if c in cfgs:
            chosen[c] = args.per_lang // 3
    print(f"{len(chosen)} languages, ~{sum(chosen.values()):,} sentences wanted")

    rng = random.Random(13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    got = Counter()
    with open(tmp, "w", encoding="utf-8") as f:
        for i, (cfg, n) in enumerate(sorted(chosen.items())):
            try:
                sents = sample(fs, cfgs[cfg][0], n, rng)
            except Exception as exc:              # one bad config must not stop the build
                print(f"  skip {cfg}: {type(exc).__name__}: {str(exc)[:80]}")
                continue
            for s in sents:
                f.write(json.dumps({"lang": cfg, "text": s}, ensure_ascii=False) + "\n")
            got[cfg] = len(sents)
            if i % 20 == 0:
                print(f"  {i + 1}/{len(chosen)} {cfg}: {len(sents)}")
    os.replace(tmp, args.out)
    print(f"wrote {sum(got.values()):,} sentences from {len(got)} languages -> {args.out}")


if __name__ == "__main__":
    main()
