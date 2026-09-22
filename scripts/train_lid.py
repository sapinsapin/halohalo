"""
Train HaloLID — a fastText language identifier for the ten PLD languages —
and measure it against GlotLID on held-out text.

Why our own model when GlotLID exists: GlotLID separates 2,000 languages and
is tuned for that; our problem is ten closely related Austronesian languages
plus English and a few confusables, on short web and prompt sentences.
A model trained only on that boundary is smaller (~9 MB quantised vs 1.7 GB),
faster, and — the number this script produces — more accurate on our data.
The `other` class is trained on Indonesian and Malay web text so the model
can abstain on the nearest non-Philippine neighbours instead of forcing every
Malayo-Polynesian page into Cebuano.

Data, deduplicated at the sentence level before splitting (PLD prompts repeat
across speakers; a random split would leak them):
  PLD        read/isolated prompts, ten languages   (local)   labels: human
  halohalo   web text, sentence-split               (Hub)     labels: page-level
  FineWeb-2  ind_Latn / zsm_Latn -> "other"         (Hub)     labels: FW-2 LID

Label noise, and what we do about it. Web sentences inherit their *page's*
language, and some pages' labels are wrong outright. Round 1 trained on page
labels and learned to call English `hil` (eng accuracy 0.63). Round 2
relabelled confident English, but still trusted the rest — and halo-hil turns
out to be largely Tagalog and English, so round 2 learned "Tagalog news is
Hiligaynon" and the Filipino scrape rejected 5,747 genuine Filipino pages.
Round 3 uses consensus labels: a web sentence trains only if GlotLID agrees
with its page label (confident English moves to `eng`). See consensus_web.

Evaluation reports two held-out sets separately:
  pld  — human-labelled prompts; the number to trust
  web  — page-labelled sentences; noisier labels, closer to scrape conditions

  python scripts/train_lid.py                 # train + eval -> $FINETUNE_DIR/lid/
  python scripts/train_lid.py --push          # also publish sapinsapin/halo-lid
  python scripts/train_lid.py --eval-only     # re-score the existing model

Improves as the scrape grows: --extra-parquet finetune_runs/scrape/text folds
accepted, undisputed, high-confidence scraped pages back into training.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from halolib.lid import LANGS, OTHER, GlotLID, HaloLID, normalize, split_sentences  # noqa: E402
from halolib.scrape.seeds import _code  # noqa: E402

OUT = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs")) / "lid"
NEG_CONFIGS = {"ind_Latn": 4000, "zsm_Latn": 3000}
ENG_RELABEL_SCORE = 0.8


def gather(token: str | None, extra_parquet: Path | None) -> list[tuple[str, str, str]]:
    """Return unique (lang, text, src) triples; src in {pld, web, neg}."""
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def add(lang, text, src):
        t = normalize(text)
        if len(t) < 3 or t in seen:          # dedup on text alone: one label per sentence
            return
        seen.add(t)
        rows.append((lang, t, src))

    root = Path(os.environ.get("PLD_RAW", "/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD"))
    if root.exists():
        from halolib.pld import index_corpus
        entries, _ = index_corpus(root)
        for e in entries:
            if not e.get("text_is_prompt") and e.get("sentence"):
                add(e["language"], e["sentence"], "pld")
        print(f"  PLD: {len(rows)} unique sentences")

    from datasets import load_dataset
    # halohalo already contains BantayWika (it is appended in build_halohalo.py)
    n0 = len(rows)
    try:
        ds = load_dataset("sapinsapin/halohalo", split="train", token=token)
        for r in ds:
            lang = _code(r.get("language"))
            if lang in LANGS and r.get("text"):
                for s in split_sentences(r["text"])[:40]:
                    add(lang, s, "web")
    except Exception as exc:
        print(f"  skip halohalo: {type(exc).__name__}")
    print(f"  halohalo (+BantayWika): +{len(rows) - n0}")

    if extra_parquet and extra_parquet.exists():
        import pyarrow.parquet as pq
        n0 = len(rows)
        for p in sorted(extra_parquet.glob("*/shard-*.parquet")):
            t = pq.read_table(p, columns=["language", "text", "lid_models_agree", "lid_score"])
            cols = [t.column(c).to_pylist() for c in ("language", "text", "lid_models_agree", "lid_score")]
            for lang, text, agree, score in zip(*cols):
                if agree and score >= 0.9:       # only confident, undisputed pages
                    for s in split_sentences(text)[:40]:
                        add(lang, s, "web")
        print(f"  scrape shards: +{len(rows) - n0}")

    for cfg, n in NEG_CONFIGS.items():
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-2", name=cfg, split="train",
                              streaming=True, token=token)
            k = 0
            for r in ds:
                for s in split_sentences(r["text"])[:10]:
                    add(OTHER, s, "neg")
                    k += 1
                if k >= n:
                    break
            print(f"  fineweb-2 {cfg}: +{k} -> other")
        except Exception as exc:
            print(f"  skip fineweb-2 {cfg}: {type(exc).__name__}: {str(exc)[:80]}")
    return rows


def consensus_web(rows, glot: GlotLID) -> list[tuple[str, str, str]]:
    """Keep a web sentence only where GlotLID agrees with its page label.

    Page labels are not sentence labels, and some page labels are simply
    wrong: an audit of sapinsapin/halo-hil found GlotLID calls 44 % of its
    sentences English, 21 % Filipino and 12 % Hiligaynon (Tagalog tabloid
    content under a `hil` label). Round 2 trained on those labels and learned
    "Tagalog news is Hiligaynon"; the Filipino scrape then rejected 5,747
    FineWeb-2 Filipino pages as hil. So web text now enters training only as
    consensus: GlotLID's label == page label, or confident English (moved to
    `eng`). PLD sentences carry human labels and bypass this.

    The cost is that web data shrinks for languages GlotLID is weak on (tsg,
    war) — which is fine, because PLD covers those and their web data is tiny.
    """
    out, moved, dropped, kept = [], Counter(), Counter(), Counter()
    for lang, text, src in rows:
        if src == "web":
            p = glot.predict(text)
            if p.lang == "eng" and p.score >= ENG_RELABEL_SCORE:
                if lang != "eng":
                    moved[lang] += 1
                lang = "eng"
            elif p.lang != lang:
                dropped[lang] += 1
                continue
            else:
                kept[lang] += 1
        out.append((lang, text, src))
    print(f"  web consensus: kept {sum(kept.values())} {dict(kept)}")
    print(f"  web -> eng:    {sum(moved.values())} {dict(moved)}")
    print(f"  web dropped:   {sum(dropped.values())} {dict(dropped)}")
    return out


def cap_and_split(rows, max_per_lang: int, test_frac: float = 0.1, seed: int = 42):
    """Per language: shuffle, cap, then hold out test_frac *per source*, so
    both the pld and the web test sets exist for every language that has
    both. Capping is PLD-first: human labels are kept before page labels."""
    rng = random.Random(seed)
    by: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for lang, text, src in rows:
        by[lang][src].append(text)

    train, test = [], []
    kept = {}
    for lang, srcs in by.items():
        budget = max_per_lang
        for src in ("pld", "web", "neg"):
            items = srcs.get(src, [])
            rng.shuffle(items)
            items = items[:budget]
            budget -= len(items)
            k = max(1, int(len(items) * test_frac)) if items else 0
            test += [(lang, x, src) for x in items[:k]]
            train += [(lang, x, src) for x in items[k:]]
        kept[lang] = max_per_lang - budget
    rng.shuffle(train)
    return train, test, kept


def write_ft(rows, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for lang, text, _ in rows:
            f.write(f"__label__{lang} {text}\n")


def evaluate(model, test, name: str, subset: str) -> dict:
    correct, total = Counter(), Counter()
    conf = defaultdict(Counter)
    short_ok = short_n = 0
    t0 = time.time()
    for lang, text, _ in test:
        p = model.predict(text)
        total[lang] += 1
        conf[lang][p.lang] += 1
        ok = p.lang == lang
        correct[lang] += ok
        if len(text.split()) <= 5:
            short_n += 1
            short_ok += ok
    labels = [l for l in LANGS if total[l]]
    f1s = []
    for l in labels:
        tp = conf[l][l]
        fp = sum(conf[o][l] for o in conf if o != l)
        fn = total[l] - tp
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    n = len(test)
    return {"model": name, "subset": subset, "n": n,
            "accuracy": round(sum(correct.values()) / n, 4) if n else None,
            "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "short_acc": round(short_ok / short_n, 4) if short_n else None,
            "per_lang": {l: round(correct[l] / total[l], 4) for l in sorted(total)},
            "per_lang_n": dict(total),
            "ms_per_1k": round((time.time() - t0) / max(n, 1) * 1e6, 1),
            "confusions": {l: dict(conf[l].most_common(3)) for l in sorted(conf)}}


def print_table(results):
    print(f"\n{'model':9} {'subset':6} {'n':>6} {'acc':>7} {'macroF1':>8} {'short':>7} {'ms/1k':>7}")
    for r in results:
        print(f"{r['model']:9} {r['subset']:6} {r['n']:6d} {r['accuracy']:7.4f} "
              f"{r['macro_f1']:8.4f} {(r['short_acc'] or 0):7.4f} {r['ms_per_1k']:7.0f}")
    for subset in ("pld", "web"):
        rs = [r for r in results if r["subset"] == subset]
        if not rs:
            continue
        print(f"\nper-language accuracy — {subset}:")
        print(f"{'lang':6} {'n':>5}" + "".join(f"{r['model']:>10}" for r in rs))
        for l in sorted(rs[0]["per_lang"]):
            print(f"{l:6} {rs[0]['per_lang_n'][l]:5d}"
                  + "".join(f"{r['per_lang'].get(l, 0):10.4f}" for r in rs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--max-per-lang", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--extra-parquet", type=Path, help="scrape text dir to fold in")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-glotlid", action="store_true",
                    help="skip GlotLID (no English relabelling, no comparison row)")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    glot = None if args.no_glotlid else GlotLID()

    print("gathering ...")
    rows = gather(token, args.extra_parquet)
    if glot:
        rows = consensus_web(rows, glot)
    train, test, kept = cap_and_split(rows, args.max_per_lang)
    print("  kept: " + "  ".join(f"{l}={n}" for l, n in sorted(kept.items())))
    write_ft(train, OUT / "train.txt")
    write_ft(test, OUT / "test.txt")
    print(f"train={len(train)} test={len(test)}")

    if not args.eval_only:
        import fasttext
        t0 = time.time()
        model = fasttext.train_supervised(
            input=str(OUT / "train.txt"), lr=0.5, epoch=args.epochs, dim=args.dim,
            wordNgrams=2, minn=2, maxn=5, minCount=2, loss="softmax",
            thread=os.cpu_count() or 4)
        print(f"trained in {time.time() - t0:.0f}s; {len(model.words)} words")
        model.save_model(str(OUT / "model.bin"))
        model.quantize(input=str(OUT / "train.txt"), qnorm=True, retrain=True, cutoff=200000)
        model.save_model(str(OUT / "model.ftz"))
        print(f"saved {OUT/'model.ftz'} ({(OUT/'model.ftz').stat().st_size/1e6:.1f} MB)")

    halo = HaloLID(OUT / "model.ftz")
    results = []
    for subset in ("pld", "web"):
        sub = [r for r in test if r[2] in (subset, "neg")] if subset == "web" else \
              [r for r in test if r[2] == subset]
        results.append(evaluate(halo, sub, "halolid", subset))
        if glot:
            results.append(evaluate(glot, sub, "glotlid", subset))
    print_table(results)

    (OUT / "results.json").write_text(json.dumps(
        {"train_size": len(train), "test_size": len(test), "kept_per_lang": kept,
         "results": results}, indent=1))

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        repo = HaloLID.REPO
        api.create_repo(repo, repo_type="model", exist_ok=True)

        def row(model, subset):
            r = next((x for x in results if x["model"] == model and x["subset"] == subset), None)
            return (f"| {model} | {subset} | {r['n']} | {r['accuracy']:.4f} | "
                    f"{r['macro_f1']:.4f} | {r['short_acc']} |") if r else ""

        table = "\n".join(filter(None, [row(m, s) for s in ("pld", "web")
                                         for m in ("halolid", "glotlid")]))
        card = f"""---
language: [tl, ceb, hil, ilo, bcl, war, pam, pag, tsg, en]
license: mit
library_name: fasttext
tags: [language-identification, fasttext, filipino, philippine-languages]
---

# halo-lid

fastText language identifier for the ten languages of the Philippine Language
Dataset — `{' '.join(LANGS)}` — plus `other` (Indonesian and Malay, so the
nearest non-Philippine neighbours are rejected, not absorbed).

Trained by `scripts/train_lid.py` in [halohalo](https://github.com/sapinsapin/halohalo)
on sentence-deduplicated PLD prompts and halohalo web text (which includes
BantayWika). Web sentences that GlotLID confidently identifies as English are
relabelled `eng`, because they otherwise inherit their page's label.

| model | test set | n | accuracy | macro F1 | ≤5-word acc |
|---|---|---|---|---|---|
{table}

**pld** is human-labelled read prompts — the number to trust. **web** uses
page-level labels, so it is noisier, and closer to scrape conditions.

```python
from halolib.lid import HaloLID
HaloLID().predict("Maayong buntag sa inyong tanan")   # -> ceb
```
"""
        (OUT / "README.md").write_text(card)
        for f in ("model.ftz", "README.md", "results.json"):
            api.upload_file(path_or_fileobj=str(OUT / f), path_in_repo=f, repo_id=repo)
        print(f"pushed -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
