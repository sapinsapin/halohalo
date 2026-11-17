"""
Benchmark HaloLID against the public LID leaderboard (Omneity Labs,
https://huggingface.co/spaces/omneity-labs/lid-benchmark), on our languages.

The leaderboard publishes per-language results for 18 open-set models on 8
benchmarks; it does not run submissions. So this script runs HaloLID over a
benchmark's *full* test set (every language, so text in the other ~230
languages that HaloLID wrongly claims counts as false positives) and scores
it per language the same way for everyone:

  recall     share of gold-L samples predicted L (the leaderboard's per-language
             "accuracy")
  precision TP / (TP + samples of any other gold language predicted L); for
             leaderboard models from their published confusion tables
  F1

HaloLID is a closed set of ten languages plus `other`. Recall alone would
flatter it (it cannot confuse Cebuano with Javanese, because it does not know
Javanese); precision on the full benchmark is what keeps the comparison fair.

Codes. Benchmarks label Tagalog/Filipino as tgl_Latn (WiLI, Bouquet) or
fil_Latn (FLORES+); CommonLID has both. HaloLID's `fil` class is scored as
correct for either. Central Bikol is bcl_Latn; CommonLID also has the Bikol
macrolanguage bik_Latn, which `bcl` is scored against as well.

--pld scores HaloLID, GlotLID and Gherbal-v4 (Omneity's model, through the
Sawalni API) on our own held-out test set instead: the 4,035 human-labelled
PLD sentences of train_lid.py's fixed hash split. It is the only test set
with all ten languages; the leaderboard's benchmarks have no Hiligaynon at
all and Tausug only in Bouquet. Gherbal-v4 supports six of the ten (not bcl,
hil, pam, tsg), so the table also reports the mean over those six. Its
answers are cached in lid_bench/gherbal_cache.json; the key is read from
SAWALNI_API_KEY or the git-ignored file `salwanikey`.

  python scripts/bench_lid.py --benchmark wili-2018
  python scripts/bench_lid.py --benchmark wili-2018 --model finetune_runs/lid/model.ftz
  python scripts/bench_lid.py --pld
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from halolib.lid import LANGS, OTHER, HaloLID  # noqa: E402

FINETUNE = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs"))
OUT = FINETUNE / "lid_bench"
BOARD = "omneity-labs/lid-benchmark"

# HaloLID label -> the benchmark gold labels it is correct for
GOLD_FOR = {l: {f"{l}_Latn"} for l in LANGS}
GOLD_FOR["fil"] = {"fil_Latn", "tgl_Latn"}
GOLD_FOR["bcl"] = {"bcl_Latn", "bik_Latn"}


def load_wili():
    """WiLI-2018 test split: 117,500 Wikipedia paragraphs, 235 languages,
    labelled with bare ISO 639-3 codes (and a few Wikipedia codes)."""
    from datasets import load_dataset
    ds = load_dataset("MartinThoma/wili_2018", split="test")
    names = ds.features["label"].names
    for text, lab in zip(ds["sentence"], ds["label"]):
        code = names[lab]
        # the leaderboard's codes are ISO 639-3 + script; ours are all Latin
        yield text, f"{code}_Latn" if len(code) == 3 else code


LOADERS = {"wili-2018": load_wili}


def board_scores(benchmark: str, golds: set[str]) -> dict[str, dict[str, dict]]:
    """{model: {gold_lang: {recall, precision, f1, n}}} from the leaderboard's
    per-language table (scope 'full'), precision rebuilt from confusions."""
    from datasets import load_dataset
    pl = load_dataset(BOARD, "results_per_language", split="train").to_pandas()
    pl = pl[(pl.benchmark == benchmark) & (pl.scope == "full")]
    out = {}
    for model, g in pl.groupby("model"):
        fp = Counter()
        tp = {}
        n = {}
        for r in g.itertuples():
            conf = r.confusions_json
            conf = json.loads(conf) if isinstance(conf, str) else {}
            for pred, c in conf.items():
                fp[pred] += c                  # gold r.language predicted as `pred`
            tp[r.language] = round(r.accuracy * r.n_samples)
            n[r.language] = r.n_samples
        res = {}
        for lang in golds & set(n):
            rec = tp[lang] / n[lang] if n[lang] else 0.0
            prec = tp[lang] / (tp[lang] + fp[lang]) if tp[lang] + fp[lang] else 0.0
            res[lang] = {"recall": rec, "precision": prec, "n": n[lang],
                         "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
        out[model] = res
    return out


def halolid_scores(samples, model_path: Path | None) -> dict[str, dict]:
    """Run HaloLID on every sample; score per benchmark gold language."""
    lid = HaloLID(model_path) if model_path else HaloLID()
    to_gold = {}
    for ours, golds in GOLD_FOR.items():
        for g in golds:
            to_gold[g] = ours
    n, tp, fp = Counter(), Counter(), Counter()
    total = 0
    for text, gold in samples:
        total += 1
        pred = lid.predict(text).lang          # one of LANGS or OTHER
        n[gold] += 1
        if pred == OTHER:
            continue
        if gold in GOLD_FOR[pred]:
            tp[gold] += 1
        else:
            # a false positive for every gold label this class answers for
            for g in GOLD_FOR[pred]:
                fp[g] += 1
    res = {}
    for gold, ours in to_gold.items():
        if not n[gold]:
            continue
        rec = tp[gold] / n[gold]
        prec = tp[gold] / (tp[gold] + fp[gold]) if tp[gold] + fp[gold] else 0.0
        res[gold] = {"recall": rec, "precision": prec, "n": n[gold],
                     "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    print(f"  scored {total:,} samples")
    return res


# -- our own test set: PLD held-out, three models ------------------------------

GHERBAL_SUPPORTED = {"ceb", "eng", "fil", "ilo", "pag", "war"}


def pld_test_set() -> list[tuple[str, str]]:
    """The PLD half of train_lid's held-out set: same normalisation, same
    sentence dedup, same hash split, so it is the set the gate scores on."""
    from halolib.lid import normalize
    from halolib.pld import index_corpus
    from scripts.train_lid import _h
    root = Path(os.environ.get("PLD_RAW", "/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD"))
    entries, _ = index_corpus(root)
    cut = int(0.1 * 2 ** 48)
    seen, out = set(), []
    for e in entries:
        if e.get("text_is_prompt") or not e.get("sentence"):
            continue
        t = normalize(e["sentence"])
        if len(t) < 3 or t in seen:
            continue
        seen.add(t)
        if _h(t) < cut:
            out.append((e["language"], t))
    return out


def _sawalni_key() -> str:
    key = os.environ.get("SAWALNI_API_KEY")
    if not key and (ROOT / "salwanikey").exists():
        key = (ROOT / "salwanikey").read_text().strip()
    if not key:
        raise SystemExit("no Sawalni key: set SAWALNI_API_KEY or create ./salwanikey")
    return key


def gherbal_predict(texts: list[str], batch: int = 50) -> list[str]:
    """Gherbal-v4 top label per text, mapped to our label space; cached."""
    import hashlib
    import time
    import requests
    cache_path = OUT / "gherbal_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key = lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()
    todo = [t for t in dict.fromkeys(texts) if key(t) not in cache]
    headers = {"Authorization": f"Bearer {_sawalni_key()}"}
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        for attempt in range(6):
            r = requests.post("https://api.sawalni.com/v1/language/identify", headers=headers,
                              json={"model": "gherbal-v4", "input": chunk}, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(f"Sawalni API {r.status_code}: {r.text[:200]}")
        else:
            raise SystemExit(f"Sawalni API kept failing: {r.status_code}")
        for t, d in zip(chunk, r.json()["data"]):
            preds = d.get("predictions") or [{}]
            cache[key(t)] = preds[0].get("language", "")
        OUT.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
        print(f"  gherbal: {min(i + batch, len(todo))}/{len(todo)} new", end="\r")
    to_ours = {g: ours for ours, golds in GOLD_FOR.items() for g in golds}
    return [to_ours.get(cache[key(t)], OTHER) for t in texts]


def closed_set_scores(gold: list[str], pred: list[str]) -> dict:
    labels = sorted(set(gold))
    res = {}
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        n = sum(g == l for g in gold)
        rec = tp / n if n else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        res[l] = {"n": n, "recall": rec, "precision": prec,
                  "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    return res


def run_pld(model_path: Path | None):
    from halolib.lid import GlotLID
    data = pld_test_set()
    gold = [g for g, _ in data]
    texts = [t for _, t in data]
    print(f"PLD held-out: {len(data)} sentences, {len(set(gold))} languages")
    halo = HaloLID(model_path) if model_path else HaloLID()
    glot = GlotLID()
    preds = {"halolid": [halo.predict(t).lang for t in texts],
             "glotlid": [glot.predict(t).lang for t in texts]}
    print("querying gherbal-v4 (Sawalni API) ...")
    preds["gherbal-v4"] = gherbal_predict(texts)
    scores = {m: closed_set_scores(gold, p) for m, p in preds.items()}
    langs = sorted(set(gold))
    print(f"\nper-language F1 on the PLD held-out set ({len(data)} human-labelled sentences)")
    print(f"{'model':12}" + "".join(f"{l:>7}" for l in langs) + f"{'all 10':>8}{'gherbal 6':>10}{'acc':>7}")
    for m, s in scores.items():
        mean = sum(s[l]["f1"] for l in langs) / len(langs)
        six = sum(s[l]["f1"] for l in langs if l in GHERBAL_SUPPORTED) / len(GHERBAL_SUPPORTED)
        acc = sum(g == p for g, p in zip(gold, preds[m])) / len(gold)
        print(f"{m:12}" + "".join(f"{s[l]['f1']:7.3f}" for l in langs) + f"{mean:8.3f}{six:10.3f}{acc:7.3f}")
    print(f"{'n':12}" + "".join(f"{scores['halolid'][l]['n']:7d}" for l in langs))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pld.json").write_text(json.dumps({"languages": langs, "scores": scores}, indent=1))
    print(f"\nwrote {OUT / 'pld.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--benchmark", choices=sorted(LOADERS), default="wili-2018")
    ap.add_argument("--pld", action="store_true", help="score HaloLID, GlotLID, Gherbal-v4 on our PLD test set")
    ap.add_argument("--model", type=Path, help="HaloLID model.ftz (default: the local or Hub model)")
    args = ap.parse_args()
    if args.pld:
        run_pld(args.model)
        return

    print(f"running HaloLID on {args.benchmark} ...")
    halo = halolid_scores(LOADERS[args.benchmark](), args.model)
    golds = set(halo)
    print("loading leaderboard results ...")
    board = board_scores(args.benchmark, golds)
    board["halolid"] = halo

    langs = sorted(golds)
    models = sorted(board, key=lambda m: -sum(board[m].get(l, {}).get("f1", 0) for l in langs))
    print(f"\nper-language F1 on {args.benchmark} (full test set; precision counts every language)")
    print(f"{'model':16}" + "".join(f"{l[:3]:>7}" for l in langs) + f"{'mean':>7}")
    for m in models:
        f1s = [board[m].get(l, {}).get("f1") for l in langs]
        mean = sum(x or 0 for x in f1s) / len(langs)
        print(f"{m:16}" + "".join(f"{x:7.3f}" if x is not None else f"{'—':>7}" for x in f1s) + f"{mean:7.3f}")
    print(f"\n{'n':16}" + "".join(f"{halo[l]['n']:7d}" for l in langs))
    print("\nHaloLID precision / recall:")
    for l in langs:
        print(f"  {l:9} P {halo[l]['precision']:.3f}  R {halo[l]['recall']:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.benchmark}.json"
    out.write_text(json.dumps({"benchmark": args.benchmark, "languages": langs, "scores": board}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
