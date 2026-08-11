"""
Tokenizer fit report for Philippine languages.

Measures how well a pretrained tokenizer covers our corpora, and how much a
vocabulary extension would buy. Background and the numbers this produced:
docs/reference/tokenizers.md

  python scripts/tokenizer_report.py --corpus pld
  python scripts/tokenizer_report.py --corpus pld --morfessor --held-out
  python scripts/tokenizer_report.py --corpus pld --extension-curve

Fertility = tokenizer units spent per whitespace word. English sits near 1.1
with Whisper's vocabulary; anything much higher means the vocabulary does not
cover the language, and for ASR that is decoder steps you pay for on every
utterance.
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def load_texts(corpus: str, max_per_lang: int) -> dict[str, list[str]]:
    """{language: [transcript, ...]} from a local corpus."""
    by_lang: dict[str, list[str]] = defaultdict(list)

    if corpus == "pld":
        from halolib.pld import index_corpus
        root = Path(os.environ.get(
            "PLD_RAW", "/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD"))
        if not root.exists():
            sys.exit(f"PLD raw corpus not found at {root}; set PLD_RAW")
        entries, _ = index_corpus(root)
        for e in entries:
            # spontaneous rows store the elicitation question, not a
            # transcript — they would skew every count here
            if e.get("text_is_prompt"):
                continue
            text = (e.get("sentence") or "").strip()
            if text and len(by_lang[e["language"]]) < max_per_lang:
                by_lang[e["language"]].append(text)

    elif corpus == "fsc":
        from datasets import load_dataset
        ds = load_dataset("sapinsapin/filipinospeechcorpus",
                          split="train", token=os.environ.get("HF_TOKEN"))
        for r in ds.select(range(min(max_per_lang, len(ds)))):
            if r.get("sentence"):
                by_lang["fil"].append(r["sentence"])

    else:
        sys.exit(f"unknown corpus {corpus!r}")

    return by_lang


def fertility_table(by_lang, tok):
    print(f"{'lang':6} {'utts':>7} {'words':>9} {'types':>8} "
          f"{'tok/word':>9} {'chars/tok':>10}")
    print("-" * 54)
    out = {}
    for lang, texts in sorted(by_lang.items()):
        words = ntok = nchar = 0
        types = Counter()
        for t in texts:
            ws = WORD.findall(t.lower())
            if not ws:
                continue
            words += len(ws)
            nchar += sum(len(w) for w in ws)
            types.update(ws)
            ntok += len(tok(" " + " ".join(ws), add_special_tokens=False).input_ids)
        if not words:
            continue
        out[lang] = ntok / words
        print(f"{lang:6} {len(texts):7d} {words:9d} {len(types):8d} "
              f"{ntok/words:9.3f} {nchar/ntok:10.2f}")

    if "eng" in out:
        print(f"\nEnglish baseline {out['eng']:.3f} — relative cost:")
        for lang, f in sorted(out.items(), key=lambda kv: -kv[1]):
            if lang != "eng":
                print(f"  {lang:6} {f:.3f}  {f/out['eng']:.2f}x")
    return out


def morfessor_compare(by_lang, tok, held_out: bool):
    """Morph units vs BPE. With --held-out, train on 80% of utterances and
    measure on the rest — the only comparison that isn't flattered by
    Morfessor memorising the training vocabulary."""
    try:
        import morfessor
    except ImportError:
        sys.exit("pip install morfessor")

    print(f"\n{'lang':6} {'types':>7} {'OOV%':>7} {'morph/word':>11} {'bpe/word':>9}")
    print("-" * 46)
    cache: dict[str, int] = {}

    def bpe_len(w):
        if w not in cache:
            cache[w] = len(tok(" " + w, add_special_tokens=False).input_ids)
        return cache[w]

    for lang, texts in sorted(by_lang.items()):
        if lang == "eng":
            continue
        cut = int(len(texts) * 0.8) if held_out else len(texts)
        train = Counter(w for t in texts[:cut] for w in WORD.findall(t.lower()))
        test = (Counter(w for t in texts[cut:] for w in WORD.findall(t.lower()))
                if held_out else train)
        if len(train) < 500 or not test:
            continue

        model = morfessor.BaselineModel()
        model.load_data([(c, w) for w, c in train.items()])
        model.train_batch(finish_threshold=0.01)

        total = sum(test.values())
        oov = sum(c for w, c in test.items() if w not in train) / total
        nm = sum(len(model.viterbi_segment(w)[0]) * c for w, c in test.items())
        nb = sum(bpe_len(w) * c for w, c in test.items())
        print(f"{lang:6} {len(train):7d} {oov*100:6.1f}% "
              f"{nm/total:11.3f} {nb/total:9.3f}")


def extension_curve(by_lang, tok, steps):
    """Fertility if the top-N units became single tokens — an upper bound on
    what vocabulary extension buys, since it assumes the new embeddings learn
    perfectly."""
    pooled = Counter()
    for lang, texts in by_lang.items():
        if lang == "eng":
            continue
        for t in texts:
            pooled.update(WORD.findall(t.lower()))
    if not pooled:
        return

    cache = {w: len(tok(" " + w, add_special_tokens=False).input_ids) for w in pooled}
    total_tok = sum(cache[w] * c for w, c in pooled.items())
    total_words = sum(pooled.values())
    base = total_tok / total_words

    print(f"\n{'added':>8} {'tok/word':>9} {'reduction':>10} {'coverage':>9}")
    print("-" * 40)
    print(f"{0:8d} {base:9.3f} {'—':>10} {'—':>9}")
    for n in steps:
        top = [w for w, _ in pooled.most_common(n)]
        saved = sum((cache[w] - 1) * pooled[w] for w in top)
        cov = sum(pooled[w] for w in top) / total_words
        new = (total_tok - saved) / total_words
        print(f"{n:8d} {new:9.3f} {(1-new/base)*100:9.1f}% {cov*100:8.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--corpus", choices=["pld", "fsc"], default="pld")
    ap.add_argument("--tokenizer", default="openai/whisper-small")
    ap.add_argument("--max-per-lang", type=int, default=20000)
    ap.add_argument("--morfessor", action="store_true",
                    help="compare unsupervised morph segmentation against BPE")
    ap.add_argument("--held-out", action="store_true",
                    help="with --morfessor, evaluate on unseen words")
    ap.add_argument("--extension-curve", action="store_true",
                    help="fertility vs number of tokens added to the vocabulary")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    by_lang = load_texts(args.corpus, args.max_per_lang)
    print(f"# {args.corpus} · {args.tokenizer}\n")
    fertility_table(by_lang, tok)

    if args.morfessor:
        morfessor_compare(by_lang, tok, args.held_out)
    if args.extension_curve:
        extension_curve(by_lang, tok, [1000, 2000, 4000, 8000, 16000, 32000])


if __name__ == "__main__":
    main()
