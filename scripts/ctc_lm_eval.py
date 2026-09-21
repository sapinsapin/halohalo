"""
D0 and D1 of docs/asr_decoder_plan.md: read a CTC model's errors, then see what
an n-gram language model repairs. Inference only; sized for an 8 GB card.

  # GPU once: frame log-probs for the frozen test split, saved and reused
  python scripts/ctc_lm_eval.py logits --language ceb

  # D0, CPU: what kind of errors are they, and could an LM reach them?
  python scripts/ctc_lm_eval.py errors --language ceb

  # D1, CPU: beam search with a KenLM ARPA, alpha/beta tuned on a dev slice
  python scripts/ctc_lm_eval.py lm --language ceb --arpa lm/ceb.arpa

The logits are saved because everything after them is CPU work that gets
repeated — every alpha, every beta, every LM — and the encoder pass is the only
part that needs the GPU.

Scoring follows finetune_ctc.py: lowercase, whitespace-split words, the same
frozen speaker- and prompt-disjoint split. Every LM number is reported beside
the greedy one, so the acoustic model and the LM are never confused in a table.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs"))
OUT = FINETUNE_DIR / "ctc_lm_eval"
ORG = "sapinsapin"
SR = 16000
PAD, DELIM = "<pad>", "|"


def model_id(lang: str, size: str) -> str:
    return f"{ORG}/omniASR_W2V_{size}_SSL-ctc-char-pld_{lang}"


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def stage_logits(args):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

    from finetune_ctc import ctc_decode
    from halolib.finetune import load_speech_dataset

    mid = model_id(args.language, args.size)
    tok = os.environ.get("HF_TOKEN")
    vocab = json.loads(Path(hf_hub_download(mid, "vocab.json", token=tok)).read_text())
    id2unit = {i: u for u, i in vocab.items()}

    ds = load_speech_dataset("pld", task="asr", language=args.language,
                             max_samples=None, token=tok)
    test, train = ds["test"], ds["train"]
    # the LM, the lexicon and the near-miss check may only ever see train text
    (OUT / f"{args.language}_train_text.txt").write_text(
        "\n".join(norm(t) for t in train["text"]), encoding="utf-8")

    extractor = AutoFeatureExtractor.from_pretrained(mid, token=tok)
    model = Wav2Vec2ForCTC.from_pretrained(mid, token=tok, dtype=torch.float16)
    model = model.to("cuda").eval()

    logps, refs, greedy = [], [], []
    for i in range(len(test)):
        row = test[i]
        x = extractor(row["audio"]["array"], sampling_rate=SR, return_tensors="pt")
        with torch.inference_mode():
            lp = model(x.input_values.to("cuda", torch.float16)).logits[0] \
                .float().log_softmax(-1).cpu().numpy()
        logps.append(lp.astype(np.float16))
        refs.append(norm(row["text"]))
        greedy.append(ctc_decode(lp.argmax(-1), id2unit))
        if i % 100 == 0:
            print(f"  {i}/{len(test)}", flush=True)

    # ragged: one (frames, vocab) array per clip. np.array() on such a list
    # tries to broadcast and fails when two clips share a length.
    ragged = np.empty(len(logps), dtype=object)
    for i, lp in enumerate(logps):
        ragged[i] = lp
    np.savez_compressed(OUT / f"{args.language}_logits.npz", logps=ragged)
    (OUT / f"{args.language}_meta.json").write_text(json.dumps(
        {"model": mid, "vocab": vocab, "refs": refs, "greedy": greedy},
        ensure_ascii=False))
    import jiwer
    print(f"{mid}: greedy CER {jiwer.cer(refs, greedy) * 100:.2f}  "
          f"WER {jiwer.wer(refs, greedy) * 100:.2f}  ({len(refs)} clips)")


def stage_errors(args):
    """D0. The number that decides D1: of the words the model got wrong, how
    many were a near-miss of the right word, and was the right word one the
    training text contains — i.e. could a lexicon or LM have reached it?"""
    import jiwer

    meta = json.loads((OUT / f"{args.language}_meta.json").read_text())
    refs, hyps = meta["refs"], meta["greedy"]
    train_vocab = set((OUT / f"{args.language}_train_text.txt")
                      .read_text(encoding="utf-8").split())

    out = jiwer.process_words(refs, hyps)
    kinds = Counter()
    sub_dist = Counter()
    reachable = 0
    examples = []
    for ref, hyp, chunks in zip(out.references, out.hypotheses, out.alignments):
        for c in chunks:
            if c.type == "equal":
                continue
            n = max(c.ref_end_idx - c.ref_start_idx, c.hyp_end_idx - c.hyp_start_idx)
            kinds[c.type] += n
            if c.type != "substitute":
                continue
            for r, h in zip(ref[c.ref_start_idx:c.ref_end_idx],
                            hyp[c.hyp_start_idx:c.hyp_end_idx]):
                d = edit_distance(r, h)
                sub_dist[min(d, 4)] += 1
                if d <= 2 and r in train_vocab:
                    reachable += 1
                if len(examples) < 12 and d <= 2:
                    examples.append(f"{r} -> {h}")

    total = sum(kinds.values())
    subs = kinds["substitute"]
    print(f"\n{meta['model']}\n{len(refs)} clips, {total} word errors")
    for k in ("substitute", "delete", "insert"):
        print(f"  {k:11s} {kinds[k]:5d}  {kinds[k] / max(total, 1) * 100:5.1f}%")
    print("\nsubstituted words, by character edit distance from the right word:")
    for d in (1, 2, 3, 4):
        lab = f"{d}" if d < 4 else "4+"
        print(f"  {lab:3s} {sub_dist[d]:5d}  {sub_dist[d] / max(subs, 1) * 100:5.1f}%")
    near = sub_dist[1] + sub_dist[2]
    print(f"\nnear-misses (distance <= 2): {near} of {subs} substitutions, "
          f"{near / max(total, 1) * 100:.1f}% of all word errors")
    print(f"  ... of which the right word is in the training text: {reachable} "
          f"({reachable / max(total, 1) * 100:.1f}% of all word errors)")
    print("  That last figure is the ceiling on what a train-text LM can repair.")
    oov = sum(1 for r in refs for w in r.split() if w not in train_vocab)
    nw = sum(len(r.split()) for r in refs)
    print(f"\ntest words never seen in training text: {oov} of {nw} "
          f"({oov / nw * 100:.1f}%) — a closed lexicon would get all of these wrong")
    print("\nexamples:", "; ".join(examples))


def stage_lm(args):
    """D1. Beam search fused with a KenLM ARPA. alpha/beta are tuned on the
    first `--dev` clips and the result is reported on the rest, so the tuning
    never sees the clips it is scored on."""
    import jiwer
    from pyctcdecode import build_ctcdecoder

    meta = json.loads((OUT / f"{args.language}_meta.json").read_text())
    logps = np.load(OUT / f"{args.language}_logits.npz", allow_pickle=True)["logps"]
    refs, greedy = meta["refs"], meta["greedy"]
    vocab = meta["vocab"]
    # pyctcdecode's conventions: "" is the blank, " " the word delimiter
    labels = [""] * len(vocab)
    for u, i in vocab.items():
        labels[i] = "" if u == PAD else " " if u == DELIM else u
    unigrams = sorted(set((OUT / f"{args.language}_train_text.txt")
                          .read_text(encoding="utf-8").split()))

    dev, test = slice(0, args.dev), slice(args.dev, None)

    def run(alpha, beta, part):
        dec = build_ctcdecoder(labels, kenlm_model_path=args.arpa,
                               unigrams=unigrams, alpha=alpha, beta=beta)
        hyps = [norm(dec.decode(np.asarray(lp, dtype=np.float32),
                                beam_width=args.beam)) for lp in logps[part]]
        return jiwer.cer(refs[part], hyps), jiwer.wer(refs[part], hyps)

    best = None
    for alpha in (0.3, 0.5, 0.8, 1.2):
        for beta in (0.0, 1.0, 2.0):
            cer, wer = run(alpha, beta, dev)
            print(f"  dev alpha={alpha} beta={beta}: CER {cer * 100:.2f} WER {wer * 100:.2f}",
                  flush=True)
            if best is None or wer < best[0]:
                best = (wer, alpha, beta)
    _, alpha, beta = best

    g_cer, g_wer = jiwer.cer(refs[test], greedy[test]), jiwer.wer(refs[test], greedy[test])
    l_cer, l_wer = run(alpha, beta, test)
    print(f"\n{meta['model']}  ({len(refs[test])} held-back clips; "
          f"alpha={alpha} beta={beta} tuned on the other {args.dev})")
    print(f"  greedy      CER {g_cer * 100:6.2f}  WER {g_wer * 100:6.2f}")
    print(f"  + {Path(args.arpa).name:9s} CER {l_cer * 100:6.2f}  WER {l_wer * 100:6.2f}")
    (OUT / f"{args.language}_lm_result.json").write_text(json.dumps({
        "model": meta["model"], "arpa": args.arpa, "alpha": alpha, "beta": beta,
        "greedy": {"cer": g_cer, "wer": g_wer}, "lm": {"cer": l_cer, "wer": l_wer},
        "n_test": len(refs[test]), "n_dev": args.dev}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stage", choices=["logits", "errors", "lm"])
    ap.add_argument("--language", required=True)
    ap.add_argument("--size", default="1B", help="1B fits an 8 GB card; 7B does not")
    ap.add_argument("--arpa", default=None)
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--dev", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"logits": stage_logits, "errors": stage_errors, "lm": stage_lm}[args.stage](args)


if __name__ == "__main__":
    main()
