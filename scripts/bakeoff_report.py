"""
Bake-off decision rule, made executable (plan R1; review finding F63 — the
bake-off had no decision rule, so "whichever wins" could not be adjudicated).

The rule, fixed before the runs:

  * primary metric is CER on the frozen speaker- AND prompt-disjoint split,
    under one normaliser;
  * two winners are declared — **best overall** and **best permissive** —
    because the commercial track may only build on permissive components;
  * a permissive candidate within MARGIN relative of the overall best takes
    the crown outright, since a licence advantage is worth more to the
    programme than a small CER edge.

    python scripts/bakeoff_report.py
    python scripts/bakeoff_report.py --margin 0.05 --markdown
"""

import argparse
import json
import os
from pathlib import Path

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
MARGIN = 0.05          # 5 % relative

# Licence of each encoder family, from the Hub on 2026-09-15. This is what
# decides whether a winner can cross into the commercial track at all.
LICENCE = {
    "ylacombe/omniASR_W2V_300M_SSL": ("Apache-2.0", True),
    "ylacombe/omniASR_W2V_1B_SSL": ("Apache-2.0", True),
    "ylacombe/omniASR_W2V_3B_SSL": ("Apache-2.0", True),
    "ylacombe/omniASR_W2V_7B_SSL": ("Apache-2.0", True),
    "facebook/w2v-bert-2.0": ("MIT", True),
    "openai/whisper-large-v3": ("Apache-2.0", True),
    "openai/whisper-small": ("Apache-2.0", True),
    "facebook/seamless-m4t-v2-large": ("CC-BY-NC", False),
    "facebook/mms-1b-all": ("CC-BY-NC", False),
}


def collect():
    """CTC runs write result.json; seq2seq runs leave trainer_state.json."""
    rows = []
    # Both trainers now write result.json with the model and split recorded.
    for p in sorted(list(FINETUNE_DIR.glob("ctc_*/result.json"))
                    + list(FINETUNE_DIR.glob("asr_*/result.json"))):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "run": p.parent.name,
            "encoder": r.get("encoder", "?"),
            "units": r.get("units", "?"),
            "language": r.get("language"),
            "split": r.get("split", "unknown"),
            "cer": r.get("eval_cer"),
            "wer": r.get("eval_wer"),
            "vocab": r.get("vocab"),
            "rows": r.get("train_rows"),
        })
    for p in sorted(FINETUNE_DIR.glob("asr_*/final/trainer_state.json")):
        if (p.parent.parent / "result.json").exists():
            continue            # already read above, with its model and split
        st = json.loads(p.read_text(encoding="utf-8"))
        best = None
        for h in reversed(st.get("log_history", [])):
            if "eval_cer" in h:
                best = h
                break
        if not best:
            continue
        run = p.parent.parent.name
        rows.append({
            "run": run,
            "encoder": "openai/whisper-large-v3" if "lv3" in run else "openai/whisper-small",
            "units": "bpe",
            "language": run.split("_")[-1],
            "split": "unknown",          # seq2seq runs do not record it
            "cer": best.get("eval_cer"),
            "wer": best.get("eval_wer"),
            "vocab": None,
            "rows": None,
        })
    return [r for r in rows if r["cer"] is not None]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows = collect()
    if not rows:
        print(f"no finished runs under {FINETUNE_DIR}")
        return

    for r in rows:
        lic, perm = LICENCE.get(r["encoder"], ("unknown", False))
        r["licence"], r["permissive"] = lic, perm

    langs = sorted({r["language"] for r in rows}, key=lambda x: (x is None, x))
    for lang in langs:
        sub = sorted([r for r in rows if r["language"] == lang],
                     key=lambda r: r["cer"])
        # Report the split that actually produced the numbers rather than
        # asserting the one we wish had been used.
        kinds = ", ".join(sorted({r["split"] for r in sub}))
        print(f"\n## {lang or '(no language filter)'}  [split: {kinds}]\n")
        if args.markdown:
            print("| candidate | units | licence | CER % | WER % | vocab |")
            print("|---|---|---|---|---|---|")
            for r in sub:
                print(f"| {r['encoder'].split('/')[-1]} | {r['units']} | "
                      f"{r['licence']} | {r['cer'] * 100:.2f} | "
                      f"{r['wer'] * 100:.2f} | {r['vocab'] or '-'} |")
        else:
            print(f"{'candidate':34s} {'units':9s} {'licence':10s} "
                  f"{'CER%':>7} {'WER%':>7} {'vocab':>6}")
            for r in sub:
                print(f"{r['encoder'].split('/')[-1]:34s} {r['units']:9s} "
                      f"{r['licence']:10s} {r['cer'] * 100:7.2f} "
                      f"{r['wer'] * 100:7.2f} {str(r['vocab'] or '-'):>6}")

        # A CER above 1.0 means the head emitted more characters than the
        # reference held — a diverged or barely-trained run. Rank only
        # converged candidates, or a broken run gets crowned by default.
        converged = [r for r in sub if r["cer"] is not None and r["cer"] <= 1.0]
        if not converged:
            print(f"\n  no converged run for {lang or 'this set'} "
                  f"(every CER above 100 %) — nothing to rank")
            continue
        if len(converged) < len(sub):
            print(f"\n  ({len(sub) - len(converged)} run(s) excluded: CER above 100 %)")
        best = converged[0]
        perm = [r for r in converged if r["permissive"]]
        best_perm = perm[0] if perm else None
        print(f"\n  best overall : {best['encoder'].split('/')[-1]} "
              f"({best['units']}, {best['licence']}) CER {best['cer'] * 100:.2f} %")
        if best_perm:
            gap = (best_perm["cer"] - best["cer"]) / max(best["cer"], 1e-9)
            print(f"  best permissive: {best_perm['encoder'].split('/')[-1]} "
                  f"({best_perm['units']}, {best_perm['licence']}) "
                  f"CER {best_perm['cer'] * 100:.2f} %  "
                  f"[{gap * 100:+.1f} % relative]")
            if best_perm["run"] == best["run"]:
                print("  -> the winner is permissive; both tracks use it")
            elif gap <= args.margin:
                print(f"  -> within the {args.margin * 100:.0f} % margin: "
                      f"the permissive candidate takes the crown for both tracks")
            else:
                print(f"  -> outside the {args.margin * 100:.0f} % margin: "
                      f"research track uses the overall winner, commercial "
                      f"track uses the permissive one, and both are reported")
        else:
            print("  no permissive candidate finished — commercial track blocked")

    # unit ablation (R2) read across languages
    print("\n## output units, averaged over languages\n")
    by_unit = {}
    for r in rows:
        by_unit.setdefault(r["units"], []).append(r["cer"])
    for u, cs in sorted(by_unit.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {u:9s} mean CER {sum(cs) / len(cs) * 100:6.2f} %  (n={len(cs)})")
    # Derive the caveat from the data rather than asserting it: a run that
    # fell back to the overlapping split must not be presented as disjoint.
    kinds = sorted({r["split"] for r in rows})
    if kinds == ["frozen-disjoint"]:
        print("\nReminder: speaker- and prompt-disjoint numbers. They are not "
              "comparable to the in-domain CERs on the dataset card, which "
              "share both speakers and prompts between train and test.")
    else:
        print(f"\nCaution: split provenance is {kinds}. Only runs marked "
              f"'frozen-disjoint' are speaker- and prompt-disjoint; the rest "
              f"are in-domain numbers and must be labelled that way.")


if __name__ == "__main__":
    main()
