"""
Re-measure the published ASR fleet on the frozen speaker- and prompt-disjoint
splits — plan item A7, and the §6.1 reassessment.

Every CER on docs/cards/pld.md and in the plan's comparison tables comes from
the *overlapping* random split, where the same speakers and the same prompts
appear in train and test. Those numbers are not wrong arithmetic; they answer a
different and much easier question than "how does this model do on a speaker it
has never heard saying a sentence it has never seen". The R1 bake-off measured
the new models honestly. This measures the fleet the same way, so the two can
sit in one table.

**This does not produce the honest column A7 asked for, and 2026-09-19 is
when we found out.** The published checkpoints trained on the hub's random
split, which overlaps the frozen test set: on ceb, 20.3% of their training
clips come from the frozen split's held-out speakers and prompts. Their
scores here are therefore on speakers they have heard. The honest column
needs scripts/retrain_fleet_frozen.sh, which retrains on the frozen splits.

Inference only — no training, no GPU-hours beyond generation. Runs happily on a
small card: whisper-small is 244M and fp16 at batch 8 needs about 2 GB.

  python scripts/eval_published_fleet.py --languages ceb
  python scripts/eval_published_fleet.py --device cuda --batch-size 8

Results: $FINETUNE_DIR/published_fleet_eval/results.json plus a markdown table.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ORG = "sapinsapin"
SR = 16000
LANGS = ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]


def score(refs: list[str], hyps: list[str]) -> dict:
    """CER and WER under the dataset cards' convention: strip and lowercase,
    nothing else. Pairs with an empty reference are dropped rather than scored
    as total failure."""
    import jiwer

    pairs = [(r.strip().lower(), h.strip().lower())
             for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return {"cer": 1.0, "wer": 1.0, "n": 0}
    r, h = zip(*pairs)
    return {"cer": jiwer.cer(list(r), list(h)),
            "wer": jiwer.wer(list(r), list(h)), "n": len(pairs)}


def evaluate(lang: str, device: str, batch_size: int, max_clips: int,
             model_template: str) -> dict:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from halolib.finetune import load_speech_dataset

    model_id = model_template.format(lang=lang)
    ds = load_speech_dataset("pld", task="asr", language=lang,
                             max_samples=max_clips,
                             token=os.environ.get("HF_TOKEN"))
    test = ds["test"]

    proc = WhisperProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype).to(device).eval()

    refs, hyps = [], []
    t0 = time.perf_counter()
    for i in range(0, len(test), batch_size):
        rows = test[i:i + batch_size]
        feats = proc.feature_extractor(
            [a["array"] for a in rows["audio"]], sampling_rate=SR,
            return_tensors="pt").input_features.to(device, dtype)
        with torch.inference_mode():
            ids = model.generate(feats, task="transcribe", max_new_tokens=200)
        hyps += proc.batch_decode(ids, skip_special_tokens=True)
        refs += rows["text"]
        if i and i % (batch_size * 20) == 0:
            print(f"    {i}/{len(test)}", flush=True)

    out = score(refs, hyps)
    # kept so a result can be re-scored under another text normalisation
    # without another pass through the model
    out.update(refs=list(refs), hyps=list(hyps))
    out.update(model=model_id, language=lang,
               seconds=round(time.perf_counter() - t0, 1))
    print(f"  {lang}: CER {out['cer']*100:.2f}%  WER {out['wer']*100:.2f}%  "
          f"({out['n']} clips, {out['seconds']}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--languages", nargs="*", default=LANGS)
    ap.add_argument("--model-template", default=f"{ORG}/whisper-small-pld-{{lang}}")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="published_fleet_eval",
                    help="results dir under FINETUNE_DIR; one per model family")
    ap.add_argument("--max-clips", type=int, default=0,
                    help="cap the test split (0 = all of it)")
    args = ap.parse_args()

    out_dir = FINETUNE_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    # resumable: a card-by-card re-measurement should survive an interrupted run
    results = (json.loads(results_path.read_text())
               if results_path.exists() else {})

    for lang in args.languages:
        if lang in results:
            print(f"  {lang}: already measured, skipping")
            continue
        try:
            results[lang] = evaluate(lang, args.device, args.batch_size,
                                     args.max_clips or None,
                                     args.model_template)
        except Exception as e:                        # noqa: BLE001
            print(f"  {lang}: FAILED — {type(e).__name__}: {e}")
            continue
        results_path.write_text(json.dumps(results, indent=2))

    print(f"\n| language | CER% | WER% | clips |")
    print("|---|---|---|---|")
    for lang in args.languages:
        r = results.get(lang)
        if r:
            print(f"| {lang} | {r['cer']*100:.2f} | {r['wer']*100:.2f} | {r['n']} |")
    print()
    print("READ THIS BEFORE QUOTING THE TABLE. These checkpoints were")
    print("trained before the splits were frozen, on the hub's random")
    print("split. Measured on ceb, 20.3% of their training clips belong to")
    print("the speakers and prompts the frozen split holds out: 10,042")
    print("clips from the 28 frozen test speakers. So this is not a")
    print("held-out score, it is a score on speakers the model has heard.")
    print("For an honest fleet column, retrain on the frozen train splits:")
    print("  bash scripts/retrain_fleet_frozen.sh")
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
