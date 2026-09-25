"""
Meta's MMS-1b-all, zero-shot, on the frozen PLD test splits — plan item A2's
baseline row.

Every ASR number in this project so far comes from a model we trained on PLD.
MMS-1b-all was not: it is Meta's 1,000-language CTC model with a small adapter
per language, and it has adapters for nine of our ten languages (Filipino as
`tgl`). Its score on the same frozen speaker- and prompt-disjoint test sets is
the "what you get without training anything" row every table has been missing.

Scored two ways, both reported: as the reference is written, and normalised
(halolib.finetune.normalise_text: lowercase, no stress accents, no
punctuation). MMS never writes PLD's stress accents, so the normalised number
is the fair one and the other is there for comparison with older tables.
Whole test split, digits included, as eval_published_fleet.py does.

Inference only; the 1B model in fp16 is ~2 GB and fits the workstation's 3070.

  python scripts/eval_mms_zeroshot.py                     # all ten
  python scripts/eval_mms_zeroshot.py --languages ceb --max-clips 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from halolib.finetune import load_speech_dataset, normalise_text  # noqa: E402

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs"))
SR = 16000
LANGS = ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]
MMS_CODES = {"bcl": "bcl", "ceb": "ceb", "eng": "eng", "fil": "tgl", "hil": "hil",
             "ilo": "ilo", "pag": "pag", "pam": "pam", "tsg": "tsg", "war": "war"}


def score(refs, hyps):
    import jiwer
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return {"cer": None, "wer": None, "n": 0}
    r, h = map(list, zip(*pairs))
    return {"cer": jiwer.cer(r, h), "wer": jiwer.wer(r, h), "n": len(pairs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--languages", nargs="*", default=LANGS)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = whole test split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    out_dir = FINETUNE_DIR / "mms_zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("results.json" if not args.max_clips else "results_smoke.json")
    results = json.loads(path.read_text()) if path.exists() else {}

    dt = torch.float16 if args.device == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", dtype=dt).to(args.device).eval()

    for lang in args.languages:
        if lang in results:
            print(f"  {lang}: done already")
            continue
        code = MMS_CODES[lang]
        try:
            proc.tokenizer.set_target_lang(code)
            model.load_adapter(code)
        except Exception as e:                            # noqa: BLE001
            print(f"  {lang}: no MMS adapter for {code} ({type(e).__name__}); skipped")
            results[lang] = {"skipped": f"no adapter for {code}"}
            path.write_text(json.dumps(results, indent=1))
            continue

        t0 = time.perf_counter()
        ds = load_speech_dataset("pld", task="asr", language=lang, max_samples=None,
                                 token=os.environ.get("HF_TOKEN"))
        test = ds["test"]
        n = min(len(test), args.max_clips) if args.max_clips else len(test)
        print(f"  {lang} [{code}]: {n} test clips (loaded in {time.perf_counter() - t0:.0f}s)",
              flush=True)

        refs, hyps = [], []
        t1 = time.perf_counter()
        for i in range(n):
            row = test[i]
            x = proc(row["audio"]["array"], sampling_rate=SR, return_tensors="pt").input_values
            with torch.no_grad():
                ids = model(x.to(args.device, dt)).logits.argmax(-1)[0]
            hyps.append(proc.decode(ids))
            refs.append(row["text"])

        raw = score([" ".join(r.lower().split()) for r in refs],
                    [" ".join(h.lower().split()) for h in hyps])
        norm = score([normalise_text(r) for r in refs], [normalise_text(h) for h in hyps])
        results[lang] = {"mms_code": code, "as_scored": raw, "normalised": norm,
                         "seconds": round(time.perf_counter() - t1, 1),
                         "refs": refs, "hyps": hyps}
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"  {lang}: as scored CER {raw['cer'] * 100:.2f} WER {raw['wer'] * 100:.2f} | "
              f"normalised CER {norm['cer'] * 100:.2f} WER {norm['wer'] * 100:.2f}", flush=True)

    print("\n| language | CER (norm) | WER (norm) | CER (as scored) | clips |")
    print("|---|---|---|---|---|")
    for lang in args.languages:
        r = results.get(lang, {})
        if "normalised" in r:
            print(f"| {lang} | {r['normalised']['cer'] * 100:.1f} | "
                  f"{r['normalised']['wer'] * 100:.1f} | {r['as_scored']['cer'] * 100:.1f} | "
                  f"{r['normalised']['n']} |")
        elif r:
            print(f"| {lang} | — | — | — | {r.get('skipped')} |")
    print(f"\nFrozen speaker- and prompt-disjoint test splits; zero-shot. Saved: {path}")


if __name__ == "__main__":
    main()
