"""
Compare a retrained ASR run against the model already on the Hub and push only
if it is better.

A longer/larger rerun is not automatically an improvement — more epochs over a
small language can overfit. This gates the upload on held-out CER (the
selection metric used during training) so a worse rerun can never overwrite a
better published model.

Usage:
  python compare_and_push_asr.py bcl --run-dir finetune_runs_v2
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

REPO_TMPL = "sapinsapin/whisper-small-pld-{lang}"


def best_metrics(run: Path) -> tuple[float, float] | None:
    """(cer, wer) of the best checkpoint, from the trainer's own history.

    The state file lives in the checkpoint dirs; the last one written holds
    the full eval history for the run.
    """
    states = sorted(run.glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    if not states:
        return None
    state = json.loads(states[-1].read_text())
    evals = [h for h in state.get("log_history", []) if "eval_cer" in h]
    if not evals:
        return None
    best = min(evals, key=lambda h: h["eval_cer"])
    return best["eval_cer"], best.get("eval_wer", float("nan"))


def hub_cer(repo_id: str, token: str | None) -> float | None:
    """Parse the CER recorded in the published model card's metrics table."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo_id, "README.md", token=token)
    except Exception:
        return None            # nothing published yet
    card = Path(p).read_text(encoding="utf-8")
    m = re.search(r"^\|\s*cer\s*\|\s*([0-9.]+)\s*\|", card, re.M | re.I)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("lang")
    ap.add_argument("--run-dir", default="finetune_runs_v2")
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--force", action="store_true",
                    help="push even if the rerun is not better")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    repo_id = REPO_TMPL.format(lang=args.lang)
    run = Path(args.run_dir) / f"asr_pld_{args.lang}"
    final = run / "final"

    new = best_metrics(run)
    if new is None:
        sys.exit(f"{args.lang}: no eval history in {run}")
    new_cer, new_wer = new
    old_cer = hub_cer(repo_id, token)

    # The published CER is rounded to 4dp in the card, and eval itself is
    # noisy, so require a real margin rather than letting rounding noise
    # overwrite an equally good model.
    MIN_REL_GAIN = 0.01
    verdict = ("no published model" if old_cer is None
               else f"published CER {old_cer:.4f}")
    better = old_cer is None or new_cer < old_cer * (1 - MIN_REL_GAIN)
    gain = "" if old_cer is None else f" ({(old_cer - new_cer) / old_cer:+.1%})"
    print(f"{args.lang}: new CER {new_cer:.4f} (WER {new_wer:.4f}) vs {verdict}"
          f"{gain} -> {'BETTER' if better else 'not better'}")

    if not (better or args.force):
        print(f"{args.lang}: keeping the published model, skipping push")
        return

    from halolib.finetune import push_model_to_hub
    push_model_to_hub(
        final, args.model, "pld", "asr", token=token,
        metrics={"cer": new_cer, "wer": new_wer},
        train_summary=(
            f"Extended run: trained to convergence on the {args.lang} portion "
            f"of PLD read speech, selected on held-out CER. WER/CER are "
            f"lowercased on the held-out split; CER is the model-selection "
            f"metric because Philippine-language orthography varies at the "
            f"word level."),
        suffix=f"pld-{args.lang}",
        lang_code=args.lang,
        extra_tags=["philippines", "philippine-languages", args.lang, "whisper"],
        license="apache-2.0",
    )


if __name__ == "__main__":
    main()
