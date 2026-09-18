"""Upload finished bake-off runs to the Hub, after the fact.

run_bakeoff.sh trains without --push, so a preempted or failed run never
publishes half a model. Run this once the bake-off is done:

    venv/bin/python3 scripts/push_bakeoff.py --dry-run
    venv/bin/python3 scripts/push_bakeoff.py

Uses the same repo names and card as the trainers' own --push, so a model
pushed here and one pushed by a trainer land in the same repo. Everything
trained on PLD is tagged cc-by-nc-4.0: the corpus is research-only.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from halolib.finetune import push_model_to_hub  # noqa: E402

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs"))


def plan(run_dir: Path) -> dict | None:
    res_path, final = run_dir / "result.json", run_dir / "final"
    if not (res_path.exists() and final.is_dir()):
        return None
    res = json.loads(res_path.read_text())
    if res.get("dataset") != "pld" or not res.get("language"):
        return None
    lang, units = res["language"], res.get("units", "")
    metrics = {k.removeprefix("eval_"): v for k, v in res.items()
               if k in ("eval_cer", "eval_wer", "eval_loss") and v is not None}
    split = res.get("split", "unknown")
    common = dict(
        final_dir=final, base_model=res["encoder"], dataset_name="pld",
        task="asr", token=os.environ.get("HF_TOKEN"), metrics=metrics,
        license="cc-by-nc-4.0", lang_code=lang,
    )
    if run_dir.name.startswith("ctc_"):
        return common | dict(
            suffix=f"ctc-{units}-pld_{lang}",
            extra_tags=["philippines", "philippine-languages", "ctc", lang,
                        "bakeoff"],
            train_summary=(
                f"CTC head on {res['encoder']} ({units} units, "
                f"{res.get('vocab', '?')} of them) for {res['steps']} steps on "
                f"{res['train_rows']} clips, {split} split (speakers and "
                f"prompts unseen in training). Part of the R1/R2 bake-off in "
                f"docs/pld_sota_track.md. Trained on PLD, which is CC-BY-NC "
                f"and research-only: this checkpoint is a research artifact "
                f"regardless of the base model's licence."),
        )
    if run_dir.name.startswith("asr_"):
        return common | dict(
            suffix=f"pld-{lang}",
            extra_tags=["philippines", "philippine-languages", lang, "whisper",
                        "bakeoff"],
            train_summary=(
                f"Trained for {res['steps']} steps on {res['train_rows']} "
                f"clips, {split} split (speakers and prompts unseen in "
                f"training). The Whisper arm of the R1 bake-off in "
                f"docs/pld_sota_track.md. Trained on PLD, which is CC-BY-NC "
                f"and research-only."),
        )
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cer", type=float, default=0.8,
                    help="skip runs that did not learn (CER at or above this); "
                         "1.0 publishes everything")
    ap.add_argument("runs", nargs="*", help="run dir names; default: all")
    args = ap.parse_args()

    dirs = ([FINETUNE_DIR / r for r in args.runs] if args.runs
            else sorted(p for p in FINETUNE_DIR.iterdir() if p.is_dir()))
    todo = []
    for d in dirs:
        if not (p := plan(d)):
            continue
        cer = p["metrics"].get("cer")
        # a collapsed arm (CER ~1.0, WER 1.0) is a bake-off finding, not a
        # model anyone should download; it stays in the results table instead
        if cer is not None and cer >= args.max_cer:
            print(f"{d.name}: skipped, CER {cer:.3f} — did not learn")
            continue
        todo.append((d, p))
    if not todo:
        sys.exit(f"nothing finished to push in {FINETUNE_DIR}")
    for d, p in todo:
        name = f"{p['base_model'].split('/')[-1]}-{p['suffix']}"
        print(f"{d.name} -> {name}  {p['metrics']}")
        if not args.dry_run:
            push_model_to_hub(**p)


if __name__ == "__main__":
    main()
