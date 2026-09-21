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

# Published for the record rather than for use: a negative result is only
# useful to the next person if they can find it.
SCALE_NOTE = {"omniASR_W2V_7B_SSL": (
    " **Scale did not help.** On Cebuano this 7B encoder scores 17.02% CER, "
    "the 1B sibling 17.04% and whisper-large-v3 16.38%, same split, same 5000 "
    "steps. Trained with fp32 weights and bitsandbytes 8-bit Adam to fit one "
    "96 GB card. Kept in case a better decoder than a linear CTC head can use "
    "what the encoder knows.")}

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs"))


ORPHEUS_BASE = "unsloth/orpheus-3b-0.1-pretrained"


def plan_orpheus(run_dir: Path) -> dict | None:
    """An Orpheus TTS adapter. These runs write no result.json: a codec LM's
    eval loss says little, so what is known about them is what tts_eval.py
    measured — round-trip CER through an ASR judge, on 50 frozen sentences."""
    final = run_dir / "final"
    parts = run_dir.name.split("_")          # orpheus, <units>, pld, <lang>
    ev_path = FINETUNE_DIR / "tts_eval" / "results.json"
    if len(parts) != 4 or not final.is_dir() or not ev_path.exists():
        return None
    units, lang = parts[1], parts[3]
    ev = json.loads(ev_path.read_text())
    mine = ev.get(run_dir.name, {}).get(lang)
    if not mine:
        return None
    human = ev.get("reference", {}).get(lang, {}).get("cer")
    mms = ev.get("mms", {}).get(lang, {}).get("cer")
    versus = (f"MMS-TTS scores {mms * 100:.1f}% on the same sentences and judge"
              if mms is not None else
              "MMS-TTS has no model for this language, so there is no bar to compare with")
    return dict(
        final_dir=final, base_model=ORPHEUS_BASE, dataset_name="pld",
        task="tts", token=os.environ.get("HF_TOKEN"),
        metrics={"cer": mine["cer"], "wer": mine["wer"],
                 "spk_sim": mine["spk_sim"]},
        license="cc-by-nc-4.0", lang_code=lang,
        suffix=f"{units}-pld-{lang}",
        extra_tags=["philippines", "philippine-languages", "orpheus", "lora",
                    "snac", lang],
        train_summary=(
            f"LoRA (r=64) on {ORPHEUS_BASE} in bf16, 2000 steps, speaker- and "
            f"prompt-disjoint split. **Text frontend: {units}.** The model "
            f"must be prompted the way it was trained — "
            f"`finetune_orpheus.frontend_text(text, '{units}')` in "
            f"github.com/sapinsapin/halohalo — or it is reading a notation it "
            f"never saw. Characters won the frontend ablation in "
            f"docs/tts_sota_plan.md over BPE and syllables.\n\n"
            f"The metrics are round-trip: 50 frozen sentences synthesized, "
            f"transcribed by an ASR judge, scored against the text. The human "
            f"recordings of those sentences score "
            f"{human * 100:.1f}% CER through the same judge, which is the floor; "
            f"{versus}. Caveats that travel with these numbers: 50 sentences "
            f"resolves a 3-point gap, not a 1-point one; and the judge is our "
            f"own model trained on the same corpus, so an independent judge is "
            f"still owed before anyone cites this.\n\n"
            f"Trained on PLD, which is CC-BY-NC and research-only. The base "
            f"model carries Llama 3.2 terms of its own."),
    )


def plan(run_dir: Path) -> dict | None:
    if run_dir.name.startswith("orpheus_"):
        return plan_orpheus(run_dir)
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
                f"regardless of the base model's licence."
                + SCALE_NOTE.get(res["encoder"].split("/")[-1], "")),
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
