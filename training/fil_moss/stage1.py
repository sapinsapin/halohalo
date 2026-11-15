"""Stage 1, a trustworthy SFT of MOSS-TTS-Realtime on Filipino (GPU, dwnd-01).

    python -m training.fil_moss.guard --max-seconds 3600 -- \
        python -m training.fil_moss.stage1 verify --codes /abs/run/codes --out /abs/run/stage1-verify
    python -m training.fil_moss.guard --max-seconds 14400 -- \
        python -m training.fil_moss.stage1 train --codes /abs/run/codes \
            --verify-report /abs/run/stage1-verify/verify_report.json --out /abs/run/stage1 \
            [--english /abs/english_regression.jsonl]

``verify`` (S1, all three must pass before ``train`` will start):
  1. parity: our per-token log-probabilities, gathered from ``local_logits``, reproduce upstream's loss on a
     frozen batch. Stage 2 optimises those log-probabilities, so this is the check that they are the real ones.
  2. overfit: 32 training clips, 200 steps at a high learning rate; loss must collapse and the overfit model
     must speak those texts at the reference pace (duration ratio 1 +/- 0.05). This catches the off-by-one /
     never-stops family of bugs that broke our Qwen3-TTS finetune.
  3. save/reload: the saved checkpoint reloads to identical weights and an identical loss.

``train`` (S2): full finetune from the pinned pretrained model, FP32 weights, BF16 autocast, AdamW8bit
(the configuration the Hiligaynon pilot measured), ``language_model.embed_tokens`` frozen as upstream does.
Evaluates on validation at step 0 (zero-shot) and every ``eval_every_updates``; stops on the duration kill or
on ending worse than zero-shot; keeps best (validation worst-judge CER) and last checkpoints; judges success
on the best checkpoint. Thresholds: ``recipe.json`` ``stage1``.
"""
import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import get_scheduler

from . import evaluation as E
from . import gates as G
from . import judges as J
from . import runtime as R

HERE = Path(__file__).resolve().parent


def load_recipe(path):
    return json.loads(Path(path).read_text())


def load_codes(codes_dir, split):
    rows = R.read_jsonl(Path(codes_dir) / ("%s_codes.jsonl" % split))
    if split == "test":
        raise SystemExit("stage 1 never reads the test split")
    return rows


def step(model, params, optimizer, scheduler, inputs, clip):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**inputs).loss
    if not torch.isfinite(loss):
        raise AssertionError("non-finite training loss")
    loss.backward()
    if not all(p.grad is None or p.grad.dtype == torch.float32 for p in params):
        raise AssertionError("gradients must be FP32")
    norm = torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return float(loss.detach()), float(norm)


# -------------------------------------------------------------------------------------------------- verify

def verify(a):
    recipe = load_recipe(a.recipe)
    v, t, gen = recipe["stage1"]["verify"], recipe["stage1"]["train"], recipe["generation"]
    out = R.new_output_dir(a.out)
    R.setup()
    R.seed_all(recipe["baseline"]["seed"])
    proc = R.processor()
    train = [r for r in load_codes(a.codes, "train") if r.get("ref_audio_codes") is not None]
    if len(train) < v["overfit_clips"]:
        raise SystemExit("need %d training clips with a reference voice, have %d" % (v["overfit_clips"], len(train)))
    clips = E.generation_subset(train, v["overfit_clips"], recipe["baseline"]["seed"])
    ds = R.dataset(clips, proc)
    report = {"recipe_sha256": _sha(a.recipe), "codes": a.codes, "overfit_ids": [r["id"] for r in clips]}

    model = R.load_model(R.MODEL_DIR)
    params = R.freeze(model, t["frozen"])
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mine, theirs = R.wrapper_loss(model, R.batch(ds, clips[0]))
    report["parity"] = G.parity_check(float(mine), float(theirs), v["parity_rtol"])
    print(json.dumps({"parity": report["parity"]}), flush=True)
    model.train()

    initial = E.teacher_forced_loss(model, ds, clips)
    optimizer = R.adamw8bit(params, v["overfit_learning_rate"], t)
    losses = []
    for i in range(v["overfit_steps"]):
        loss, _ = step(model, params, optimizer, None, R.batch(ds, clips[i % len(clips)]), t["grad_clip"])
        losses.append(loss)
    final = E.teacher_forced_loss(model, ds, clips)
    codec = R.load_codec()
    eng = R.engine(model, codec)
    ratios = []
    model.eval()
    for r in clips:
        R.seed_all(recipe["baseline"]["seed"])
        (codes, _), = R.generate(eng, [r["conversations"][0]["text"]], [r["ref_audio_codes"]],
                                 R.max_frames(r["seconds"], gen), gen)
        gen16 = J.resample(R.decode(codec, codes), R.SAMPLE_RATE)
        real, sr = sf.read(r["conversations"][0]["wav"], dtype="float32")
        ref_secs = J.trimmed_seconds(J.resample(real, sr), J.JUDGE_SR)
        ratios.append(J.trimmed_seconds(gen16, J.JUDGE_SR) / ref_secs)
    report["overfit"] = G.overfit_check(initial, final, ratios, v)
    report["overfit"]["train_losses"] = losses
    report["overfit"]["duration_ratios"] = ratios
    print(json.dumps({"overfit": {k: x for k, x in report["overfit"].items() if k not in ("train_losses",)}}),
          flush=True)

    ckpt = out / "overfit_checkpoint"
    R.save(model, proc, ckpt)
    probe = {n: p.detach()[:8].cpu().clone() for n, p in list(model.named_parameters())[:4]}
    del eng, optimizer
    model.cpu()
    del model
    torch.cuda.empty_cache()
    reloaded = R.load_model(ckpt, train=False)
    same = all(torch.equal(probe[n], p.detach()[:8].cpu()) for n, p in list(reloaded.named_parameters())[:4])
    again = E.teacher_forced_loss(reloaded, ds, clips)
    report["reload"] = {"weights_equal": same, "loss_before": final, "loss_after": again,
                        "loss_equal": abs(again - final) <= 1e-6 * max(1.0, abs(final))}
    report["verification"] = G.verification_verdict(report["parity"], report["overfit"],
                                                    same and report["reload"]["loss_equal"])
    R.write_json(out / "verify_report.json", report)
    print("STAGE1_VERIFY", report["verification"]["verdict"], flush=True)
    return 0 if report["verification"]["verdict"] == G.PASS else 3


# --------------------------------------------------------------------------------------------------- train

def train(a):
    recipe = load_recipe(a.recipe)
    t, gen, seed = recipe["stage1"]["train"], recipe["generation"], recipe["baseline"]["seed"]
    verified = json.loads(Path(a.verify_report).read_text())
    if verified.get("verification", {}).get("verdict") != G.PASS:
        raise SystemExit("stage 1 verification has not passed; refusing to train")
    if verified.get("codes") != a.codes:
        raise SystemExit("verification ran on different codes (%s)" % verified.get("codes"))
    out = R.new_output_dir(a.out)
    R.setup()
    R.seed_all(seed)
    proc = R.processor()
    train_rows, val_rows = load_codes(a.codes, "train"), load_codes(a.codes, "validation")
    if {r["id"] for r in train_rows} & {r["id"] for r in val_rows}:
        raise SystemExit("train/validation id overlap")
    train_ds, val_ds = R.dataset(train_rows, proc), R.dataset(val_rows, proc)
    subset = E.generation_subset(val_rows, t["eval_generation_clips"], seed)
    english = R.read_jsonl(a.english) if a.english else None

    model = R.load_model(R.MODEL_DIR)
    params = R.freeze(model, t["frozen"])
    optimizer = R.adamw8bit(params, t["learning_rate"], t)
    per_epoch = len(train_rows)
    total = per_epoch * t["epochs"]
    scheduler = get_scheduler(t["schedule"], optimizer, num_warmup_steps=t["warmup_steps"],
                              num_training_steps=total)
    codec = R.load_codec()
    eng = R.engine(model, codec)
    fil_judges, fil_failed = J.load(recipe["judges"]["fil"])
    eng_judges, eng_failed = J.load(recipe["judges"]["eng"]) if english else ({}, {})
    speaker = J.Speaker(recipe["judges"]["speaker"])
    report = {"recipe_sha256": _sha(a.recipe), "codes": a.codes, "verification": verified["verification"],
              "train_rows": len(train_rows), "validation_rows": len(val_rows), "updates": total,
              "judges": sorted(fil_judges), "judges_failed": fil_failed, "english_judges_failed": eng_failed,
              "evals": [], "steps": [], "best": None, "best_checkpoint": None, "status": "running"}

    def evaluate(update):
        m = E.generate_and_score(model, eng, subset, fil_judges, speaker, gen, t["eval_seeds"])
        m["validation_loss"] = E.teacher_forced_loss(model, val_ds, val_rows)
        if english:
            m.update(E.english_regression(model, eng, english, eng_judges, gen, t["eval_seeds"][0]))
        m["update"], m["epoch_fraction"] = update, update / per_epoch
        report["evals"].append(m)
        print(json.dumps({"eval": {k: v for k, v in m.items()}}), flush=True)
        return m

    zero = evaluate(0)
    report["zero_shot"] = zero
    g = torch.Generator().manual_seed(seed)
    order = [i for _ in range(t["epochs"]) for i in torch.randperm(per_epoch, generator=g).tolist()]
    start = time.monotonic()
    verdict = G._result(G.CONTINUE, [])
    for update, i in enumerate(order, 1):
        loss, norm = step(model, params, optimizer, scheduler, R.batch(train_ds, train_rows[i]), t["grad_clip"])
        report["steps"].append({"update": update, "loss": loss, "grad_norm": norm})
        if update % t["eval_every_updates"] == 0 or update == total:
            m = evaluate(update)
            verdict = G.sft_evaluation_verdict(m, zero, m["epoch_fraction"], t)
            m["verdict"] = verdict
            if G.better_checkpoint(m, report["best"]):
                R.save(model, proc, out / "best_checkpoint")
                report["best"], report["best_checkpoint"] = m, str(out / "best_checkpoint")
            R.write_json(out / "progress.json", report)
            if verdict["verdict"] == G.KILL:
                break
    report["train_seconds"] = time.monotonic() - start
    report["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    R.save(model, proc, out / "last_checkpoint")
    report["last_checkpoint"] = str(out / "last_checkpoint")
    if report["best"] is None:  # the pretrained model was never beaten
        report["best"], report["best_checkpoint"] = zero, str(R.MODEL_DIR)
    report["status"] = "killed" if verdict["verdict"] == G.KILL else "complete"
    report["kill"] = verdict if verdict["verdict"] == G.KILL else None
    report["success"] = (G._result(G.FAIL, ["run killed"]) if report["status"] == "killed"
                         else G.sft_success(report["best"], zero, t))
    if report["best_checkpoint"] == str(R.MODEL_DIR) and report["success"]["verdict"] == G.PASS:
        report["success"] = G._result(G.FAIL, ["no finetuned checkpoint beat zero-shot"])
    R.write_json(out / "report.json", report)
    print("STAGE1_TRAIN", report["status"], report["success"]["verdict"], flush=True)
    return 0 if report["success"]["verdict"] == G.PASS else 3


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "train"):
        s = sub.add_parser(name)
        s.add_argument("--codes", required=True, help="output directory of training.fil_moss.encode")
        s.add_argument("--out", required=True)
        s.add_argument("--recipe", default=str(HERE / "recipe.json"))
        if name == "train":
            s.add_argument("--verify-report", required=True)
            s.add_argument("--english", help="English regression prompts: jsonl of {text, ref_audio_codes, seconds}")
    a = p.parse_args(argv)
    return verify(a) if a.cmd == "verify" else train(a)


if __name__ == "__main__":
    raise SystemExit(main())
