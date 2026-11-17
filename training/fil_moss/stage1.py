"""Stage 1, engineering smoke of MOSS-TTS-Realtime on the frozen Filipino codes (GPU, dwnd-01).

    python -m training.fil_moss.guard --max-seconds 1800 -- \
        python -m training.fil_moss.stage1 --codes /abs/run/codes --out /abs/run/stage1

Execution-sequence step 4 of README.md, with the Hiligaynon pilot's checks, before anything is scaled:

1. parity: per-token log-probabilities gathered from ``local_logits`` reproduce upstream's loss (evaluation and
   per-domain losses are computed from them);
2. batch parity: the same per-utterance values in one padded batch and one at a time (batched training and
   evaluation are allowed only if this holds);
3. one step: forward/backward/AdamW8bit on real training codes; FP32 gradients, BF16 linear outputs observed with
   a hook, finite loss and gradient norm, and the probed weights moved;
4. save/reload: the checkpoint reloads to identical weights and an identical validation loss;
5. comparison samples: matched-seed generations of validation texts from the pretrained model and the one-step
   model, both finite and not silent, written as WAVs for listening.

All five must pass before stage 2 will start. Nothing here is a quality measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import evaluation as E
from . import gates as G
from . import judges as J
from . import runtime as R
from .scoring import silence_fraction

HERE = Path(__file__).resolve().parent


def load_codes(codes_dir, split):
    if split == "test":
        raise SystemExit("stages 1 and 2 never read the test split")
    return R.read_jsonl(Path(codes_dir) / ("%s_codes.jsonl" % split))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def per_utterance_logp(model, ds, rows, batch_size):
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for part in R.chunks(rows, batch_size):
            logp, valid, _, sample = R.token_logprobs(model, R.batch_many(ds, part))
            for j in range(len(part)):
                m = sample == j
                out.append(float(logp[m][valid[m]].mean()))
    return out


def samples(eng, rows, gen_cfg, seed, out_dir, label):
    """Matched-seed, unconditioned generations (as in the pilot), each checked for finite, non-silent audio."""
    import soundfile as sf
    res = []
    for part, audio in E._batches(eng, rows, "none", gen_cfg, len(rows), seed):
        for r, (a24, trunc) in zip(part, audio):
            name = "%s_%s" % (label, r["id"])
            sf.write(Path(out_dir) / (name + ".wav"), a24, R.SAMPLE_RATE)
            res.append({"name": name, "seconds": len(a24) / R.SAMPLE_RATE, "truncated": bool(trunc),
                        "finite": E.finite(a24), "silence_fraction": silence_fraction(a24.tolist(), R.SAMPLE_RATE)})
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--codes", required=True, help="output directory of training.fil_moss.encode")
    p.add_argument("--out", required=True)
    p.add_argument("--recipe", default=str(HERE / "recipe.json"))
    a = p.parse_args(argv)
    recipe = json.loads(Path(a.recipe).read_text())
    s1, s2, gen, seed = recipe["stage1"], recipe["stage2"], recipe["generation"], recipe["baseline"]["seed"]
    out = R.new_output_dir(a.out)
    (out / "samples").mkdir()
    R.setup()
    R.seed_all(seed)
    proc = R.processor()
    train = [E.condition(r, "none") for r in load_codes(a.codes, "train")]
    val = [E.condition(r, "none") for r in load_codes(a.codes, "validation")]
    probe = E.generation_subset(train, s1["probe_clips"], seed, "none")
    texts = E.generation_subset(val, s1["sample_texts"], seed, "none")
    ds = R.dataset(probe, proc)
    report = {"recipe_sha256": sha(a.recipe), "codes": a.codes, "probe_ids": [r["id"] for r in probe],
              "sample_ids": [r["id"] for r in texts], "checks": {}}
    checks = report["checks"]

    model = R.load_model(R.MODEL_DIR)
    params = R.freeze(model, s2["frozen"])
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mine, theirs = R.wrapper_loss(model, R.batch(ds, probe[0]))
    checks["parity"] = G.parity_check(float(mine), float(theirs), s1["parity_rtol"])
    checks["batch_parity"] = G.batch_parity_check(per_utterance_logp(model, ds, probe, 1),
                                                  per_utterance_logp(model, ds, probe, len(probe)),
                                                  s1["batch_parity_atol"])
    print(json.dumps({k: v for k, v in checks.items()}), flush=True)

    codec = R.load_codec()
    mirror = R.mirror(R.MODEL_DIR)
    eng = R.engine(mirror, codec)
    before = samples(eng, texts, gen, seed, out / "samples", "pretrained")

    model.train()
    optimizer = R.adamw8bit(params, s2["learning_rate"], s2)
    probe_param = next(q for q in params if q.ndim == 2)
    probe_before = probe_param.detach().flatten()[:4096].clone()
    seen = R.observe_linear_dtype(model)
    loss, norm = R.train_step(model, params, optimizer, None, R.batch(ds, probe[0]), s2["grad_clip"])
    one = {"loss": loss, "grad_norm": norm, "grads_fp32": True, "linear_output_dtype": seen.get("dtype"),
           "probe_changed_elements": int((probe_before != probe_param.detach().flatten()[:4096]).sum()),
           "optimizer_state_bytes": sum(t.numel() * t.element_size() for st in optimizer.state.values()
                                        for t in st.values() if torch.is_tensor(t)),
           "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    checks["one_step"] = G.one_step_check(one)
    report["one_step"] = one
    print(json.dumps({"one_step": one}), flush=True)

    R.sync(mirror, model)
    after = samples(eng, texts, gen, seed, out / "samples", "one_step")
    checks["samples"] = G.sample_check(before + after)
    report["samples"] = before + after

    val_ds = R.dataset(val, proc)
    loss_before = E.domain_losses(model, val_ds, val[:16], "none", s2["eval_batch_size"])["balanced_validation_loss"]
    ckpt = out / "checkpoint"
    R.save(model, proc, ckpt)
    probe_w = {n: q.detach()[:8].cpu().clone() for n, q in list(model.named_parameters())[:4]}
    del eng, mirror, optimizer
    model.cpu()
    del model
    torch.cuda.empty_cache()
    reloaded = R.load_model(ckpt, train=False)
    same = all(torch.equal(probe_w[n], q.detach()[:8].cpu()) for n, q in list(reloaded.named_parameters())[:4])
    loss_after = E.domain_losses(reloaded, val_ds, val[:16], "none", s2["eval_batch_size"])["balanced_validation_loss"]
    checks["reload"] = G.reload_check(same, loss_before, loss_after)
    del reloaded
    import shutil
    shutil.rmtree(ckpt)  # an 8 GB one-step checkpoint has no further use
    report["smoke"] = G.smoke_verdict(checks)
    R.write_json(out / "report.json", report)
    print("STAGE1_SMOKE", report["smoke"]["verdict"], flush=True)
    return 0 if report["smoke"]["verdict"] == G.PASS else 3


if __name__ == "__main__":
    raise SystemExit(main())
