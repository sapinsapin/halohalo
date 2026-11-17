"""Wall-clock estimate for stage 1 (smoke) and stage 2 (baseline SFT + evaluation) from measured component times.
Standard library only.

``bench.py`` measures the components on the GB10 (and the remote judges on the 3070); ``estimate`` combines them
with the dataset size and recipe. Every term is in the output, so a wrong assumption is visible.
"""
import math


def judge_costs(t, recipe, remote):
    """Per-utterance (GB10-side, remote-side) judge seconds under the recipe's placement. Whisper runs batched
    in both places, so its batched per-utterance time is used when measured."""
    local_times = dict(t["judge_local_seconds"])
    if t.get("judge_local_seconds_whisper_batch8_per_utt"):
        local_times["whisper"] = t["judge_local_seconds_whisper_batch8_per_utt"]
    placement = recipe["judges"].get("placement", {}) if remote else {}
    on_gb10 = sum(v for k, v in local_times.items() if placement.get(k) != "remote")
    far = t["judge_remote_seconds_per_utt"] if remote and any(v == "remote" for v in placement.values()) else 0.0
    return on_gb10, far


def gen_seconds(t, audio_seconds, batch):
    """Seconds to generate one batch whose longest item is ``audio_seconds`` long."""
    key = "8" if batch > 1 else "1"
    return t["gen_prefill_seconds"] + audio_seconds * t["frames_per_second"] * t["gen_seconds_per_frame"][key]


def eval_seconds(t, recipe, n_gen, n_val, clip_seconds, gen_batch, remote):
    """One evaluation: per-domain teacher-forced losses, then batched generation and judging; remote judging of
    batch i overlaps generation of batch i + 1, so only its excess and the last batch show."""
    batches = math.ceil(n_gen / gen_batch)
    gen = batches * gen_seconds(t, clip_seconds * 1.2, gen_batch)
    decode = n_gen * t["decode_seconds_per_audio_second"] * clip_seconds
    on_gb10, far = judge_costs(t, recipe, remote)
    local = n_gen * (on_gb10 + 2 * t["ecapa_seconds"] + 2 * t["trim_seconds"])
    judge = (max(0.0, n_gen * far - gen) + n_gen * far / batches) if far else 0.0
    tf = math.ceil(n_val / 8) * t["forward_seconds"]["8"]
    return gen + decode + local + judge + tf


def estimate(t, data, recipe, remote=True, batch_size=None):
    s1, s2 = recipe["stage1"], recipe["stage2"]
    clip = data["mean_clip_seconds"]
    n_train = int(data["train_hours"] * 3600 / clip)
    n_val = int(data["validation_hours"] * 3600 / clip)
    bs = batch_size or s2["batch_size"]
    out = {"inputs": dict(data, train_clips=n_train, validation_clips=n_val, remote_judges=remote, batch_size=bs)}

    smoke = (t["load_seconds"] * 2 + 2 * gen_seconds(t, clip * 1.2, s1["sample_texts"])
             + t["train_step_seconds"]["1"] + t["save_reload_seconds"] + 4 * t["forward_seconds"]["4"])
    out["stage1_seconds"] = smoke

    updates = math.ceil(n_train / bs) * s2["epochs"]
    step = t["train_step_seconds"][str(bs)]
    n_evals = math.ceil(updates / s2["eval_every_updates"]) + 1
    ev = eval_seconds(t, recipe, s2["eval_generation_clips"], n_val, clip, s2["gen_batch"], remote)
    saves = n_evals * t["save_seconds"]
    out["stage2"] = {"updates": updates, "train_seconds": updates * step, "evals": n_evals,
                     "eval_seconds_each": ev, "save_seconds": saves,
                     "total_seconds": t["load_seconds"] * 2 + updates * step + n_evals * ev + saves}
    out["latency_seconds"] = t["load_seconds"] + 11 * (gen_seconds(t, clip * 1.2, 1) + 1.0)

    s3 = recipe["stage3"]
    k, micro = s3["rollouts_per_prompt"], s3["logprob_microbatch"]
    groups = math.ceil(k / micro)
    produce = (gen_seconds(t, clip * 1.2, k) + k * t["decode_seconds_per_audio_second"] * clip
               + (groups * t["forward_seconds"][str(micro)] if s3["pipeline"] else 0.0))
    on_gb10, far = judge_costs(t, recipe, remote)
    consume = (k * (on_gb10 + t["ecapa_seconds"] + t["trim_seconds"])
               + groups * (t["train_step_seconds"][str(micro)] + t["ref_forward_seconds"][str(micro)])
               + t["sync_seconds"])
    judge = k * far
    step3 = max(produce + consume, judge) if (far and s3["pipeline"]) else produce + consume + judge
    evals3 = s3["max_steps"] // s3["eval_every_steps"] + 1
    ev3 = eval_seconds(t, recipe, s3["eval_generation_clips"], n_val, clip, s3["gen_batch"], remote)
    setup3 = t["load_seconds"] * 3 + math.ceil(s3["anchor_clips"] / s3["gen_batch"]) * gen_seconds(t, clip * 1.2, 8)
    arm = setup3 + s3["max_steps"] * step3 + evals3 * (ev3 + t["save_seconds"])
    early = setup3 + s3["kill"]["no_gain_step"] * step3 + (s3["kill"]["no_gain_step"] // s3["eval_every_steps"] + 1) * (
        ev3 + t["save_seconds"])
    arms_runnable = 3  # a, b, c; d needs a reference-conditioned stage 2 and consented bilingual speakers
    out["stage3_arm"] = {"step_seconds": step3, "produce": produce, "consume": consume, "judge": judge,
                         "steps": s3["max_steps"], "evals": evals3, "eval_seconds_each": ev3,
                         "total_seconds": arm, "killed_at_no_gain_step_seconds": early}
    out["stage3_arms_abc_seconds"] = arms_runnable * arm
    out["everything_seconds"] = (smoke + out["stage2"]["total_seconds"] + out["latency_seconds"]
                                 + out["stage3_arms_abc_seconds"])
    return out
