"""
ASR finetuning — Whisper on Philippine speech corpora (NOT auto-run; script
prepared ahead of a decision to train).

Swappable dataset via --dataset:
  fsc        — sapinsapin/filipinospeechcorpus (read + spontaneous)
  livestream — sapinsapin/halo-livestream asr config (QC-gated stream clips)

Model: openai/whisper-small (244M) by default — the standard 8GB-card choice:
mixed-precision training with batch 8 fits ~6GB. whisper-medium needs gradient
checkpointing and ~7.5GB (tight when the desktop shares the card).

Precision is bf16 wherever the card supports it (Ampere and later), fp16
otherwise. whisper-large-v3 in the cloud bake-off needs bf16: at 1.55B
parameters fp16 overflows to NaN loss.

The recipe is the canonical HF seq2seq one: log-mel inputs from the feature
extractor, tokenized labels with BOS stripped, generation-based eval scored
with both WER and CER (CER is the more honest metric on Taglish — see
docs/livestream_pipeline.md on orthography variance).

Usage (when ready to train):
  python finetune_asr.py --dataset fsc --max-steps 2000
  python finetune_asr.py --dataset livestream --model openai/whisper-small
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# Whisper's mels are a fixed 30 s, so this matters less here than in the CTC
# arm (see finetune_ctc.py), but keeping both arms on the same allocator makes
# their step times comparable.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
SR = 16000


def prepare_dataset(ds, processor, num_proc: int):
    """Tokenize labels up front; leave log-mel features to the collator.

    Precomputing features used to materialise an 80x3000 float32 mel (about
    1 MB, padded to 30 s whatever the clip length) for every row. At 25k clips
    that is ~25 GB held in RAM, which pushed WSL into swap on a nearly full C:
    and took the VM down on 2026-09-17. Computing them per batch costs a few
    milliseconds a clip on the dataloader workers and holds one batch at a
    time. The features are identical either way.
    """
    # input_columns keeps the map from decoding audio it does not need
    ds = ds.map(lambda text: {"labels": processor.tokenizer(text).input_ids},
                input_columns=["text"], num_proc=num_proc)
    keep = {"audio", "labels"}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    # Whisper's decoder context is 448 tokens; longer labels are truncated by
    # the tokenizer only at generation time, so drop them here instead.
    ds = ds.filter(lambda labels: len(labels) <= 448, input_columns=["labels"])
    return ds


@dataclass
class ASRDataCollator:
    processor: object

    def __call__(self, features):
        batch = self.processor.feature_extractor(
            [f["audio"]["array"] for f in features], sampling_rate=SR,
            return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # BOS is re-added by the model during teacher forcing
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_compute_metrics(processor):
    import jiwer

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        pairs = [(r.strip().lower(), h.strip().lower())
                 for r, h in zip(label_str, pred_str) if r.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        refs, hyps = zip(*pairs)
        return {
            "wer": jiwer.wer(list(refs), list(hyps)),
            "cer": jiwer.cer(list(refs), list(hyps)),
        }

    return compute_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", choices=["fsc", "livestream", "pld", "fsc+pld"], default="fsc",
                    help="fsc+pld concatenates both corpora (equal shares of "
                         "--max-samples); use with --language fil")
    ap.add_argument("--language", default=None,
                    help="ISO 639-3 language filter (pld only), e.g. bcl, ceb")
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-proc", type=int, default=1,
                    help="dataset map workers; keep 1 on WSL — multiprocess "
                         "map over decoded audio deadlocks on WSL2/9p (hangs "
                         "at 0/N forever rather than failing). On a Linux "
                         "cloud VM raise it: prep is otherwise minutes of idle GPU")
    ap.add_argument("--dataloader-workers", type=int, default=2)
    # Mid-training evals are generative and cost ~8 minutes over the full test
    # set: at eval_steps=500 that was ~40% of a run's wall clock. Select the
    # checkpoint on a fixed subsample; the reported number still comes from a
    # final pass over the whole test set.
    ap.add_argument("--eval-samples", type=int, default=500,
                    help="clips used for mid-training evals (0 = all)")
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--gen-max-length", type=int, default=128,
                    help="225 is Whisper's long-form default; PLD prompts are "
                         "single sentences, and generation time scales with it")
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"],
                    help="sdpa is PyTorch's fused attention; the Whisper "
                         "encoder runs 1500 frames per clip, where it pays off")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model: fuses the many small "
                         "kernels the profile showed, at a few minutes of "
                         "compile time on the first step")
    ap.add_argument("--profile", action="store_true",
                    help="profile ~10 steps and stop: prints where step time "
                         "goes and writes a Chrome trace. Trains nothing.")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="trade memory for ~25%% more speed where the card has "
                         "headroom")
    ap.add_argument("--push", action="store_true",
                    help="upload the finetuned model to the Hub as "
                         "<model>-{fsc|halohaloLS} after training")
    ap.add_argument("--normalise", action="store_true",
                    help="train and score on halolib.finetune.normalise_text: "
                         "no stress accents, no punctuation. Run dir gains "
                         "_norm. To continue a finetuned model rather than "
                         "restart, pass it as --model")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in the run dir "
                         "if one exists; required for preemptible cloud VMs")
    args = ap.parse_args()

    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    from halolib.finetune import load_speech_dataset

    if "pld" in args.dataset and not args.language:
        raise SystemExit("--dataset pld requires --language (e.g. --language bcl)")

    ds_tag = args.dataset.replace("+", "-")
    run_name = (f"asr_{ds_tag}_{args.language}" if args.language
                else f"asr_{ds_tag}")
    if args.normalise:
        run_name += "_norm"
    out_dir = FINETUNE_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Whisper's decoder only has language tokens for ~100 languages: English
    # and Tagalog are in the vocabulary, the other Philippine languages are
    # not. Those train under the <|tl|> token — the closest relative — which
    # finetuning effectively repurposes as the language slot.
    whisper_lang = {"eng": "english"}.get(args.language, "tagalog")

    processor = WhisperProcessor.from_pretrained(
        args.model, language=whisper_lang, task="transcribe")
    # transformers 5 loads weights in the checkpoint's own dtype, and
    # whisper-large-v3 ships fp16. Mixed precision wants fp32 master weights,
    # and generation-based eval crashed on fp16 weights meeting fp32 log-mels
    # ("Input type (float) and bias type (c10::Half) should be the same").
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation=args.attn)
    model.generation_config.language = whisper_lang
    model.generation_config.task = "transcribe"
    # finetuned models must not inherit the base's forced en-transcribe ids
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    print(f"Loading dataset: {args.dataset}"
          + (f" [{args.language}]" if args.language else ""))
    # num_proc matters more than it looks: selecting one language means three
    # filter passes over PLD's 300k rows, ~6 minutes each single-process with
    # the GPU idle the whole time
    ds = load_speech_dataset(args.dataset, task="asr",
                             max_samples=args.max_samples,
                             token=os.environ.get("HF_TOKEN"),
                             num_proc=args.num_proc,
                             language=args.language)
    print(ds)

    print("Preprocessing (text→labels; log-mels are computed per batch)...")
    if args.normalise:
        from halolib.finetune import normalise_text
        ds = ds.map(lambda text: {"text": normalise_text(text)},
                    input_columns=["text"])
        ds = ds.filter(lambda text: bool(text), input_columns=["text"])
    ds = prepare_dataset(ds, processor, args.num_proc)

    # fixed subsample, so every checkpoint of every arm is selected on the
    # same clips; seeded, so re-running a preempted job scores the same set
    eval_ds = ds["test"]
    if args.eval_samples and len(eval_ds) > args.eval_samples:
        eval_ds = eval_ds.shuffle(seed=42).select(range(args.eval_samples))
        print(f"  mid-training eval on {len(eval_ds)} of {len(ds['test'])} "
              f"clips; final metric uses all of them")

    trainer = Seq2SeqTrainer(
        args=Seq2SeqTrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=200,
            max_steps=args.max_steps,
            gradient_checkpointing=not args.no_grad_checkpoint,
            # bf16 wherever the card supports it (Ampere and later). fp16 on a
            # 1.55B model such as whisper-large-v3 overflows to NaN loss, and
            # the CTC arm of the bake-off already trains in bf16 — matching the
            # precision keeps that comparison about the model, not the dtype.
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            save_total_limit=2,
            per_device_eval_batch_size=args.batch_size,
            # fused AdamW: one kernel for the whole parameter update instead of
            # a launch per tensor, which the low SM occupancy said we were
            # paying for
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            torch_compile=args.compile,
            logging_steps=25,
            report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
            run_name=run_name,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=args.gen_max_length,
            # features are computed in the collator, so the workers do the
            # decode + log-mel work (~30 ms a clip) that used to be a map
            dataloader_num_workers=args.dataloader_workers,
            remove_unused_columns=False,
        ),
        model=model,
        train_dataset=ds["train"],
        eval_dataset=eval_ds,
        data_collator=ASRDataCollator(processor),
        compute_metrics=build_compute_metrics(processor),
    )

    # On a preemptible VM the process can die at any moment; --resume picks up
    # the newest checkpoint instead of restarting the run. Harmless on the
    # first launch (no checkpoint yet -> fresh start).
    if args.profile:
        # profiling stops after a handful of steps and writes no model
        from halolib.profiling import profiler_callback
        trainer.add_callback(profiler_callback(out_dir))
        trainer.train()
        return

    ckpt = None
    if args.resume:
        from halolib.finetune import latest_checkpoint
        ckpt = latest_checkpoint(out_dir)
        print(f"resume: {ckpt or 'no checkpoint found, starting fresh'}")
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    print(f"Saved: {out_dir / 'final'}")

    # Same record finetune_ctc.py writes, so the bake-off report can put this
    # run beside the CTC arms and state which split produced its number.
    # load_best_model_at_end has restored the best checkpoint; score it once on
    # the WHOLE test set, since mid-training evals only saw a subsample.
    import json as _json
    best = trainer.evaluate(ds["test"], metric_key_prefix="eval")
    print(f"final eval on {len(ds['test'])} clips: "
          f"CER {best.get('eval_cer')} WER {best.get('eval_wer')}")
    split_kind = "random-overlapping"
    if args.language and os.environ.get("PLD_SPLIT") != "random":
        try:
            from halolib.splits import load_spec
            load_spec(args.language)
            split_kind = "frozen-disjoint"
        except FileNotFoundError:
            pass
    (out_dir / "result.json").write_text(_json.dumps({
        "encoder": args.model, "units": "bpe", "dataset": args.dataset,
        "language": args.language, "split": split_kind,
        "steps": args.max_steps, "train_rows": len(ds["train"]),
        "eval_cer": best.get("eval_cer"), "eval_wer": best.get("eval_wer"),
        "eval_loss": best.get("eval_loss"),
    }, indent=1))

    if args.push:
        from halolib.finetune import push_model_to_hub
        final_eval = best   # already the full-test-set pass, do not redo it
        push_kwargs = {}
        if "pld" in args.dataset:
            push_kwargs = dict(
                suffix=f"{ds_tag}-{args.language}",
                lang_code=args.language,
                extra_tags=["philippines", "philippine-languages",
                            args.language, "whisper"],
            )
        push_model_to_hub(
            out_dir / "final", args.model, args.dataset, "asr",
            token=os.environ.get("HF_TOKEN"),
            **push_kwargs,
            metrics={k.removeprefix("eval_"): v for k, v in final_eval.items()
                     if k in ("eval_wer", "eval_cer", "eval_loss")},
            train_summary=(
                f"Trained for {args.max_steps} steps on "
                f"{len(ds['train'])} clips (batch {args.batch_size}×"
                f"{args.grad_accum}, lr {args.lr}, fp16 + gradient "
                f"checkpointing). WER/CER are on the held-out split, "
                f"lowercased; CER is the model-selection metric (Taglish "
                f"orthography varies at the word level)."),
            # PLD is CC-BY-NC and research-only; the weights inherit that
            license="cc-by-nc-4.0" if "pld" in args.dataset else "apache-2.0",
        )


if __name__ == "__main__":
    main()
