"""
CTC finetuning — the foundation bake-off (plan R1) and the output-unit
ablation (R2) in one script.

Encoders (all permissive, so results feed both tracks):
  omni-300m|1b|3b|7b   Meta Omnilingual ASR self-supervised encoders,
                       Apache-2.0, pretrained on nine of the ten PLD
                       languages (ylacombe/* are the safetensors conversions
                       of facebook/omniASR_W2V_*; Meta's own .pt files need
                       fairseq2, which we deliberately avoid)
  w2v-bert             facebook/w2v-bert-2.0, MIT, 4.5M h / 143 languages

Output units (--units), the R2 ablation:
  char       characters — the MMS and Omnilingual baseline
  syllable   Philippine syllables, English words spelled out
             (halolib.syllables; measured inventory 4.4k, top-2k covers 99.5 %)

Why CTC at all: it streams, it hallucinates far less than an attention
decoder, and it has no subword vocabulary to tax — the 2x token penalty
measured for every BPE tokenizer on these languages simply does not apply.

  python finetune_ctc.py --encoder omni-1b --language ceb --units char
  python finetune_ctc.py --encoder w2v-bert --language pam --units syllable --resume
  python finetune_ctc.py --encoder omni-300m --language ceb --smoke   # 20 clips, CPU-able

Runs land in $FINETUNE_DIR/ctc_{encoder}_{units}_{dataset}_{lang}/.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

# Audio batches run from 0.4 s to 20 s, so every step asks the CUDA allocator
# for a differently shaped block and its pools fragment. Measured on the 3070
# with the 300M encoder: a step on 20 s clips costs 6.54 s under the default
# allocator and 2.14 s with expandable segments, while 5 s clips are unaffected
# — the long batches are the ones that thrash. Same arithmetic, same memory.
# Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
SR = 16000

ENCODERS = {
    "omni-300m": ("ylacombe/omniASR_W2V_300M_SSL", "wav2vec2"),
    "omni-1b": ("ylacombe/omniASR_W2V_1B_SSL", "wav2vec2"),
    "omni-3b": ("ylacombe/omniASR_W2V_3B_SSL", "wav2vec2"),
    "omni-7b": ("ylacombe/omniASR_W2V_7B_SSL", "wav2vec2"),
    "w2v-bert": ("facebook/w2v-bert-2.0", "wav2vec2-bert"),
}

PAD, UNK, DELIM = "<pad>", "<unk>", "|"      # PAD doubles as the CTC blank


def text_to_units(text: str, kind: str) -> list[str]:
    from halolib.syllables import units as syl_units
    if kind == "syllable":
        return syl_units(text, english="chars", delim=DELIM)
    # char: same word-delimiter convention so both arms are comparable
    out = []
    for i, w in enumerate(text.lower().split()):
        if i:
            out.append(DELIM)
        out.extend(list(w))
    return out


def build_vocab(texts, kind: str, min_count: int = 2) -> dict:
    """Closed unit vocabulary. Units seen once are dropped to <unk>: a CTC
    head cannot emit what it never saw enough of to learn, and a long tail of
    singletons only dilutes the softmax."""
    from collections import Counter
    counts = Counter()
    for t in texts:
        counts.update(u for u in text_to_units(t, kind) if u != DELIM)
    keep = [u for u, c in counts.most_common() if c >= min_count]
    vocab = {PAD: 0, UNK: 1, DELIM: 2}
    for u in keep:
        vocab.setdefault(u, len(vocab))
    cov = sum(c for u, c in counts.items() if u in vocab) / max(sum(counts.values()), 1)
    print(f"  vocab: {len(vocab)} units ({len(counts)} seen), "
          f"token coverage {cov * 100:.2f}%")
    return vocab


def ctc_decode(ids, id2unit: dict) -> str:
    """Greedy CTC over per-frame argmax ids: collapse repeats, drop blanks and
    batch padding, join. Takes ids rather than logits because the logits are
    reduced on the GPU during evaluation (see preprocess_logits_for_metrics)."""
    out, prev = [], None
    for i in ids:
        i = int(i)
        if i < 0:                 # -100 padding added when batches are stacked
            prev = None
            continue
        if i != prev and i != 0:  # 0 is the blank
            out.append(id2unit.get(i, ""))
        prev = i
    return "".join(out).replace(DELIM, " ").strip()


class CTCCollator:
    """Extract and pad audio features per batch; label padding is -100 so it
    leaves the loss. Extracting here rather than in a map keeps one batch of
    waveforms in memory instead of the whole corpus."""

    def __init__(self, extractor, audio_key):
        self.extractor = extractor
        self.audio_key = audio_key

    def __call__(self, features):
        if self.audio_key in features[0]:       # --eager-features
            batch = self.extractor.pad(
                [{self.audio_key: f[self.audio_key]} for f in features],
                padding=True, return_tensors="pt")
        else:
            batch = self.extractor([f["audio"]["array"] for f in features],
                                   sampling_rate=SR, padding=True,
                                   return_tensors="pt")
        n = max(len(f["labels"]) for f in features)
        labels = torch.full((len(features), n), -100, dtype=torch.long)
        for i, f in enumerate(features):
            labels[i, :len(f["labels"])] = torch.tensor(f["labels"], dtype=torch.long)
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--encoder", choices=list(ENCODERS), default="omni-1b")
    ap.add_argument("--units", choices=["char", "syllable"], default="char")
    ap.add_argument("--dataset", choices=["fsc", "livestream", "pld", "fsc+pld"],
                    default="pld")
    ap.add_argument("--language", default=None, help="ISO 639-3 filter (pld)")
    ap.add_argument("--max-samples", type=int, default=25000)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)   # CTC heads like more LR than seq2seq
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--num-proc", type=int, default=1)
    ap.add_argument("--dataloader-workers", type=int, default=2)
    ap.add_argument("--profile", action="store_true",
                    help="profile ~10 steps and stop: prints where step time "
                         "goes and writes a Chrome trace. Trains nothing.")
    ap.add_argument("--eval-samples", type=int, default=0,
                    help="clips for mid-training evals (0 = all; CTC eval is "
                         "argmax, not generation, so it is already cheap)")
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile; expect recompiles on this arm, whose "
                         "audio batches vary in length")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="trade memory for ~25%% more speed; safe on a card "
                         "with headroom (the 1B encoder at batch 16 uses "
                         "~40 GB of the 96 GB RTX PRO 6000)")
    ap.add_argument("--optim", default=None,
                    help="override the optimiser, e.g. adamw_bnb_8bit. Fused "
                         "AdamW keeps 8 bytes of state per parameter, which is "
                         "56 GB for the 7B encoder on its own; bitsandbytes' "
                         "8-bit Adam keeps 2")
    ap.add_argument("--bf16-weights", action="store_true",
                    help="hold the weights in bf16 rather than fp32: halves "
                         "weights and gradients (28 GB -> 14 GB each at 7B). "
                         "The fallback when 8-bit Adam alone does not fit — "
                         "small updates round away in bf16, so prefer fp32 "
                         "weights whenever they fit")
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint (preemptible VMs)")
    ap.add_argument("--eager-features", action="store_true",
                    help="precompute audio features for the whole corpus "
                         "instead of per batch: faster per step where the RAM "
                         "exists, but ~31 GB for 24k clips at 20 s")
    ap.add_argument("--smoke", action="store_true",
                    help="20 clips, 4 steps — validates the whole path cheaply")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    from transformers import (AutoFeatureExtractor, Trainer, TrainingArguments,
                              Wav2Vec2BertForCTC, Wav2Vec2ForCTC)

    from halolib.finetune import load_speech_dataset

    if "pld" in args.dataset and not args.language:
        raise SystemExit("--dataset pld requires --language")
    if args.smoke:
        args.max_samples, args.max_steps, args.batch_size = 20, 4, 2

    repo, kind = ENCODERS[args.encoder]
    tag = f"{args.dataset.replace('+', '-')}" + (f"_{args.language}" if args.language else "")
    run_name = f"ctc_{args.encoder}_{args.units}_{tag}"
    out_dir = FINETUNE_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}" + (f" [{args.language}]" if args.language else ""))
    ds = load_speech_dataset(args.dataset, task="asr",
                             max_samples=args.max_samples,
                             token=os.environ.get("HF_TOKEN"),
                             num_proc=args.num_proc,
                             language=args.language)
    print(ds)

    # Digits are the one place the two unit systems disagree: the syllable
    # frontend's word regex drops them outright (units("100") == []) while the
    # character arm keeps '1','0','0'. Left alone that gives the two arms
    # different training sets and confounds the very comparison this script
    # exists to make. Both drop digit-bearing rows — the same choice the TTS
    # harness makes, and a standing reminder that numeral verbalisation is an
    # open gap (plan §4, text normalisation).
    digit = re.compile(r"\d")
    before = {k: len(v) for k, v in ds.items()}
    ds = ds.filter(lambda r: not digit.search(r["text"] or ""))
    dropped = ", ".join(f"{k} {before[k] - len(ds[k])}" for k in sorted(ds))
    print(f"  dropped digit-bearing rows: {dropped}")

    vocab = build_vocab(ds["train"]["text"], args.units)
    id2unit = {v: k for k, v in vocab.items()}
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1))

    if args.units == "syllable":
        from halolib.syllables import fallback_rate
        fb = fallback_rate(ds["train"]["text"])
        print(f"  english fallback: {fb['rate'] * 100:.1f}% of words")

    extractor = AutoFeatureExtractor.from_pretrained(repo)
    audio_key = "input_features" if kind == "wav2vec2-bert" else "input_values"

    # Labels and durations only: the audio features are built per batch in the
    # collator. Materialising them here meant holding every clip's waveform in
    # memory — up to ~31 GB for 24k clips at 20 s, against WSL's 32 GB — and on
    # 2026-09-18 the map's worker pool took the VM down twice at exactly this
    # point. The features are identical either way.
    def prepare(audio, text):
        us = text_to_units(text, args.units)
        row = {"labels": [vocab.get(u, vocab[UNK]) for u in us],
               "n_sec": len(audio["array"]) / SR}
        if args.eager_features:
            row[audio_key] = extractor(audio["array"], sampling_rate=SR)[audio_key][0]
        return row

    cols = [c for c in ds["train"].column_names
            if c != "audio" or args.eager_features]
    ds = ds.map(prepare, input_columns=["audio", "text"],
                remove_columns=cols, num_proc=args.num_proc)
    # CTC needs at least one output frame per label; drop clips that cannot
    # satisfy it (very long text on very short audio) and cap length for VRAM.
    ds = ds.filter(lambda r: 0.4 <= r["n_sec"] <= args.max_seconds
                   and len(r["labels"]) >= 1
                   and len(r["labels"]) < r["n_sec"] * 40)
    ds = ds.remove_columns(["n_sec"])
    print(ds)

    Model = Wav2Vec2BertForCTC if kind == "wav2vec2-bert" else Wav2Vec2ForCTC
    model = Model.from_pretrained(
        repo,
        dtype=torch.bfloat16 if args.bf16_weights else torch.float32,
        attn_implementation=args.attn,
        vocab_size=len(vocab),
        ctc_loss_reduction="mean",
        pad_token_id=vocab[PAD],
        ctc_zero_infinity=True,
        attention_dropout=0.05,
        hidden_dropout=0.05,
        layerdrop=0.0,
        mask_time_prob=0.05,
    )
    # The SSL checkpoints are *ForPreTraining; the CTC head is new by design.
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()      # conv frontend: pretrained, tiny, unstable to tune
    model.config.use_cache = False

    import jiwer

    def compute_metrics(pred):
        ids = pred.predictions      # (N, T) argmax ids, reduced on the GPU
        hyps = [ctc_decode(ids[i], id2unit) for i in range(len(ids))]
        labels = pred.label_ids
        refs = []
        for row in labels:
            us = [id2unit.get(int(i), "") for i in row if i != -100]
            refs.append("".join(us).replace(DELIM, " ").strip())
        pairs = [(r, h) for r, h in zip(refs, hyps) if r]
        if not pairs:
            return {"cer": 1.0, "wer": 1.0}
        refs, hyps = zip(*pairs)
        return {"cer": jiwer.cer(list(refs), list(hyps)),
                "wer": jiwer.wer(list(refs), list(hyps))}

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # same fixed subsample logic as the Whisper arm; 0 keeps the full set
    eval_ds = ds["test"]
    if args.eval_samples and len(eval_ds) > args.eval_samples:
        eval_ds = eval_ds.shuffle(seed=42).select(range(args.eval_samples))
        print(f"  mid-training eval on {len(eval_ds)} of {len(ds['test'])} clips")

    trainer = Trainer(
        args=TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=max(1, args.batch_size // 2),
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=args.warmup,
            max_steps=args.max_steps,
            gradient_checkpointing=not args.no_grad_checkpoint,
            bf16=bf16,
            eval_strategy="steps",
            eval_steps=args.eval_steps if not args.smoke else 4,
            save_steps=args.eval_steps if not args.smoke else 4,
            optim=args.optim or ("adamw_torch_fused" if torch.cuda.is_available()
                                 else "adamw_torch"),
            torch_compile=args.compile,
            save_total_limit=2,
            logging_steps=25,
            report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
            run_name=run_name,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            label_names=["labels"],
            dataloader_num_workers=args.dataloader_workers,
            remove_unused_columns=False,
            eval_accumulation_steps=4,
        ),
        model=model,
        train_dataset=ds["train"],
        eval_dataset=eval_ds,
        data_collator=CTCCollator(extractor, audio_key),
        compute_metrics=compute_metrics,
        # Without this the trainer keeps every frame's full score vector for
        # every eval clip: 2,500 clips x ~350 frames x a ~2,000-unit syllable
        # vocabulary is several GB of RAM, enough to stall or kill the syllable
        # arm at its first evaluation. Argmax on the GPU keeps one id per frame.
        preprocess_logits_for_metrics=lambda logits, labels: (
            logits[0] if isinstance(logits, tuple) else logits).argmax(-1),
    )

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
        print(f"resume: {ckpt or 'no checkpoint yet, starting fresh'}")
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(str(out_dir / "final"))
    extractor.save_pretrained(str(out_dir / "final"))
    (out_dir / "final" / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=1))

    final = trainer.evaluate(ds["test"])   # headline number: whole test set
    print({k: v for k, v in final.items() if k in ("eval_cer", "eval_wer", "eval_loss")})
    # Record which split produced these numbers. Without this a run that
    # silently fell back to the random split would be reported as though it
    # were speaker- and prompt-disjoint, which is the exact overclaim the
    # review flagged.
    split_kind = "random-overlapping"
    if args.language:
        try:
            from halolib.splits import load_spec
            load_spec(args.language)
            split_kind = "frozen-disjoint"
        except FileNotFoundError:
            pass
    (out_dir / "result.json").write_text(json.dumps(
        {"encoder": repo, "units": args.units, "dataset": args.dataset,
         "language": args.language, "vocab": len(vocab),
         "split": split_kind,
         "steps": args.max_steps, "train_rows": len(ds["train"]),
         **{k: v for k, v in final.items()
            if k in ("eval_cer", "eval_wer", "eval_loss")}}, indent=1))

    if args.push:
        from halolib.finetune import push_model_to_hub
        push_model_to_hub(
            out_dir / "final", repo, args.dataset, "asr",
            token=os.environ.get("HF_TOKEN"),
            suffix=f"ctc-{args.units}-{tag}",
            lang_code=args.language or "tl",
            metrics={k.removeprefix("eval_"): v for k, v in final.items()
                     if k in ("eval_cer", "eval_wer", "eval_loss")},
            train_summary=(
                f"CTC head on {repo} ({args.units} units, {len(vocab)} of them) "
                f"for {args.max_steps} steps on {len(ds['train'])} clips. "
                f"Trained on PLD, which is CC-BY-NC and research-only: this "
                f"checkpoint is a research artifact regardless of the base "
                f"model's licence. See docs/pld_sota_track.md §2."),
            license="cc-by-nc-4.0",
            extra_tags=["philippines", "philippine-languages", "ctc",
                        args.language or "tl"],
        )


if __name__ == "__main__":
    main()
