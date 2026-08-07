"""
ASR finetuning — Whisper on Philippine speech corpora (NOT auto-run; script
prepared ahead of a decision to train).

Swappable dataset via --dataset:
  fsc        — sapinsapin/filipinospeechcorpus (read + spontaneous)
  livestream — sapinsapin/halo-livestream asr config (QC-gated stream clips)

Model: openai/whisper-small (244M) by default — the standard 8GB-card choice:
fp16 training with batch 8 fits ~6GB. whisper-medium needs gradient
checkpointing and ~7.5GB (tight when the desktop shares the card).

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

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
SR = 16000


def prepare_dataset(ds, processor, num_proc: int):
    def _process(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=SR).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    cols = [c for c in ds["train"].column_names if c not in ("input_features", "labels")]
    ds = ds.map(_process, remove_columns=cols, num_proc=num_proc)
    # Whisper's decoder context is 448 tokens; longer labels are truncated by
    # the tokenizer only at generation time, so drop them here instead.
    ds = ds.filter(lambda r: len(r["labels"]) <= 448)
    return ds


@dataclass
class ASRDataCollator:
    processor: object

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

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
    ap.add_argument("--dataset", choices=["fsc", "livestream"], default="fsc")
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-proc", type=int, default=4)
    args = ap.parse_args()

    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    from halolib.finetune import load_speech_dataset

    out_dir = FINETUNE_DIR / f"asr_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(
        args.model, language="tagalog", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "tagalog"
    model.generation_config.task = "transcribe"
    # finetuned models must not inherit the base's forced en-transcribe ids
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    print(f"Loading dataset: {args.dataset}")
    ds = load_speech_dataset(args.dataset, task="asr",
                             max_samples=args.max_samples,
                             token=os.environ.get("HF_TOKEN"))
    print(ds)

    print("Preprocessing (audio→log-mel, text→labels)...")
    ds = prepare_dataset(ds, processor, args.num_proc)

    trainer = Seq2SeqTrainer(
        args=Seq2SeqTrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=200,
            max_steps=args.max_steps,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            eval_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            logging_steps=25,
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=225,
            dataloader_num_workers=2,
            remove_unused_columns=False,
        ),
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=ASRDataCollator(processor),
        compute_metrics=build_compute_metrics(processor),
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    print(f"Saved: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
