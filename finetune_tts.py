"""
TTS finetuning — SpeechT5 on Philippine speech corpora (8GB-VRAM-friendly).

Swappable dataset via --dataset:
  fsc        — sapinsapin/filipinospeechcorpus (read speech; the default —
               studio-ish recordings are what TTS wants)
  livestream — sapinsapin/halo-livestream tts config (QC-gated stream clips;
               useful once the corpus has enough files to matter)

Model: microsoft/speecht5_tts (144M) + speechbrain x-vectors for speaker
conditioning + microsoft/speecht5_hifigan vocoder for the post-train sample.

Why SpeechT5: it is the only mainstream TTS with a first-class HF Trainer
finetuning path that fits an 8GB card (fp32 + gradient checkpointing ≈ 5GB).
VITS/XTTS/F5 finetunes all need bigger cards or bespoke trainers.

Usage:
  python finetune_tts.py --dataset fsc --max-steps 1000
  python finetune_tts.py --dataset livestream --max-samples 500 --max-steps 200
  python finetune_tts.py --dataset fsc --synthesize-only --checkpoint <dir>

Outputs land in {FINETUNE_DIR}/tts_{dataset}/ (checkpoints + sample wav).
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
SR = 16000

DIGIT_RE = re.compile(r"\d")

SAMPLE_TEXTS = [
    "Magandang umaga po sa inyong lahat.",
    "Salamat sa pakikinig, hanggang sa muli.",
]


def clean_text(text: str) -> str | None:
    """SpeechT5's tokenizer is character-based latin. Normalize apostrophes;
    reject digit-bearing text (numeral verbalization is an open gap — see
    docs/livestream_pipeline.md) rather than teach the model to skip them."""
    text = text.replace("’", "'").replace("‘", "'").strip()
    if not text or DIGIT_RE.search(text):
        return None
    return text


def build_speaker_embedder():
    """speechbrain x-vector encoder (512-dim), the embedding SpeechT5 expects."""
    from speechbrain.inference.speaker import EncoderClassifier
    savedir = Path(os.environ.get("HF_HOME", "~/.cache")).expanduser() / "speechbrain-xvect"
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=str(savedir),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )


def prepare_dataset(ds, processor, embedder, num_proc: int):
    """Map raw (audio, text, speaker_id) → model inputs.

    Speaker embeddings are computed per-clip (not per-speaker-average): clips
    of one speaker vary in channel/distance, and per-clip embeddings let the
    model see that variance instead of a single collapsed point.
    """

    def _process(batch):
        text = clean_text(batch["text"])
        if text is None:
            return {"input_ids": None, "labels": None, "speaker_embeddings": None}

        audio = batch["audio"]
        example = processor(
            text=text,
            audio_target=audio["array"],
            sampling_rate=SR,
            return_attention_mask=False,
        )
        with torch.no_grad():
            wav = torch.tensor(np.asarray(audio["array"], dtype=np.float32)).unsqueeze(0)
            emb = embedder.encode_batch(wav)
            emb = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu().numpy()

        return {
            "input_ids": example["input_ids"],
            "labels": example["labels"][0],
            "speaker_embeddings": emb,
        }

    cols = ds["train"].column_names
    ds = ds.map(_process, remove_columns=cols, num_proc=num_proc)
    ds = ds.filter(lambda r: r["input_ids"] is not None)
    # cap sequence lengths: long mels dominate VRAM (labels are ~62 frames/s)
    ds = ds.filter(lambda r: len(r["input_ids"]) <= 220 and len(r["labels"]) <= 960)
    return ds


class TTSDataCollator:
    """Pad text/mel/embedding batches; mask padded mel frames with -100 and
    round lengths down to the model's reduction factor."""

    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def __call__(self, features):
        input_ids = [{"input_ids": f["input_ids"]} for f in features]
        label_features = [{"input_values": f["labels"]} for f in features]
        speaker_features = [f["speaker_embeddings"] for f in features]

        batch = self.processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt")

        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100).float()
        del batch["decoder_attention_mask"]

        # trim to a multiple of the reduction factor
        if self.model.config.reduction_factor > 1:
            lengths = torch.tensor([len(f["input_values"]) for f in label_features])
            target = (lengths.max() // self.model.config.reduction_factor
                      ) * self.model.config.reduction_factor
            batch["labels"] = batch["labels"][:, :target]

        batch["speaker_embeddings"] = torch.tensor(
            np.array(speaker_features, dtype=np.float32))
        return batch


def synthesize_samples(model, processor, out_dir: Path, speaker_embedding,
                       texts=None):
    """Vocode a couple of sentences for a quick listen test (defaults to the
    fixed Tagalog pair; pass texts from the corpus for other languages)."""
    import soundfile as sf
    from transformers import SpeechT5HifiGan

    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(model.device)
    model.eval()
    for i, text in enumerate(texts or SAMPLE_TEXTS):
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            speech = model.generate_speech(
                inputs["input_ids"],
                speaker_embedding.to(model.device),
                vocoder=vocoder,
            )
        path = out_dir / f"sample_{i}.wav"
        sf.write(path, speech.cpu().numpy(), SR)
        print(f"  sample: {path} :: {text}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", choices=["fsc", "livestream", "pld"], default="fsc")
    ap.add_argument("--language", default=None,
                    help="ISO 639-3 language filter (pld only), e.g. bcl, ceb")
    ap.add_argument("--max-samples", type=int, default=3000,
                    help="training clips to use (bounds preprocessing + epoch size)")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-proc", type=int, default=1,
                    help="dataset map workers (keep 1: the mapper holds the GPU embedder)")
    ap.add_argument("--checkpoint", default="microsoft/speecht5_tts",
                    help="base model or a local checkpoint dir to continue from")
    ap.add_argument("--synthesize-only", action="store_true",
                    help="skip training; just vocode samples from --checkpoint")
    ap.add_argument("--push", action="store_true",
                    help="upload the finetuned model to the Hub as "
                         "speecht5_tts-{fsc|halohaloLS} after training")
    args = ap.parse_args()

    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              SpeechT5ForTextToSpeech, SpeechT5Processor)

    from halolib.finetune import load_speech_dataset

    if args.dataset == "pld" and not args.language:
        raise SystemExit("--dataset pld requires --language (e.g. --language bcl)")

    run_name = (f"tts_{args.dataset}_{args.language}" if args.language
                else f"tts_{args.dataset}")
    out_dir = FINETUNE_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained(args.checkpoint)
    model.config.use_cache = False

    print(f"Loading dataset: {args.dataset}"
          + (f" [{args.language}]" if args.language else ""))
    ds = load_speech_dataset(args.dataset, task="tts",
                             max_samples=args.max_samples,
                             token=os.environ.get("HF_TOKEN"),
                             language=args.language)
    print(ds)

    # listen-test sentences in the corpus language (the fixed Tagalog pair
    # would be nonsense for e.g. Kapampangan)
    sample_texts = None
    if args.dataset == "pld":
        sample_texts = [t for raw in ds["test"]["text"]
                        if (t := clean_text(raw)) and len(t) < 120][:2]

    embedder = build_speaker_embedder()

    if args.synthesize_only:
        # mean embedding of a few training clips = a stable "house voice"
        ds_small = ds["train"].select(range(min(16, len(ds["train"]))))
        embs = []
        for ex in ds_small:
            wav = torch.tensor(np.asarray(ex["audio"]["array"], dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                e = embedder.encode_batch(wav)
            embs.append(torch.nn.functional.normalize(e, dim=2).squeeze())
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        synthesize_samples(model, processor, out_dir, torch.stack(embs).mean(0).unsqueeze(0))
        return

    print("Preprocessing (text→ids, audio→mel, clip→x-vector)...")
    ds = prepare_dataset(ds, processor, embedder, args.num_proc)
    print(ds)

    trainer = Seq2SeqTrainer(
        args=Seq2SeqTrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=100,
            max_steps=args.max_steps,
            gradient_checkpointing=True,
            fp16=False,                      # fp16 NaNs on SpeechT5 mel loss
            eval_strategy="steps",
            eval_steps=250,
            save_steps=250,
            save_total_limit=2,
            logging_steps=25,
            report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
            run_name=run_name,
            load_best_model_at_end=True,
            greater_is_better=False,
            label_names=["labels"],
            dataloader_num_workers=2,
            remove_unused_columns=False,
        ),
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=TTSDataCollator(processor, model),
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    print(f"Saved: {out_dir / 'final'}")

    # listen test with the mean speaker embedding of the eval set
    embs = [torch.tensor(np.array(r["speaker_embeddings"], dtype=np.float32)) for r in
            ds["test"].select(range(min(16, len(ds["test"]))))]
    synthesize_samples(model, processor, out_dir,
                       torch.stack(embs).mean(0).unsqueeze(0),
                       texts=sample_texts)

    if args.push:
        from halolib.finetune import push_model_to_hub
        push_kwargs = {}
        if args.dataset == "pld":
            push_kwargs = dict(
                suffix=f"pld-{args.language}",
                lang_code=args.language,
                extra_tags=["philippines", "philippine-languages",
                            args.language],
            )
        push_model_to_hub(
            out_dir / "final", "microsoft/speecht5_tts", args.dataset, "tts",
            token=os.environ.get("HF_TOKEN"),
            metrics={"eval_loss": trainer.state.best_metric}
            if trainer.state.best_metric is not None else None,
            train_summary=(
                f"Trained for {args.max_steps} steps on "
                f"{len(ds['train'])} clips (batch {args.batch_size}×"
                f"{args.grad_accum}, lr {args.lr}, fp32 + gradient "
                f"checkpointing). Synthesized listen-test samples are in "
                f"`samples/` (speechbrain x-vector speaker conditioning + "
                f"`microsoft/speecht5_hifigan` vocoder)."),
            sample_files=sorted(out_dir.glob("sample_*.wav")),
            **push_kwargs,
        )


if __name__ == "__main__":
    main()
