"""
Speech-to-speech finetuning — SpeechT5 voice conversion on PLD, all languages.

PLD has no parallel translations, so cross-lingual S2S translation is not
trainable from it. What it does have — uniquely — is the same prompt list read
by many speakers per language, which is exactly the parallel data voice
conversion needs. This script mines (source speaker, target speaker) pairs of
the same sentence and finetunes `microsoft/speecht5_vc` (SpeechT5's
speech-to-speech configuration) across all ten languages at once, conditioning
on the target clip's x-vector.

The result is one multilingual any-to-any voice conversion model:
audio in any of the 10 languages → same utterance in a target speaker's voice.

Usage:
  python finetune_s2s.py                       # all languages, then --push
  python finetune_s2s.py --languages bcl ceb   # subset
  python finetune_s2s.py --pairs-per-lang 200 --max-steps 500   # quick run

Outputs land in {FINETUNE_DIR}/s2s_pld/ (checkpoints + converted samples).
"""

import argparse
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
SR = 16000
MAX_INPUT_SECS = 12.0     # encoder memory cap on an 8GB card
MAX_LABEL_FRAMES = 960    # ~15.5s of mel

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def sentence_key(text: str) -> str:
    """Normalize a prompt so the same sentence matches across session logs."""
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(_PUNCT_RE.sub(" ", text).split())


def mine_pairs(root: Path, languages: list[str] | None,
               pairs_per_lang: int, seed: int = 42) -> list[dict]:
    """Same-sentence cross-speaker pairs from the PLD read-speech portion.

    Grouping key is (language, normalized sentence); every group read by two
    or more speakers yields pairs. Round-robin over sentences so no single
    prompt list dominates, capped per language so Bikol (95h) cannot drown
    Tausug (6h).
    """
    import random

    from halolib.pld import index_corpus

    entries, _ = index_corpus(root)
    groups = defaultdict(list)
    for e in entries:
        if (e["speech_type"] != "read" or e["text_is_prompt"]
                or e["num_words"] < 3):
            continue
        if languages and e["language"] not in languages:
            continue
        groups[(e["language"], sentence_key(e["sentence"]))].append(e)

    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for (lang, _key), clips in groups.items():
        speakers = defaultdict(list)
        for c in clips:
            speakers[c["speaker_id"]].append(c)
        if len(speakers) < 2:
            continue
        sp = rng.sample(sorted(speakers), 2)
        src = rng.choice(speakers[sp[0]])
        tgt = rng.choice(speakers[sp[1]])
        by_lang[lang].append({
            "src_path": str(src["wav_path"]),
            "tgt_path": str(tgt["wav_path"]),
            "language": lang,
            "sentence": src["sentence"],
            "src_speaker": src["speaker_id"],
            "tgt_speaker": tgt["speaker_id"],
        })

    pairs = []
    for lang, lst in sorted(by_lang.items()):
        rng.shuffle(lst)
        pairs.extend(lst[:pairs_per_lang])
        print(f"  {lang}: {min(len(lst), pairs_per_lang)} pairs "
              f"(of {len(lst)} candidate sentences)")
    rng.shuffle(pairs)
    return pairs


def build_s2s_model_class():
    """SpeechT5ForSpeechToSpeech with a training loss.

    HF implements the L1+BCE spectrogram loss only on the TTS head; the
    speech-to-speech head returns `loss=None` unconditionally, which makes it
    generate-only out of the box. This subclass reuses the library's own
    `SpeechT5SpectrogramLoss` on the S2S forward. Guided attention loss stays
    off: its mask is in input tokens, and the speech encoder's conv prenet
    downsamples waveforms so the lengths no longer correspond.
    """
    from transformers import SpeechT5ForSpeechToSpeech
    from transformers.models.speecht5.modeling_speecht5 import (
        Seq2SeqSpectrogramOutput, SpeechT5SpectrogramLoss,
        shift_spectrograms_right)

    class SpeechT5ForSpeechToSpeechWithLoss(SpeechT5ForSpeechToSpeech):
        def forward(self, input_values=None, attention_mask=None,
                    decoder_input_values=None, decoder_attention_mask=None,
                    speaker_embeddings=None, labels=None, **kwargs):
            if labels is not None and decoder_input_values is None:
                decoder_input_values, decoder_attention_mask = \
                    shift_spectrograms_right(
                        labels, self.config.reduction_factor,
                        decoder_attention_mask)

            outputs = self.speecht5(
                input_values=input_values,
                attention_mask=attention_mask,
                decoder_input_values=decoder_input_values,
                decoder_attention_mask=decoder_attention_mask,
                speaker_embeddings=speaker_embeddings,
                use_cache=False,
                return_dict=True,
            )
            before_postnet, after_postnet, logits = \
                self.speech_decoder_postnet(outputs[0])

            loss = None
            if labels is not None:
                criterion = SpeechT5SpectrogramLoss(self.config)
                loss = criterion(attention_mask, before_postnet,
                                 after_postnet, logits, labels)

            return Seq2SeqSpectrogramOutput(
                loss=loss,
                spectrogram=after_postnet,
                past_key_values=outputs.past_key_values,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            )

    return SpeechT5ForSpeechToSpeechWithLoss


def build_speaker_embedder():
    from speechbrain.inference.speaker import EncoderClassifier
    savedir = Path(os.environ.get("HF_HOME", "~/.cache")).expanduser() / "speechbrain-xvect"
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=str(savedir),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )


def prepare_dataset(ds, processor, embedder, num_proc: int):
    """(src wav, tgt wav) → input_values, labels (tgt mel), tgt x-vector."""

    def _process(batch):
        src = batch["src_audio"]["array"]
        tgt = batch["tgt_audio"]["array"]
        example = processor(
            audio=src, audio_target=tgt, sampling_rate=SR,
            return_attention_mask=False,
        )
        with torch.no_grad():
            wav = torch.tensor(np.asarray(tgt, dtype=np.float32)).unsqueeze(0)
            emb = embedder.encode_batch(wav)
            emb = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu().numpy()
        return {
            "input_values": example["input_values"][0],
            "labels": example["labels"][0],
            "speaker_embeddings": emb,
        }

    cols = ds["train"].column_names
    ds = ds.map(_process, remove_columns=cols, num_proc=num_proc)
    ds = ds.filter(lambda r: (len(r["input_values"]) <= int(MAX_INPUT_SECS * SR)
                              and len(r["labels"]) <= MAX_LABEL_FRAMES))
    return ds


class S2SDataCollator:
    """Pad waveform inputs and mel targets; mask padding with -100 and trim to
    the reduction factor — the speech-in analogue of the TTS collator."""

    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def __call__(self, features):
        input_values = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_values": f["labels"]} for f in features]
        speaker_features = [f["speaker_embeddings"] for f in features]

        batch = self.processor.pad(
            input_values=input_values, labels=label_features,
            return_tensors="pt")

        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100).float()
        del batch["decoder_attention_mask"]

        if self.model.config.reduction_factor > 1:
            lengths = torch.tensor([len(f["input_values"]) for f in label_features])
            target = (lengths.max() // self.model.config.reduction_factor
                      ) * self.model.config.reduction_factor
            batch["labels"] = batch["labels"][:, :target]

        batch["input_values"] = batch["input_values"].float()
        batch["speaker_embeddings"] = torch.tensor(
            np.array(speaker_features, dtype=np.float32))
        return batch


def convert_samples(model, processor, out_dir: Path, raw_test, embedder, n=2):
    """Run held-out pairs through the model: source audio → target voice."""
    import soundfile as sf
    from transformers import SpeechT5HifiGan

    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan").to(model.device)
    model.eval()
    for i in range(min(n, len(raw_test))):
        row = raw_test[i]
        src = np.asarray(row["src_audio"]["array"], dtype=np.float32)
        tgt = np.asarray(row["tgt_audio"]["array"], dtype=np.float32)
        inputs = processor(audio=src, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            emb = embedder.encode_batch(torch.tensor(tgt).unsqueeze(0))
            emb = torch.nn.functional.normalize(emb, dim=2).squeeze(0).cpu()
            speech = model.generate_speech(
                inputs["input_values"].to(model.device),
                emb.to(model.device),
                vocoder=vocoder,
            )
        sf.write(out_dir / f"sample_{i}_src.wav", src, SR)
        sf.write(out_dir / f"sample_{i}_converted.wav", speech.cpu().numpy(), SR)
        sf.write(out_dir / f"sample_{i}_target_ref.wav", tgt, SR)
        print(f"  sample {i} [{row['language']}] {row['src_speaker']} → "
              f"{row['tgt_speaker']} :: {row['sentence'][:60]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--languages", nargs="*", default=None,
                    help="ISO 639-3 subset (default: all)")
    ap.add_argument("--pairs-per-lang", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-proc", type=int, default=1)
    ap.add_argument("--checkpoint", default="microsoft/speecht5_vc")
    ap.add_argument("--push", action="store_true",
                    help="upload to the Hub as speecht5_vc-pld after training")
    args = ap.parse_args()

    from datasets import Audio, Dataset, DatasetDict
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              SpeechT5Processor)

    out_dir = FINETUNE_DIR / "s2s_pld"
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(os.environ.get(
        "PLD_RAW", "/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD"))

    print("Mining same-sentence cross-speaker pairs...")
    pairs = mine_pairs(root, args.languages, args.pairs_per_lang)
    print(f"Total pairs: {len(pairs)}")

    ds = Dataset.from_list(pairs)
    ds = ds.rename_column("src_path", "src_audio")
    ds = ds.rename_column("tgt_path", "tgt_audio")
    ds = ds.cast_column("src_audio", Audio(sampling_rate=SR))
    ds = ds.cast_column("tgt_audio", Audio(sampling_rate=SR))
    n_test = max(20, len(ds) // 20)
    ds = DatasetDict({"train": ds.select(range(n_test, len(ds))),
                      "test": ds.select(range(n_test))})
    print(ds)
    raw_test = ds["test"]

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    model_cls = build_s2s_model_class()
    model = model_cls.from_pretrained(args.checkpoint)
    model.config.use_cache = False
    model.config.use_guided_attention_loss = False

    embedder = build_speaker_embedder()

    print("Preprocessing (wav→features, tgt→mel, tgt→x-vector)...")
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
            fp16=False,                      # same NaN risk as TTS mel loss
            eval_strategy="steps",
            eval_steps=250,
            save_steps=250,
            save_total_limit=2,
            logging_steps=25,
            report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
            run_name="s2s_pld",
            load_best_model_at_end=True,
            greater_is_better=False,
            label_names=["labels"],
            dataloader_num_workers=2,
            remove_unused_columns=False,
        ),
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=S2SDataCollator(processor, model),
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    print(f"Saved: {out_dir / 'final'}")

    convert_samples(model, processor, out_dir, raw_test, embedder)

    if args.push:
        from halolib.finetune import push_model_to_hub
        langs = sorted({p["language"] for p in pairs})
        push_model_to_hub(
            out_dir / "final", "microsoft/speecht5_vc", "pld", "s2s",
            token=os.environ.get("HF_TOKEN"),
            metrics={"eval_loss": trainer.state.best_metric}
            if trainer.state.best_metric is not None else None,
            train_summary=(
                f"Any-to-any voice conversion across {len(langs)} Philippine "
                f"languages ({', '.join(langs)}). Trained for "
                f"{args.max_steps} steps on {len(ds['train'])} same-sentence "
                f"cross-speaker pairs mined from PLD's shared prompt lists "
                f"(batch {args.batch_size}×{args.grad_accum}, lr {args.lr}, "
                f"fp32 + gradient checkpointing), conditioned on the target "
                f"clip's speechbrain x-vector. `samples/` holds held-out "
                f"conversions: `*_src` → `*_converted` vs `*_target_ref`."),
            sample_files=sorted(out_dir.glob("sample_*.wav")),
            suffix="pld",
            lang_code="multilingual",
            extra_tags=["philippines", "philippine-languages",
                        "voice-conversion", "speech-to-speech", *langs],
        )


if __name__ == "__main__":
    main()
