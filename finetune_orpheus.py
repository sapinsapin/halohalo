"""
TTS finetuning — QLoRA on a 3B Orpheus-class codec-LM (fits an 8GB card).

The home-scale pilot for the halo-tts track; see docs/orpheus_pilot_plan.md
for the rationale, VRAM budget, and success criteria.

Pipeline: speech -> SNAC 24kHz codec tokens -> flattened into the base model's
audio-token id space -> causal LM training of LoRA adapters on the audio span.

Why QLoRA and not a full finetune: 3.3B params in full finetune needs ~40GB.
4-bit base + LoRA on the projections is ~5-6GB. Notably we do NOT set
modules_to_save=["lm_head","embed_tokens"] the way Canopy's reference lora.py
does — with a 156,940-token vocab that trains 482M params at full precision
and OOMs instantly here. It is unnecessary anyway: the audio tokens already
exist in the base checkpoint and Llama's BPE covers Filipino without new
tokens.

Usage:
  python finetune_orpheus.py --dataset fsc --max-samples 2000 --max-steps 1500
  python finetune_orpheus.py --dataset fsc --synthesize-only --checkpoint <dir>
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

# Unsloth's ungated mirror of canopylabs/orpheus-3b-0.1-pretrained (the
# official repo is gated behind a click-through licence). Same weights:
# llama arch, 28 layers, hidden 3072, vocab 156939 — and that vocab size is
# exactly AUDIO_BASE + 7*4096 + 1, i.e. sized to the audio-token layout below.
BASE_MODEL = "unsloth/orpheus-3b-0.1-pretrained"
SNAC_MODEL = "hubertsiuzdak/snac_24khz"
SNAC_SR = 24000

# Sequence framing, verified against Canopy's inference code (engine_class.py
# builds exactly SOH + text + EOT + EOH + SOAI + SOS before generation).
SOH, EOT, EOH = 128259, 128009, 128260      # start/end of the "human" turn
SOAI, SOS = 128261, 128257                  # start of AI turn, start of speech
EOS_SPEECH, EOAI = 128258, 128262           # end of speech, end of AI turn

# decoder.py recovers a code with  int(n) - 10 - (index % 7) * 4096  where the
# token id is 128256 + n; inverted, that is the offset below.
AUDIO_BASE = 128266
CODEBOOK_STRIDE = 4096

DIGIT_RE = re.compile(r"\d")

SAMPLE_TEXTS = [
    "Magandang umaga po sa inyong lahat.",
    "Salamat sa pakikinig, hanggang sa muli.",
]


def clean_text(text: str) -> str | None:
    """Reject digit-bearing text rather than teach numeral skipping (the
    verbalization gap is unresolved — see docs/livestream_pipeline.md)."""
    text = text.replace("’", "'").replace("‘", "'").strip()
    if not text or DIGIT_RE.search(text):
        return None
    return text


def load_snac(device: str):
    from snac import SNAC
    return SNAC.from_pretrained(SNAC_MODEL).eval().to(device)


def encode_audio(snac_model, wav: np.ndarray, device: str) -> list[int]:
    """Waveform @24kHz -> flat audio token ids.

    SNAC returns three hierarchical codebooks (rates 1x, 2x, 4x). Orpheus
    flattens each frame to 7 tokens in this exact interleave, and offsets each
    slot into its own 4096-wide band so position is unambiguous:
        [c0[j], c1[2j], c2[4j], c2[4j+1], c1[2j+1], c2[4j+2], c2[4j+3]]
    """
    with torch.inference_mode():
        x = torch.tensor(wav, dtype=torch.float32, device=device)[None, None, :]
        codes = snac_model.encode(x)

    c0, c1, c2 = (c[0].tolist() for c in codes)
    tokens = []
    for j in range(len(c0)):
        tokens += [
            c0[j] + AUDIO_BASE,
            c1[2 * j] + AUDIO_BASE + CODEBOOK_STRIDE,
            c2[4 * j] + AUDIO_BASE + 2 * CODEBOOK_STRIDE,
            c2[4 * j + 1] + AUDIO_BASE + 3 * CODEBOOK_STRIDE,
            c1[2 * j + 1] + AUDIO_BASE + 4 * CODEBOOK_STRIDE,
            c2[4 * j + 2] + AUDIO_BASE + 5 * CODEBOOK_STRIDE,
            c2[4 * j + 3] + AUDIO_BASE + 6 * CODEBOOK_STRIDE,
        ]
    return tokens


def decode_tokens(snac_model, tokens: list[int], device: str) -> np.ndarray:
    """Inverse of encode_audio: flat token ids -> waveform @24kHz.

    Verified as an exact inverse at the code level (encode->decode returns the
    identical three codebooks). Note SNAC's decoder is stochastic — decoding
    the same codes twice differs by ~0.5 peak — so compare codes, not
    waveforms, when testing this path.
    """
    tokens = [t for t in tokens if t >= AUDIO_BASE]
    n_frames = len(tokens) // 7
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    c0, c1, c2 = [], [], []
    for j in range(n_frames):
        f = tokens[7 * j:7 * j + 7]
        # undo the per-slot band offset, then clamp: a sampled token can land
        # outside its band, and SNAC would index out of range
        v = [(t - AUDIO_BASE - i * CODEBOOK_STRIDE) % CODEBOOK_STRIDE
             for i, t in enumerate(f)]
        c0.append(v[0])
        c1 += [v[1], v[4]]
        c2 += [v[2], v[3], v[5], v[6]]

    codes = [torch.tensor(c, dtype=torch.int32, device=device)[None]
             for c in (c0, c1, c2)]
    with torch.inference_mode():
        audio = snac_model.decode(codes)
    return audio[0, 0].cpu().numpy()


def build_examples(ds_split, tokenizer, snac_model, device, max_tokens: int):
    """(audio, text, speaker_id) rows -> input_ids/labels, loss on speech only."""
    import librosa

    examples = []
    skipped = 0
    for row in ds_split:
        text = clean_text(row["text"])
        if text is None:
            skipped += 1
            continue

        audio = row["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        if audio["sampling_rate"] != SNAC_SR:
            # FSC is 16kHz; SNAC needs 24kHz. This adds no content above 8kHz
            # — a data ceiling documented in the pilot plan, not a bug.
            wav = librosa.resample(wav, orig_sr=audio["sampling_rate"],
                                   target_sr=SNAC_SR)

        audio_tokens = encode_audio(snac_model, wav, device)

        prompt = f"{row['speaker_id']}: {text}"
        text_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        prefix = [SOH] + text_ids + [EOT, EOH, SOAI, SOS]
        input_ids = prefix + audio_tokens + [EOS_SPEECH, EOAI]

        if len(input_ids) > max_tokens:
            skipped += 1
            continue

        # -100 over the prompt: grade the model on producing speech, not on
        # parroting back the text it was given
        labels = [-100] * len(prefix) + input_ids[len(prefix):]
        examples.append({"input_ids": input_ids, "labels": labels})

    print(f"  built {len(examples)} examples ({skipped} skipped)")
    return examples


class OrpheusCollator:
    """Right-pad a batch; padded positions are masked out of the loss."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        n = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def synthesize_samples(model, tokenizer, snac_model, out_dir: Path, voice: str,
                       device: str, texts=None):
    """Generate speech for a few sentences as a listen test."""
    import soundfile as sf

    model.eval()
    for i, text in enumerate(texts or SAMPLE_TEXTS):
        ids = tokenizer(f"{voice}: {text}", add_special_tokens=False).input_ids
        prompt = torch.tensor([[SOH] + ids + [EOT, EOH, SOAI, SOS]],
                              device=device)
        with torch.inference_mode():
            out = model.generate(
                prompt,
                max_new_tokens=1400,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=EOS_SPEECH,
                pad_token_id=tokenizer.pad_token_id or 128263,
            )
        gen = out[0, prompt.shape[1]:].tolist()
        wav = decode_tokens(snac_model, gen, device)
        path = out_dir / f"sample_{i}.wav"
        sf.write(path, wav, SNAC_SR)
        print(f"  sample: {path} ({len(wav) / SNAC_SR:.1f}s) :: {text}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", choices=["fsc", "livestream", "pld"],
                    default="fsc")
    ap.add_argument("--language", default=None, help="ISO 639-3 filter (pld)")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1408,
                    help="sequence cap; 87.5 audio tokens per second of speech")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--checkpoint", default=None,
                    help="adapter dir to resume from or synthesize with")
    ap.add_argument("--synthesize-only", action="store_true")
    ap.add_argument("--voice", default=None,
                    help="speaker_id to condition synthesis on")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    from halolib.finetune import load_speech_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.dataset + (f"_{args.language}" if args.language else "")
    out_dir = FINETUNE_DIR / f"orpheus_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quant,
        dtype=torch.bfloat16, device_map={"": 0} if device == "cuda" else None,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    snac_model = load_snac(device)

    if args.synthesize_only:
        if args.checkpoint:
            model = PeftModel.from_pretrained(model, args.checkpoint)
        synthesize_samples(model, tokenizer, snac_model, out_dir,
                           args.voice or "1", device)
        return

    print(f"Loading dataset: {args.dataset}")
    ds = load_speech_dataset(args.dataset, task="tts",
                             max_samples=args.max_samples,
                             token=os.environ.get("HF_TOKEN"),
                             language=args.language)
    print(ds)

    print("Encoding audio to SNAC tokens...")
    train_ex = build_examples(ds["train"], tokenizer, snac_model, device,
                              args.max_tokens)
    eval_ex = build_examples(ds["test"], tokenizer, snac_model, device,
                             args.max_tokens)
    if not train_ex:
        raise SystemExit("no training examples survived filtering")

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        args=TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            max_steps=args.max_steps,
            gradient_checkpointing=True,
            bf16=True,
            optim="paged_adamw_8bit",
            eval_strategy="steps",
            eval_steps=250,
            save_steps=250,
            save_total_limit=2,
            logging_steps=10,
            report_to=[],
            load_best_model_at_end=True,
            greater_is_better=False,
            label_names=["labels"],
            dataloader_num_workers=2,
            remove_unused_columns=False,
        ),
        model=model,
        train_dataset=train_ex,
        eval_dataset=eval_ex,
        data_collator=OrpheusCollator(tokenizer.pad_token_id or 128263),
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"Saved: {out_dir / 'final'}")

    voice = args.voice or str(ds["train"][0]["speaker_id"])
    synthesize_samples(model, tokenizer, snac_model, out_dir, voice, device)

    if args.push:
        from halolib.finetune import push_model_to_hub
        push_model_to_hub(
            out_dir / "final", args.base_model, args.dataset, "tts",
            token=os.environ.get("HF_TOKEN"),
            metrics={"eval_loss": trainer.state.best_metric}
            if trainer.state.best_metric is not None else None,
            suffix=f"{tag}-qlora",
            train_summary=(
                f"QLoRA (r={args.lora_rank}, 4-bit NF4 base) for "
                f"{args.max_steps} steps on {len(train_ex)} clips, single "
                f"RTX 3070. Speech is tokenized with SNAC 24kHz; source audio "
                f"is 16kHz upsampled, so output is band-limited above 8kHz — "
                f"a property of the corpus, documented in "
                f"docs/orpheus_pilot_plan.md. Adapters only: load on top of "
                f"`{args.base_model}`."),
            sample_files=sorted(out_dir.glob("sample_*.wav")),
            license="apache-2.0",
        )


if __name__ == "__main__":
    main()
