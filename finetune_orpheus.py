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

# word delimiter for the spelled-out frontends, as in finetune_ctc
DELIM = "|"
PUNCT_RE = re.compile(r"[^\w\s']")

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


def cache_path(cache_root, dataset: str, language: str | None, split: str) -> Path:
    return Path(cache_root) / f"snac_{dataset}_{language or 'all'}_{split}.jsonl"


def frontend_text(text: str, units: str) -> str:
    """The text as the model is asked to read it — the R2 question, put to a
    codec LM instead of a CTC head.

    `bpe` is the raw sentence through Llama's tokenizer as shipped, which is
    what Orpheus was pretrained on. The other two spell the sentence out in
    units the tokenizer never saw as units, on the theory that a model with no
    Philippine pretraining does better shown structure than left to infer it.
    Same word-delimiter convention as the CTC arm (finetune_ctc.text_to_units),
    so the two ablations are readable side by side.

    This is a different mechanism from the CTC ablation: there the units were
    the output alphabet, here they are the conditioning text. The CTC result
    (characters beat syllables) therefore does not settle this one.
    """
    if units == "bpe":
        return text
    from halolib.syllables import units as syl_units
    if units == "syllable":
        return " ".join(syl_units(text, english="chars", delim=DELIM))
    # the syllabifier drops punctuation, so the char arm must too: the two
    # spelled-out arms have to differ in units and nothing else
    out = []
    for i, w in enumerate(PUNCT_RE.sub("", text.lower()).split()):
        if i:
            out.append(DELIM)
        out.extend(list(w))
    return " ".join(out)


def build_examples(ds_split, tokenizer, snac_model, device, max_tokens: int,
                   units: str = "bpe", cache: Path | None = None):
    """(audio, text, speaker_id) rows -> input_ids/labels, loss on speech only.

    SNAC encoding is codec inference on the GPU, and it used to run inside
    every training job and die with it: a preempted run re-encoded the whole
    corpus on restart, and each arm of an ablation paid the cost again for the
    same audio. The tokens depend only on the waveform, so cache them keyed by
    (dataset, language, split) and reuse them across arms, runs and languages.

    The cache holds audio tokens and the raw sentence, not the assembled
    sequence: --units changes only the prompt, so all three frontends of the
    ablation share one cache and only the first of them pays for encoding.
    """
    import json

    import librosa

    rows = None
    if cache and cache.exists():
        rows = [json.loads(line) for line in cache.read_text().splitlines()]
        print(f"  snac cache: {cache} ({len(rows)} clips)")

    if rows is None:
        rows, skipped = [], 0
        for row in ds_split:
            text = clean_text(row["text"])
            if text is None:
                skipped += 1
                continue

            audio = row["audio"]
            wav = np.asarray(audio["array"], dtype=np.float32)
            if audio["sampling_rate"] != SNAC_SR:
                # FSC is 16kHz; SNAC needs 24kHz. This adds no content above
                # 8kHz — a data ceiling documented in the pilot plan, not a bug.
                wav = librosa.resample(wav, orig_sr=audio["sampling_rate"],
                                       target_sr=SNAC_SR)

            rows.append({"audio_tokens": encode_audio(snac_model, wav, device),
                         "text": text,
                         "speaker_id": str(row["speaker_id"])})

        print(f"  encoded {len(rows)} clips ({skipped} skipped)")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text("\n".join(json.dumps(r) for r in rows))
            tmp.rename(cache)          # atomic: a killed run leaves no half file
            print(f"  snac cached: {cache}")

    examples, too_long = [], 0
    for r in rows:
        prompt = f"{r['speaker_id']}: {frontend_text(r['text'], units)}"
        text_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        prefix = [SOH] + text_ids + [EOT, EOH, SOAI, SOS]
        input_ids = prefix + r["audio_tokens"] + [EOS_SPEECH, EOAI]
        if len(input_ids) > max_tokens:
            too_long += 1
            continue

        # -100 over the prompt: grade the model on producing speech, not on
        # parroting back the text it was given
        labels = [-100] * len(prefix) + input_ids[len(prefix):]
        # `length` feeds group_by_length, which batches similar-length
        # sequences so a batch is not padded out to its longest member
        examples.append({"input_ids": input_ids, "labels": labels,
                         "length": len(input_ids)})

    print(f"  {len(examples)} examples, units={units} "
          f"({too_long} over {max_tokens} tokens)")
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
    ap.add_argument("--units", choices=["bpe", "syllable", "char"],
                    default="bpe",
                    help="text frontend: the P1 ablation in "
                         "docs/tts_sota_plan.md. bpe is Orpheus as pretrained")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=1500)
    # Defaults below are the 8 GB recipe. On the 96 GB RTX PRO 6000 use
    # --cloud, which sets the batch, precision and optimiser in one flag.
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--full", action="store_true",
                    help="full bf16 finetune instead of LoRA: ~53 GB of "
                         "optimiser state, so a big card only")
    ap.add_argument("--no-quant", action="store_true",
                    help="load the base in bf16 rather than 4-bit NF4. "
                         "Quantisation is a memory compromise, and on a card "
                         "with spare memory it only makes the matmuls slower")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="stop recomputing activations; ~25%% faster when the "
                         "memory is there")
    ap.add_argument("--cloud", action="store_true",
                    help="one flag for the RTX PRO 6000: no quantisation, no "
                         "checkpointing, batch 8, fused optimiser, LoRA r=64")
    ap.add_argument("--snac-cache", default=os.environ.get(
        "SNAC_CACHE", str(Path(os.environ.get("PLD_WORK_DIR", ".")) / "snac_cache")),
        help="where encoded SNAC tokens are reused across runs and arms")
    ap.add_argument("--dataloader-workers", type=int, default=2)
    ap.add_argument("--profile", action="store_true",
                    help="profile ~10 steps and stop; trains nothing")
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
    if args.cloud:
        # the 8 GB defaults cost the ASR bake-off half its throughput before
        # they were found; make the big-card recipe a single flag instead of
        # six that have to be remembered together
        args.no_quant = True
        args.no_grad_checkpoint = True
        if args.batch_size == 1:
            args.batch_size, args.grad_accum = 8, 1
        if args.lora_rank == 16 and not args.full:
            args.lora_rank = 64
        if args.dataloader_workers == 2:
            args.dataloader_workers = 8

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    from halolib.finetune import load_speech_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.dataset + (f"_{args.language}" if args.language else "")
    tag = f"{args.units}_{tag}"
    out_dir = FINETUNE_DIR / f"orpheus_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    quantize = not (args.no_quant or args.full)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if quantize else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quant,
        dtype=torch.bfloat16, device_map={"": 0} if device == "cuda" else None,
        attn_implementation="sdpa",
    )
    print(f"  base: {'4-bit NF4' if quantize else 'bf16'}, "
          f"{'full finetune' if args.full else f'LoRA r={args.lora_rank}'}, "
          f"batch {args.batch_size}x{args.grad_accum}, "
          f"checkpointing {'off' if args.no_grad_checkpoint else 'on'}")
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
    train_ex = build_examples(
        ds["train"], tokenizer, snac_model, device, args.max_tokens,
        units=args.units,
        cache=cache_path(args.snac_cache, args.dataset, args.language, "train"))
    eval_ex = build_examples(
        ds["test"], tokenizer, snac_model, device, args.max_tokens,
        units=args.units,
        cache=cache_path(args.snac_cache, args.dataset, args.language, "test"))
    if not train_ex:
        raise SystemExit("no training examples survived filtering")
    # the codec is only needed for encoding and for the listen test; freeing it
    # returns ~1 GB and stops it holding fragments of the allocator
    del snac_model
    if device == "cuda":
        torch.cuda.empty_cache()
    snac_model = None

    if args.full:
        # Nothing to graft on: train the checkpoint itself. Embeddings stay
        # trainable here (unlike the LoRA path) because the audio tokens are
        # what we are teaching, and at bf16 the 157k-token vocab is affordable.
        # checkpointing is set from TrainingArguments below
        print(f"  full finetune: "
              f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    else:
        if quantize:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=not args.no_grad_checkpoint)
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
            gradient_checkpointing=not args.no_grad_checkpoint,
            bf16=True,
            # paged 8-bit Adam exists to survive an 8 GB card; with memory to
            # spare, fused AdamW is both faster and exact
            optim=("adamw_torch_fused" if args.no_quant or args.full
                   else "paged_adamw_8bit"),
            # audio token sequences run 300-1400 long, so padding to the
            # longest in a random batch wastes a large fraction of every step
            group_by_length=True,
            length_column_name="length",
            eval_strategy="steps",
            eval_steps=250,
            save_steps=250,
            save_total_limit=2,
            logging_steps=10,
            report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
            run_name=f"orpheus_{tag}",
            load_best_model_at_end=True,
            greater_is_better=False,
            label_names=["labels"],
            dataloader_num_workers=args.dataloader_workers,
            remove_unused_columns=False,
        ),
        model=model,
        train_dataset=train_ex,
        eval_dataset=eval_ex,
        data_collator=OrpheusCollator(tokenizer.pad_token_id or 128263),
    )

    if args.profile:
        from halolib.profiling import profiler_callback
        trainer.add_callback(profiler_callback(out_dir))
        trainer.train()
        return

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"Saved: {out_dir / 'final'}")

    voice = args.voice or str(ds["train"][0]["speaker_id"])
    snac_model = load_snac(device)      # freed after encoding; needed again here
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
