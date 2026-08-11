# halo-tts pilot — QLoRA on a 3B Orpheus codec-LM (single RTX 3070)

The home-scale preview of the `halo-tts` track in
[`sota_speech_plan.md`](sota_speech_plan.md). Goal: find out what an
Orpheus-class codec-LM sounds like on Filipino **before** committing cloud
budget to the full run — and produce a real, publishable model on the way.

Not a substitute for the cluster run. This is the go/no-go evidence for it.

## Why this fits on 8GB when full finetuning does not

Full finetuning 3.3B params needs ~40GB (weights + Adam states + activations).
QLoRA changes the arithmetic:

| Component | Choice | VRAM |
|---|---|---|
| Base weights | NF4 4-bit quantized, frozen | ~2.2 GB |
| Trainable params | LoRA r=16 on q/k/v/o + gate/up/down (~25M, 0.7% of model) | ~50 MB |
| Optimizer | paged AdamW 8-bit over LoRA params only | ~200 MB |
| Activations | batch 1 × grad-accum 16, gradient checkpointing, seq ≤ 1408 | ~1.5–2.5 GB |
| **Total** | | **~5–6 GB** |

The one trap: Canopy's own `finetune/lora.py` sets
`modules_to_save=["lm_head", "embed_tokens"]`. With a 156,940-token vocab that
is 482M params trained at full precision — ~4GB of optimizer state on its own,
instantly OOM on this card. We omit it. That is safe here because we are not
adding vocabulary: the audio tokens already exist in the base model, and Llama's
BPE covers Filipino text (Latin script) without new tokens.

## Model and data

- **Base**: `canopylabs/orpheus-3b-0.1-pretrained` — the *pretrained* checkpoint,
  not `-ft`. The `-ft` model is tuned to specific English voices (tara, leah…);
  for a new language the un-voice-tuned base adapts more cleanly. No Filipino
  checkpoint exists in Canopy's multilingual research release (de/es/it/fr/ko/zh/hi
  only) — this pilot fills that gap.
- **Codec**: `hubertsiuzdak/snac_24khz`, 3 hierarchical codebooks, 7 tokens per
  frame at 12.5 Hz ⇒ **87.5 audio tokens per second** of speech.
- **Data**: `sapinsapin/filipinospeechcorpus`, read speech, 1–13 s, ≥3 words,
  digit-free (the numeral-verbalization gap is unresolved — see
  `livestream_pipeline.md`), reusing the existing `halolib.finetune` adapter.

### Sequence format (verified against Canopy's inference code, not guessed)

```
[128259] text_ids [128009] [128260] [128261] [128257] audio_tokens [128258] [128262]
  SOH                EOT     EOH      SOAI     SOS                   EOS     EOAI
```

Text is `"{speaker_id}: {sentence}"`, so speaker identity is learned as a voice
name and can be selected at inference. Loss is masked to the audio span: the
model is graded on generating speech, not on echoing the prompt.

Audio token ids come from SNAC codes by
`token = code + 128266 + (position_in_frame × 4096)`, with each frame flattened
as `[c0[j], c1[2j], c2[4j], c2[4j+1], c1[2j+1], c2[4j+2], c2[4j+3]]` — the exact
inverse of `decoder.py`'s `int(n) - 10 - ((index % 7) * 4096)`.

## Known quality ceiling: 16 kHz source into a 24 kHz codec

FSC is 16 kHz natively (as published, and as the raw corpus was processed).
SNAC operates at 24 kHz, so clips are resampled up. Resampling adds no content
above 8 kHz, so synthesis will sound band-limited next to a natively-24 kHz
model — sibilance and "air" will be missing. **This is a property of the data,
not of the method**, and it must not be read as a verdict on the architecture.

The cluster plan already addresses it: `halo-livestream` exports a 24 kHz TTS
set, and future crawled data will be captured at 24 kHz+. A second pilot on
24 kHz livestream data, once that corpus is large enough, isolates this
variable.

## Run configuration

| | |
|---|---|
| Clips | 2,000 (≈3–4 h of speech after filtering) |
| Steps | 1,500 at effective batch 16 (~12 epochs) |
| LR | 2e-4, cosine, 100 warmup (LoRA tolerates ~10× full-finetune LR) |
| Precision | 4-bit NF4 base, bf16 compute, fp32 LoRA master weights |
| Eval | every 250 steps on held-out; sample synthesis at the end |
| Wall clock | ~8–14 h on the 3070 (est. 20–35 s/step) |

## Success criteria

Judged against `sapinsapin/speecht5_tts-fsc`, our current best, on the same
sentences:

1. **Intelligibility** — whisper-large-v3 re-transcription CER of synthesized
   speech beats SpeechT5's. Primary automated gate.
2. **Naturalness** — native-listener preference, informal panel (n≈5) at pilot
   scale. The formal MOS panel belongs to the cluster run.
3. **Voice control** — different `speaker_id` prompts produce audibly different
   voices. Tests whether cloning survives a small-data adaptation.
4. **Taglish** — code-switched sentences don't degrade into English phonology.

**Go for the cluster run** if the codec-LM wins on 1 and 3 even at this scale.
**Reconsider** (favor the flow-matching secondary track) if it is unstable or
loses to a 144M SpeechT5 finetune — which would suggest the architecture needs
far more data than the roadmap assumes.

## Deliverable

`sapinsapin/orpheus-3b-fsc` — LoRA adapters plus merged weights, model card
carrying the eval table, the 16 kHz caveat, and the training config, published
through the existing `push_model_to_hub()` path so it appears on the dashboard
automatically.

## Scheduling note

The GPU is shared with the PLD ASR fleet (`run_pld_asr.sh`, sequential
per-language whisper-small runs). This pilot **queues behind it** rather than
competing for VRAM — two jobs on an 8GB card OOMs both.
