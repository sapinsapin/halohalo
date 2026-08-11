# Methods and toolkits

Techniques that hold up when you have hundreds of hours instead of hundreds of
thousands — and the software that implements them.

---

## Methods

### Transfer from a multilingual pretrained model

Never train from scratch. Start from a checkpoint that has already seen many
languages (XLS-R, MMS, Whisper, SeamlessM4T for speech; a multilingual
codec-LM for TTS) and continue training on the target language. Cross-lingual
transfer is the single biggest lever in low-resource speech, and it is why our
plan is written as *continued pretraining*, not pretraining.

For Philippine languages there's an extra prior worth exploiting: they are
Austronesian, and share structure with Indonesian and Malay — languages with
far more data. Related-language pretraining or joint training is a cheap
experiment with real upside.

### Pseudo-labeling with quality gates

Transcribe unlabeled audio with the best available ASR, filter aggressively,
train on what survives. The filter is the whole method — ours is round-trip
CER plus VAD and overlap detection in the livestream `qc` stage. Without a
gate this degrades into training on your own errors.

Hold pseudo-labeled data out of evaluation entirely, and spot-check ~1% by
hand per batch. Cheap insurance against silent drift.

### Forced alignment for long-audio mining

The unlock for archive-scale data: take hours-long recordings with approximate
transcripts (broadcasts, parliament, streams, audiobooks) and recover
segment-level pairs. MMS CTC alignment does this at scale for many languages;
Montreal Forced Aligner and WhisperX are the other common routes. This is what
turns "we have a licence to some radio archives" into training data.

### Morphology-aware subword units

Philippine languages build words by affixation, infixation, and reduplication.
Naive BPE fragments them inconsistently, which hurts both ASR language models
and TTS text frontends. Unsupervised morphological segmentation (Morfessor,
FlatCat — see Aalto in [research-groups.md](research-groups.md)) is the
established alternative for morphologically rich languages. Worth an ablation
before we commit a tokenizer for the cluster run.

### Parameter-efficient finetuning (LoRA / QLoRA)

4-bit base weights plus small trainable adapters put 3B-class models on a
single consumer GPU. This is what makes the Orpheus pilot possible on an 8GB
card at all, and more generally it's how you run ten cheap experiments instead
of one expensive one. Adapters are also easy to publish and compose.

### Codec-LM TTS

Modern open TTS: encode speech to discrete tokens with a neural codec (SNAC,
DAC, Mimi), then model those tokens with a decoder-only transformer
conditioned on text. Gives expressive prosody and zero-shot voice cloning,
scales with data, and — importantly for us — supports continued pretraining
from a multilingual checkpoint. The non-autoregressive flow-matching family
(F5-TTS and kin) is the robust fallback: lower ceiling, fewer surprises.

### Text normalization

Numbers, dates, currency, abbreviations, and code-switched English inside
Filipino text. Our current TTS sidesteps this by rejecting digit-bearing
sentences — an honest workaround, not a solution. A real TN frontend is a
prerequisite for anything user-facing, and it's language-specific work nobody
else will do for Filipino.

### Data augmentation

Speed and tempo perturbation, SpecAugment, noise and reverberation for ASR
robustness; voice conversion to multiply speaker diversity for TTS when the
speaker count is small (a real constraint in PLD's smaller languages).

---

## Toolkits

| Tool | Use it for |
|---|---|
| **HuggingFace `transformers` + `datasets`** | Our default. Everything in this repo builds on it. |
| **PEFT / bitsandbytes** | LoRA/QLoRA adapters and 4-bit quantized training. |
| **ESPnet** | End-to-end speech research toolkit with strong recipes for ASR/TTS/ST; the reference implementations many papers use. |
| **SpeechBrain** | Cleaner, more readable PyTorch speech toolkit. Source of the x-vector speaker encoder in our TTS pipeline. |
| **NVIDIA NeMo** | Production-scale speech training, good multi-GPU story. Worth evaluating for the cluster runs. |
| **Kaldi / k2 + icefall** | Classical and next-gen WFST-based ASR. Still competitive with limited data, and much of the low-resource literature is written in it. |
| **fairseq / fairseq2** | Reference implementations for wav2vec2, XLS-R, MMS, SeamlessM4T. |
| **Montreal Forced Aligner, WhisperX, pyannote** | Alignment and diarization. |
| **torchtitan / FSDP / DeepSpeed** | Multi-node training for the cluster phase. |
| **Weights & Biases** | Experiment tracking — already wired into our finetune scripts. |

---

## Things that reliably go wrong

- **Evaluating on read speech only**, then shipping to users who speak
  spontaneously.
- **Speaker leakage across splits**, which quietly inflates every number.
- **Selecting on WER** in a language whose orthography isn't standardized.
- **Training on a benchmark** because it was conveniently large.
- **Trusting automatic MOS** as a stand-in for listeners.
- **Letting a licence question become load-bearing** — check before a corpus
  or checkpoint is deep in the training mix, not after.
