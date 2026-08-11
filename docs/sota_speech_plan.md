# SOTA Philippine-Language Speech Plan — TTS + English↔Filipino S2S

Target: the best open text-to-speech and speech-to-speech translation models
for Philippine languages, trained on a cloud cluster, released under the
`sapinsapin` org with the same card/dashboard discipline as the corpora.

This plan assumes cluster-scale compute. Local hardware (the 3070) is used
only for smoke tests of training code before cluster submission — the same
pattern already proven with `finetune_tts.py`/`finetune_asr.py`.

---

## 1. What "SOTA" means here, measurably

| Model | Beat | On |
|---|---|---|
| TTS | MMS-TTS-tgl, our `speecht5_tts-fsc`, XTTS-v2 zero-shot | CMOS/MOS with native listeners; UTMOS proxy; WER of re-transcribed synthesis (whisper-large-v3); speaker-similarity for cloning |
| S2S | SeamlessM4T-v2-large (fil↔en), cascaded whisper→NLLB→our-TTS | ASR-BLEU and BLASER-2.0 on FLEURS-fil ↔ FLEURS-en; latency; speaker/prosody retention (expressive track) |
| Both | — | A new **Taglish code-switching test set** (nothing public covers it; we build it from livestream + FSC spontaneous held-out data) |

Two models, one data engine:

- **halo-tts** — multilingual Philippine TTS (Filipino first-class, the other
  nine PLD languages as a multilingual head start), zero-shot voice cloning.
- **halo-s2s** — speech-to-speech translation English↔Filipino, expressive
  (preserves speaker identity and prosody), with text output as a free
  byproduct (ASR + S2TT come out of the same architecture).

## 2. Data strategy — the actual moat

Model architectures are public; the differentiator is Philippine-language
data volume and quality. Current holdings (all already processed and carded):

| Corpus | Hours | Role |
|---|---|---|
| `filipinospeechcorpus` (FSC) | ~100 | Studio read + spontaneous Filipino; TTS gold set, eval anchor |
| `pld` (private) | 448 | 10 languages, 980 speakers; **bilingual speakers reading English and Philippine-language prompts** — natural seed for cross-lingual voice transfer. `text_is_prompt` rows excluded from supervised use |
| `halo-livestream` | growing | Spontaneous Taglish; the QC pipeline (align → round-trip CER → export) is built and incremental |
| BantayWika + halohalo web corpus | text | Text normalization training, tokenizer coverage, MT augmentation |

### 2.1 Scale to 5,000–10,000 hours (the SOTA threshold)

Modern codec-LM TTS needs thousands of hours; S2ST needs the same via
pseudo-labeling. Sources, in priority order:

1. **Scale the livestream pipeline.** It was explicitly built for hundreds of
   files with incremental resume. Point it at licensed/permitted Filipino
   YouTube, podcast, and radio archives. The QC gates (round-trip CER, VAD,
   overlap detection) are exactly the pseudo-label filter needed — this is the
   single highest-leverage asset we own.
2. **Public corpora sweep**: Common Voice (tl), FLEURS-fil (eval only — never
   train), MMS-aligned religious recordings, OpenSLR Philippine holdings,
   SEACrowd catalog entries (we're in that org already).
3. **Pseudo-labeling factory**: whisper-large-v3 (or our finetuned successor)
   transcribes crawled audio → forced alignment (MMS CTC, already in halolib)
   → QC gate → training shards. Human spot-check 1% per batch.
4. **Community recording drive** — the "it takes a village" angle: a
   Common-Voice-style prompt-reading Space under the org, feeding the same QC
   pipeline. Slow burn, but produces the cleanest eval and finetune data and
   builds the community the initiative is founded on.

### 2.2 Parallel data for S2S (none exists — we manufacture it)

CVSS-style: take every (audio, transcript) pair we have, machine-translate the
transcript (NLLB-200-3.3B, both directions), then synthesize target-language
speech with our own TTS once it exists (bootstrap round 1 with MMS-TTS/XTTS).
Result: pseudo-parallel S2ST corpus at the scale of our ASR data. English
source speech comes free from LibriSpeech/LibriTTS/Common Voice-en with the
same treatment. PLD's bilingual speakers give a small **genuinely parallel**
same-speaker seed set — rare and valuable for the expressive objective;
reserve a slice for eval.

### 2.3 Text normalization (the known gap)

Our current TTS rejects digit-bearing text — documented gap. Build a Filipino
TN module (numbers, dates, currency, mixed-code Taglish) trained from
BantayWika/halohalo text + rule seed, and run it in both the data factory and
inference. Without this, "SOTA" fails on the first street address a user types.

## 3. Model architecture

### 3.1 TTS — codec-LM, continued pretraining (primary track)

- **Recipe**: neural-codec language model (CosyVoice/Orpheus lineage): speech
  → discrete codec tokens (Mimi/DAC @ 24kHz) → decoder-only transformer
  conditioned on text tokens + speaker prompt → codec decoder.
- **Why**: current open SOTA for expressiveness + zero-shot cloning, scales
  with data, and supports **continued pretraining from a multilingual
  checkpoint** — we do not train from scratch. Start from the strongest
  open multilingual codec-LM at kickoff time (candidates re-evaluated then:
  CosyVoice-3, Orpheus, Fish-Speech), swap its tokenizer coverage for our
  text distribution, and continue-pretrain on the full Philippine corpus.
- **Size**: 0.5B–3B. Pilot at 0.5B to validate the data mix; final run at
  1–3B depending on pilot scaling curves.
- **Secondary track (cheap insurance)**: flow-matching non-AR model
  (F5-TTS-style) finetuned on FSC+PLD read speech only — lower ceiling but
  very robust, and a strong fallback if the codec-LM's Taglish prosody
  disappoints.

### 3.2 S2S — two tracks, one clearly pragmatic

- **Track A (deliverable)**: **SeamlessM4T-v2-large finetune** (2.3B) on the
  pseudo-parallel corpus + PLD bilingual seed. It already handles fil speech
  input; we add/strengthen the fil target side (unit decoder + vocoder on our
  speech). Expressive variant: SeamlessExpressive-style prosody encoder
  trained on the same-speaker PLD pairs. This is the fastest route to
  beating the public baseline, because the public baseline *is* the base
  model before our data.
- **Track B (research, time-boxed)**: unified speech-LLM — interleave our
  TTS codec tokens and text in a 3–8B LLM (AudioPaLM/Qwen-Omni pattern) so
  ASR, S2TT, S2ST, and TTS are one model. Only promoted if Track A plateaus;
  otherwise it ships as a tech report + checkpoint.

## 4. Evaluation before training

Freeze all test sets in week 1, before any large run:

- FLEURS-fil/en (never trained on), FSC held-out, PLD held-out per language,
  livestream held-out (spontaneous Taglish), the new Taglish benchmark,
  and a native-listener MOS panel recruited via the community (paid).
- Automated gates in CI: UTMOS, whisper-WER of synthesis, ASR-BLEU, BLASER,
  speaker-sim (ECAPA cosine). Every checkpoint gets the full sweep; the
  dashboard Space grows an "evals" tab reading a results dataset.

## 5. Compute plan

Assumptions: H100-80GB class, on-demand cloud (~$2–3/GPU-hr; reserved or
spot cheaper). All numbers are planning ranges, refined after the pilot.

| Phase | Hardware | Duration | GPU-hrs | Est. cost |
|---|---|---|---|---|
| Data factory (ASR pseudo-label + align + QC, 10k h audio) | 16×L40S or 8×H100 + big CPU pool | 2–3 wk | ~3–5k | $8–15k |
| TTS pilot (0.5B, 1–2k h) | 8×H100 | ~1 wk | ~1.3k | $3–5k |
| TTS main (1–3B, 5–10k h, continued pretrain) | 32–64×H100 | 2–4 wk | 15–60k | $40–150k |
| S2S Track A finetune (2.3B) | 8–16×H100 | 1–2 wk | 2–5k | $6–15k |
| S2S Track B (if promoted, 3–8B) | 32×H100 | 2–3 wk | 15–35k | $40–90k |
| Eval + ablations + reruns (reserve ~20%) | — | ongoing | — | ~20% on top |

**Realistic total: ~$70–200k** depending on final model size and whether
Track B is promoted. The pilot ($5k-ish) is the go/no-go gate for the big
line item — scaling curves from the pilot decide the main run's size.

Stack: PyTorch + FSDP/DeepSpeed (or torchtitan — `bantaywika/prepare_torchtitan.py`
already points this direction), WebDataset/parquet shards streamed from the
Hub or object storage, W&B for tracking, Slurm or SkyPilot for orchestration.
Checkpoint every 30 min; assume preemption.

## 6. Roadmap

| Weeks | Milestone |
|---|---|
| 1–2 | Freeze eval sets; run all baselines (MMS-TTS, XTTS, SeamlessM4T-v2, cascaded, our current finetunes) and publish the baseline table |
| 1–4 | Data factory build-out: scale livestream pipeline, public-corpus sweep, TN module v1 |
| 3–6 | Pseudo-label crawl to first 2k h; TTS pilot run; scaling-curve readout → **go/no-go on main run size** |
| 5–10 | Crawl to 5–10k h; pseudo-parallel S2S corpus v1 |
| 8–14 | TTS main run + instruction/cloning finetune; MOS panel round 1 |
| 12–18 | S2S Track A finetune + expressive head; ASR-BLEU/BLASER sweep |
| 16–20 | Release: `halo-tts-{size}`, `halo-s2s-{size}` + demo Spaces + tech report; Track B decision |

## 7. Risks and mitigations

- **PLD licensing** — the corpus is private pending redistribution terms.
  Models trained on it inherit the question. *Resolve terms in writing before
  the main runs*; keep an FSC+public-only ablation so a releasable model
  exists regardless.
- **Crawled-audio rights** — pseudo-label only sources with explicit license
  or permission; keep per-shard provenance (the manifest discipline already
  in halolib) so any source can be excised and the model retrained/filtered.
- **Pseudo-label drift** — QC round-trip CER gates catch most; human 1%
  audits per batch; hold pseudo-labeled data out of eval entirely.
- **Taglish prosody/orthography** — CER-based selection (already our
  convention), the Taglish benchmark, and MOS panels with Taglish-fluent
  raters rather than formal-Filipino-only raters.
- **Nine smaller PLD languages** — 448 h split ten ways is thin per language;
  they ride the multilingual model (transfer from Filipino) with per-language
  eval published honestly, not overclaimed.
- **Cloud spend runaway** — pilot-gated scaling, spot instances for the data
  factory, reserve budget line, weekly burn review against the roadmap table.

## 8. First three actions

1. Freeze and publish the eval suite (repo: `halolib/evals/`, results dataset
   + dashboard tab).
2. Run the baseline sweep on current hardware where possible, cloud burst for
   the big baselines — this is a week of work and makes every later claim
   credible.
3. Start the PLD license conversation and the crawl-permission list — the
   legal clock is the longest pole and it starts today.
