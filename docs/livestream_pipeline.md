# Livestream Corpus Pipeline

Converts zipped `{id}.json + {id}.mp4` diarized livestream recordings (e.g.
Kumu sessions) into TTS/ASR-ready training data. Designed for hundreds of
source files: every stage is incremental, resumable, and re-runnable.

```
             ┌────────┐   ┌────────┐   ┌────────┐   ┌────────────────┐
  zips ───▶  │ parse  │─▶ │ align  │─▶ │   qc   │─▶ │     export     │
             └────────┘   └────────┘   └────────┘   └────────────────┘
             manifests    +timings     +scores       asr/ (16k)  tts/ (24k)
                          (GPU 1.6GB)  (GPU 4GB)     [--push] parquet → Hub
```

## Quick start

```bash
# drop new zips into $LIVESTREAM_DIR, then:
source venv/bin/activate
python process_livestream.py                       # all stages, new files only
python process_livestream.py --stages export --push   # publish to the Hub
python stats_livestream.py                         # quality report
```

Only new/changed files are processed — each source's `.status.json` records
which stages completed and with what configuration.

## Requirements

- `ffmpeg` on PATH (`sudo apt install ffmpeg` in WSL)
- CPU stages: the base `requirements.txt`
- GPU stages (align, qc): a **version-matched** `torch`+`torchaudio` pair from
  the same CUDA index (e.g. both `2.11.0+cu126`; mismatched pairs fail to load
  torchaudio's C extension), plus `faster-whisper`, `silero-vad`,
  `onnxruntime`, `jiwer`, `pyloudnorm`
- VRAM: align ≤ ~1.6GB, qc ≤ ~4GB (large-v3 int8). Stages never co-load
  models. On a shared 8GB card, drop to `--whisper-model medium` (~1.7GB).
- WSL note: keep model caches off the C:-drive VHD (`HF_HOME=/mnt/d/hf_cache`)
  but keep the xet cache VHD-local (`HF_XET_CACHE=$HOME/.cache/xet` — xet's
  MerkleDB fails with I/O errors on 9p-mounted drives). Both are set in
  `.env`. For big pip installs: `--no-cache-dir` with `TMPDIR=/mnt/d/tmp/pip`.

## Source format

Each zip contains `{id}.json` + `{id}.mp4`:

```jsonc
{
  "metadata": {
    "linguistic_profile": {"primary_language": "Taglish", ...},
    "speaker_profile": {"speakers": [{"speaker_id": "Speaker 1", "gender": "...", "role": "..."}]}
  },
  "transcription": [
    {"time_range": "02:25 - 02:33",
     "dialogue": [{"s": "Speaker 1", "txt": "..."}, {"s": "Speaker 2", "txt": "..."}]}
  ]
}
```

Only **block-level** time ranges exist; a block may bundle several speaker
turns with no per-turn timestamps. Loose (unzipped) `{id}.json + {id}.mp4`
pairs dropped in `$LIVESTREAM_DIR` also work.

## Stages

### parse (CPU)
- Extracts zips into `raw/_extracted/{id}/`, decodes a 16kHz mono WAV cache
  (`{id}_16k.wav`) via ffmpeg.
- **Dedup**: MD5 fingerprint of the first 60s of decoded PCM; re-uploaded
  streams are skipped and stamped `duplicate_of` in their status file.
- Baseline per-turn timings: single-turn blocks map exactly onto their range
  (`alignment: "exact"`); multi-turn blocks are split proportionally by
  character count (`"interpolated"`).
- **Speaker namespacing**: `"Speaker 1"` → `{file_id}#S1` (raw labels collide
  across files).
- **Split**: file-level, `md5(file_id) % 10 == 0 → test`. Speaker-disjoint by
  construction and stable as files are appended (no reshuffling).

### align (GPU ~1.6GB)
- Per block: forced-aligns the concatenated turn texts against a padded audio
  window (block ± 2s, clipped at neighbor midpoints — human timestamps can be
  off by ±1–2s) with `torchaudio.pipelines.MMS_FA` (Meta's multilingual
  wav2vec2 aligner; lowercase-latin charset covers Taglish code-switching
  natively). `<star>` tokens at turn edges absorb untranscribed audio
  (cross-talk, backchannels, music) so incomplete transcripts don't poison
  the alignment. Hyphens are stripped from targets ('-' is the CTC blank).
- Turn boundaries = first/last word spans → `alignment: "forced"`,
  `align_score` = mean frame probability (0–1).
- **Retry pass**: turns scoring below `min_score` (typically quiet/remote
  speakers drowned out in long blocks) are re-aligned alone in a narrow
  window between their neighbors' aligned bounds — rescues most of them.
- Silero VAD (CPU) snaps boundaries to speech edges (max trim 1s per side,
  re-pad ±0.15s) and computes `speech_ratio` per segment.
- **Fallback**: still-low-scoring turns keep the aligner's timings when
  available (better than character-proportional) but stay labeled
  `"interpolated"` with their `align_score` recorded; total failures keep the
  parse-time baseline. Segments are never lost at this stage; gating happens
  at export. The parse baseline is snapshotted (`start0`/`end0`), so re-runs
  are idempotent.

### qc (GPU ~4GB)
- faster-whisper round-trip per segment → `asr_text`, `asr_cer` (CER over
  normalized text; CER not WER because Taglish orthography variants make WER
  unstable).
- Audio metrics: `lufs` (pyloudnorm), `clip_ratio`, `snr_db` (speech vs
  nonspeech RMS within the block window — doubles as a background-music
  indicator).
- `overlap` heuristic: tight adjacent turns + low alignment confidence, or
  high CER despite confident alignment.

### export (CPU)
- Applies the gates below, writes audiofolder datasets to
  `$LIVESTREAM_OUTPUT_DIR/asr` (16kHz) and `/tts` (24kHz, decoded lazily from
  the 44.1kHz source), and with `--push` streams parquet shards to the Hub
  (`data/asr/*`, `data/tts/*`) with upload resume via `shards/progress.jsonl`.

## Export gates

| Metric | ASR set | TTS set |
|---|---|---|
| alignment | any (column kept) | `forced` only |
| align_score | ≥ 0.15 (forced rows) | ≥ 0.50 |
| asr_cer | ≤ 0.50 | ≤ 0.15 |
| duration | 0.3–30s | 1.0–15s |
| overlap | allowed (flagged) | excluded |
| speech_ratio | ≥ 0.50 | ≥ 0.85 |
| clip_ratio | ≤ 1% | ≤ 0.1% |
| lufs | — | −30 … −10 |
| min words | 1 | 3 |
| unscored rows | pass (metrics null) | excluded |

Thresholds live in `halolib/qc.py` (`ASR_GATE` / `TTS_GATE`).

## Resume & invalidation

- Each stage stamps `manifests/{id}.status.json` with a **config hash** of the
  parameters it depends on (model names, thresholds, pad values). Editing any
  of these re-runs only the affected stage on the next invocation.
- `--force <stage>` clears that stage's stamp for the selected files.
- `--file-id <ID>` / `--limit N` scope a run.
- Parquet upload resume is independent: `shards/progress.jsonl` keys on
  `prefix/split/shard_idx`, so an interrupted `--push` continues where it
  stopped.
- Manifests are the single source of truth; audio caches (`*_16k.wav`,
  `*_24k.wav`) are disposable and re-decoded on demand.

## Manifest row (one per turn)

```jsonc
{
  "file_id": "...", "idx": 7, "block_idx": 3, "turn_idx": 1,
  "block_start": 145.0, "block_end": 153.0,
  "speaker_raw": "Speaker 2", "speaker_id": "{file_id}#S2",
  "gender": "female", "role": "Co-host", "language": "tgl-eng",
  "sentence": "Yes mi. Saglit lang...",
  "start": 146.56, "end": 153.0, "duration": 6.44,
  "alignment": "forced|interpolated|exact", "align_score": 0.71,
  "word_timings": [[146.6, 146.9], ...],          // after align
  "speech_ratio": 0.93, "asr_text": "...", "asr_cer": 0.12,
  "overlap": false, "lufs": -17.2, "clip_ratio": 0.0, "snr_db": 18.3,
  "split": "train", "audio_src": "...", "wav16": "..."
}
```

## Known gaps & caveats

**Mitigated** — timestamp coarseness (forced alignment + VAD snapping); gross
transcript errors (CER gate); speaker-ID collisions (namespacing); train/test
speaker leakage (file-level split); music/notification audio (speech_ratio +
SNR gates); clipping (metrics + TTS gate); re-uploaded streams (PCM
fingerprint dedup).

**Partially mitigated** — overlapping speech (heuristic flag only; some
overlap passes the ASR gate); speaker-attribution errors in the source JSON
(CER catches wrong-words but not right-words-wrong-speaker — a residual risk
for TTS speaker conditioning); CER penalizes valid Taglish orthography
variants (ASR threshold deliberately loose).

**Accepted / open** — no per-segment language ID (corpus tagged `tgl-eng`
file-level; back-fillable with a lang-ID model without re-cutting audio); no
speaker age metadata; TTS numeral/abbreviation normalization ("50k") is
unresolved; PII: real names/handles are spoken and transcribed in public
broadcasts — the dataset card states this; lossy 44.1kHz AAC source caps TTS
audio quality (fine for 24kHz training, not studio-grade); transcripts are
human-quality but unverified against ground truth.

## Verification protocol

After processing a new batch:
1. `python stats_livestream.py` — expect ≥90% forced alignment, forced-CER
   median well below interpolated-CER median.
2. Spot-listen: 10 random TTS passers, 10 worst forced by `align_score`,
   5 overlap-flagged, 5 interpolated. Pass bar: 9/10 TTS passers have
   text matching audio end-to-end with no adjacent-speaker bleed.
3. After `--push`: `load_dataset("<repo>", "asr")` row count matches the
   export log; re-running the driver is a no-op.
