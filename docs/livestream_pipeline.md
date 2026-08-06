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

Each stage reads and writes the same per-file manifest, adding columns.
Nothing is deleted between stages: `parse` writes approximate timings,
`align` overwrites them with better ones, `qc` annotates quality, and only
`export` discards anything — by filtering, never by mutating.

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

## Source format and the core problem

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

**The central difficulty:** timestamps exist only at *block* level, and a
block can bundle several speaker turns. In the sample file, one block spans
92 seconds and contains 2 turns — the JSON says only "these two people spoke
somewhere in this 92-second window." Training data needs per-utterance
boundaries accurate to a few tens of milliseconds. Recovering those
boundaries is what the `align` stage exists to do; everything else is
plumbing and quality control around it.

Loose (unzipped) `{id}.json + {id}.mp4` pairs dropped in `$LIVESTREAM_DIR`
also work.

---

# Stage algorithms

## parse (CPU)

### 1. Source discovery — `sources/livestream.find_pairs`

Globs `*.zip` under `$LIVESTREAM_DIR`, extracting each into
`raw/_extracted/{stem}/` (skipped if the directory already exists, so
re-runs don't re-inflate). macOS resource forks (`__MACOSX/`, `._*`) are
filtered out during extraction.

Inside each extracted directory it locates the audio track by trying
extensions in priority order `[.mp4, .mov, .m4a, .mp3, .wav]` and taking the
first match. Files whose stem ends in `_16k`/`_24k` are excluded — these are
the pipeline's own decode caches, and without this guard a second run would
"discover" its own output as a new source.

### 2. Audio decode — `audio.decode_audio`

`ffmpeg -i {src} -vn -ar 16000 -ac 1 {dest}` produces a mono 16 kHz WAV
cache next to the source. ffmpeg rather than soundfile because soundfile
(libsndfile) cannot open MP4/AAC containers at all. 16 kHz mono because it
is what every downstream model wants: MMS_FA, Silero VAD, and Whisper all
operate at exactly this rate, so decoding once here avoids resampling three
more times later.

The function no-ops if the destination exists, which makes the whole stage
cheap to re-run and lets a crashed run pick up where it left off.

### 3. Transcript parsing — `sources/livestream.parse_doc`

Walks `transcription[]` into `Block`/`Turn` dataclasses. Per turn:

- `clean_sentence` strips bracket annotations (`[laughter]`, `[music]`,
  `[inaudible]`) via `\[[^\]]*\]` and collapses whitespace. These are
  transcriber notes describing audio events, not spoken words — leaving them
  in would train a TTS model to say "laughter" out loud and would inflate CER
  against an ASR system that (correctly) never emits them.
- Turns shorter than 2 characters, or matching `^[.\-–—!?,;:'"\s]*$`
  (punctuation only), are dropped. These arise when a turn was *only* a
  bracket tag and cleaning emptied it.
- Blocks with `end <= start`, or with no surviving turns, are skipped
  entirely.

Language is mapped through `LANG_MAP` to ISO-ish codes (`"Taglish"` →
`tgl-eng`). Speaker gender/role are lifted from `metadata.speaker_profile`
into a lookup keyed by raw label.

### 4. Baseline timing — `sources/livestream.interpolate_block`

Before any GPU work, every turn needs *some* timing. For a single-turn block
the answer is exact: the turn owns the whole block range (`alignment:
"exact"`). For multi-turn blocks the block range is divided proportionally
to each turn's character count:

```
wᵢ = len(turnᵢ.text)
durationᵢ = block_duration × wᵢ / Σw
startᵢ = block_start + Σ_{j<i} durationⱼ
```

This assumes every speaker talks at the same characters-per-second rate,
which is wrong in detail but is a reasonable prior and — critically — is
*monotonic and gap-free*, so it always yields a usable, non-overlapping
partition of the block. It is labeled `"interpolated"` and treated as a
fallback, never as truth.

Worked example, block 3 of the sample (205–212 s, 4 turns):

| turn | chars | baseline | forced (after align) | drift |
|---|---|---|---|---|
| t0 "Ah. Dali mo si ano doon." | 24 | 205.00–206.93 | 204.97–206.73 | −0.2 s |
| t1 "Sino po mi?" | 11 | 206.93–207.82 | 207.42–208.26 | **+0.5 s** |
| t2 "Si Jatsu kasi ano. Ayan mi." | 27 | 207.82–209.99 | 207.98–210.62 | +0.6 s |
| t3 "Si Armans, kay Armans mi." | 25 | 209.99–212.00 | 210.60–212.31 | +0.6 s |

The baseline is in the right neighborhood but is off by half a second on a
0.9-second utterance — t1's clip would have started mid-word and ended
early. That is the error the align stage removes.

### 5. Content fingerprint — `audio.fingerprint`

`md5(first 60 s of decoded PCM as float32 bytes)`. Deliberately computed on
*decoded audio* rather than on the file: the same stream re-exported with
different container metadata, a different filename, or a different upload
timestamp produces a byte-identical PCM prefix and therefore collides, while
a file-level hash would not. A registry at `manifests/fingerprints.json` maps
fingerprint → first file_id that claimed it; later collisions are skipped and
stamped `duplicate_of` in their status file.

60 seconds is a compromise: long enough that two genuinely different streams
essentially never collide, short enough that fingerprinting a 200-file batch
costs seconds rather than minutes.

### 6. Speaker namespacing and split assignment

`namespace_speaker` rewrites `"Speaker 1"` → `{file_id}#S1` via
`speaker\s*(\d+)`. Raw labels are file-local — "Speaker 1" in two different
streams is two different humans — so without namespacing a corpus-wide
`speaker_id` column would silently merge them, corrupting both speaker
statistics and any speaker-conditioned TTS training.

Split assignment is `int(md5(file_id), 16) % 10 == 0 → test`, applied at
**file** level. Two properties matter here:

- **Speaker-disjoint.** Segment-level random splitting would put the same
  speaker in both train and test, letting a model score well by recognizing
  voices rather than generalizing.
- **Stable under append.** The hash depends only on the file's own id, so
  adding the 101st file never reshuffles the first 100. A ratio-based split
  would.

---

## align (GPU ~1.6 GB) — `halolib/align.py`

The core stage. Runs one model over one file at a time, in three passes per
block, then commits.

### Model choice

`torchaudio.pipelines.MMS_FA` — Meta's Massively Multilingual Speech forced
aligner, a wav2vec2 CTC model trained on 1,100+ languages over a
lowercase-Latin character vocabulary. Character-level rather than
phoneme-level output is exactly what makes it work here: Taglish
code-switches mid-sentence ("I think 50k diamonds 'yung minimum na
nare-receive nila"), and a character model has no notion of "which
language's phoneme set is this" to get confused by. Tagalog is in its
training set; English shares the alphabet.

### Pass 0: window construction — `pad_window`

Human block timestamps in this corpus drift by ±1–2 s. Aligning against
exactly `[block_start, block_end]` would clip speech that starts slightly
early. So each block gets a padded window:

```
lo = block_start − 2.0,  clipped to ≥ (prev_block_end + block_start) / 2
hi = block_end   + 2.0,  clipped to ≤ (block_end + next_block_start) / 2
```

The midpoint clipping is the important part. Padding alone would let a
block's window reach into a neighboring block's speech, and the aligner —
which must place every supplied word *somewhere* — would happily align this
block's last turn onto the next block's audio. Cutting at the midpoint of
the inter-block gap gives the padding room to work while guaranteeing two
adjacent windows can never claim the same audio.

### Pass 1: block-level forced alignment — `ForcedAligner.align_window`

**Token construction.** All turns in the block are flattened into one token
sequence with `<star>` tokens at every turn boundary:

```
<star> turn0_word0 turn0_word1 … <star> turn1_word0 … <star>
```

Each word is normalized first: NFKD Unicode decomposition → ASCII-fold to
strip diacritics → keep only characters in the model's vocabulary,
**excluding vocabulary id 0**. That exclusion is load-bearing: in MMS_FA the
blank token is `-`, and Taglish orthography is full of hyphens
("ire-resume", "naka-join", "nare-receive"). Passing a hyphen through as a
literal target raises `targets Tensor shouldn't contain blank index` and
fails the entire block. Words that normalize to nothing (pure digits, pure
punctuation) are held out of the target sequence and back-filled later.

The `<star>` tokens are what make this robust on real livestream audio. A
star matches *any* audio at zero cost, so it absorbs whatever is in the
window that the transcript does not account for: the other speaker's
backchannels, gift-notification jingles, background music, the 2 s of
padding itself. Without stars, CTC must explain every frame using the
supplied words, and unlabeled audio drags word boundaries outward to cover
it.

**Alignment.** `torchaudio.functional.forced_align(emission, targets,
blank=0)` runs Viterbi decoding over the CTC lattice, returning the
most-likely monotonic frame→token assignment and a per-frame probability.
`F.merge_tokens` collapses runs of repeated frames into one span per token.

**Regrouping and time mapping.** Spans come back in target order, so they are
walked in the same order the tokens were built — one span per `<star>`,
`len(chars)` spans per word — to recover which spans belong to which word of
which turn. Frame indices become seconds via

```
ratio  = waveform_samples / emission_frames        # ≈ 320 samples ≈ 20 ms
t_sec  = ratio × span_frame_index / 16000
```

and then get offset by the window's own start time to become absolute file
timestamps.

**Scoring.** `align_score` is the mean per-frame probability across the
turn's word spans (probabilities, not log-probs — `scores.exp()` is applied
before merging). Words that were held out of the target sequence contribute
no score and get timings linearly interpolated between their nearest aligned
neighbors.

### Pass 2: narrow-window retry

Long blocks systematically fail one particular way. In a 92-second block
where the host speaks for 85 seconds and a remote guest answers briefly, the
guest's turn is a small, quiet, possibly phone-quality fragment competing
against a large amount of confident speech; the Viterbi path squeezes it into
a few frames and its score collapses.

So any turn scoring below `min_score` (0.15) gets a second attempt, alone,
in a narrow window bounded by its neighbors' *already-aligned* positions:

```
lo₂ = max(window_lo, min(prev_turn_end, next_turn_start) − 0.75)
hi₂ = min(window_hi, max(prev_turn_end, next_turn_start) + 0.75)
```

The turn is re-aligned as the sole content of that window; the retry result
is accepted only if it scores strictly better than the first attempt. This
is cheap (a fraction of a second on a few seconds of audio) and effective —
measured on the sample file:

| block dur | turn | pass 1 | after retry | outcome |
|---|---|---|---|---|
| 49 s | "Naka-join ko lang kahapon." | 0.095 | **0.565** | rescued |
| 20 s | "Mayayaman." | 0.057 | **0.446** | rescued |
| 44 s | "Ayan 'yan o ayan na mi." | 0.025 | **0.339** | rescued |
| 92 s | "I think 50k diamonds…" | 0.085 | **0.239** | rescued |
| 12 s | "At itong 500 million…" | 0.002 | 0.114 | still below gate |
| 29 s | "Yes." | 0.121 | 0.150 | borderline |

Overall this pass moved the file from 81% to **90%** forced alignment.

### Pass 3: VAD snapping and commit

Aligned boundaries land on the first and last *phoneme* of a turn, which
tends to clip plosive onsets and swallow trailing breath. Silero VAD (ONNX,
CPU) is run once over the whole file (cached to `{id}.status.vad.json`) to
get speech regions, then `trim_to_speech` adjusts each boundary:

1. Find VAD speech regions overlapping the segment.
2. Snap the start forward to the first speech onset — but only if the
   required move is ≤ `max_trim` (1.0 s). A larger disagreement means VAD and
   the aligner have found different things; trust the aligner and leave the
   boundary alone.
3. Symmetrically snap the end back to the last speech offset.
4. Re-pad both sides by 0.15 s so the final cut does not clip the phoneme
   the snap just found.

`speech_ratio` — the fraction of the segment covered by VAD speech — is
recorded for every segment regardless of alignment outcome. It is the
primary detector for "this clip is mostly music or notification audio."

**Commit policy.** Segments are never dropped here; they are labeled:

- score ≥ 0.15 → `alignment: "forced"`, timings replaced, `word_timings`
  stored.
- score < 0.15 → timings still replaced (the aligner's guess beats
  character-proportional even when unconfident) but labeled
  `"interpolated"`, with the score recorded so export gating can exclude it.
- Total failure (exception, empty output) → parse-time baseline retained.

**Idempotency.** On entry the stage snapshots the parse baseline into
`start0`/`end0`/`alignment0` and restores from it. Without this, re-running
`align` would align against its own previous output and drift cumulatively.

---

## qc (GPU ~4 GB) — `halolib/qc.py`

Scores every segment independently. No segment is removed at this stage
either; QC only annotates.

### ASR round-trip

Each segment is re-transcribed with faster-whisper (`large-v3`, int8) and
the result compared to the human transcript. The reasoning: if an
independent ASR system listening to *exactly the clip we cut* produces
roughly the transcript we claim is there, then the text, the boundaries, and
the speaker attribution are all probably right. A high CER means something
is wrong — bad boundaries, wrong transcript, overlapping speakers, or
unintelligible audio — without needing to know which.

Decode settings and why:

- `language="tl"` — pins the language; on Taglish, autodetect flips between
  Tagalog and English between clips and adds spurious variance.
- `beam_size=1` — greedy. This is a scoring pass over thousands of short
  clips, not a transcription product; beam search costs several times more
  for a marginally better hypothesis.
- `condition_on_previous_text=False` — **important**. Whisper's default
  carries prior context forward, which on short disfluent clips triggers
  repetition loops and hallucinated continuations. Disabling it makes each
  clip's score independent.
- `vad_filter=False` — segments were already VAD-snapped in `align`;
  re-VADing would trim them a second time and misalign the comparison.

### CER rather than WER — `normalize_for_cer`

Both texts are normalized (NFKC → lowercase → unify curly apostrophes →
strip punctuation → collapse whitespace) and scored with `jiwer.cer`,
clamped to 1.0.

Character error rate, not word error rate, because Taglish orthography is
genuinely unsettled: `'pag` vs `pag`, `ire-resume` vs `ireresume`,
`naka-join` vs `naka join`. Every one of those is a full word error under
WER — a single clip could score WER 0.4 while being a perfect
transcription — whereas under CER they cost one or two characters. CER
measures the thing being tested (does the audio contain this text) rather
than orthographic convention.

### Audio metrics

- `clip_ratio` = fraction of samples with |x| ≥ 0.999 — digital clipping,
  which sounds like distortion and is unusable for TTS.
- `lufs` — ITU-R BS.1770 integrated loudness via pyloudnorm, computed only
  for segments ≥ 400 ms (the standard's gating window). Catches both
  near-silent and over-compressed clips.
- `snr_db` — computed per block window, not per segment:

  ```
  SNR = 10 · log₁₀( mean(speech²) / mean(nonspeech²) )
  ```

  where the speech/nonspeech partition comes from the VAD mask. Requires at
  least 0.1 s on each side, else `None`. In a livestream the "noise" term is
  literally the background music and gift-notification audio between
  utterances, so this doubles as a background-music detector.

### Overlap heuristic — `flag_overlaps`

No dedicated overlap model (pyannote's are HF-gated and would add VRAM
pressure). Two signals, either sufficient:

1. **Structural** — within a multi-turn block, an adjacent turn begins
   within 0.15 s *and* this turn's alignment confidence is < 0.30. Speakers
   butting up against each other with degraded alignment is what talking
   over each other looks like.
2. **Contradictory** — `asr_cer > 0.5` while `align_score ≥ 0.5`. The
   aligner is confident the words are there, but an ASR system can't hear
   them cleanly; the usual cause is a second voice on top.

Flagged rows are kept and marked (`overlap: true`), excluded from the TTS
set, allowed into the ASR set. 22% of the sample file was flagged, which is
plausible for a 3-way livestream conversation.

---

## export (CPU)

### Gating

`passes_gate(row, gate)` applies the thresholds below. One subtlety: gate
metrics that are `None` (because the stage that computes them hasn't run)
are *skipped* for the ASR gate but *fail* the TTS gate, via the
`require_metrics` flag. The effect is that an unscored corpus still produces
a usable ASR set, while the TTS set is never populated by unverified audio.

### Slicing

Rows are grouped by source WAV so each file is opened once and all its
segments are cut in one pass, then groups are fanned across a
`ProcessPoolExecutor`. Cutting uses `soundfile`'s windowed read
(`sf.read(start=, stop=)`), which seeks to the byte offset rather than
loading the file — pulling a 5-second clip from a 27-minute WAV reads only
5 seconds of samples.

The 24 kHz TTS cache is decoded **lazily**, only for source files that
actually contributed a TTS-passing segment. On a corpus where most files
yield few TTS clips this avoids decoding a second full-length WAV per file.

### Output

- **Audiofolder** — `$LIVESTREAM_OUTPUT_DIR/{asr,tts}/{split}/` with
  `audio/*.wav` + `metadata.jsonl`. For local inspection and spot-listening.
- **Parquet shards** (`--push`) — 2,000 rows per shard, audio embedded as
  WAV bytes, streamed to the Hub as `data/asr/*` and `data/tts/*`. Each
  shard is written, uploaded, then unlinked, so disk never holds more than
  ~2 shards regardless of corpus size, and `shards/progress.jsonl` makes an
  interrupted upload resumable.

---

## Export gates

| Metric | ASR set | TTS set |
|---|---|---|
| alignment | any (column kept) | `forced` only |
| align_score | ≥ 0.15 (forced rows) | ≥ 0.50 |
| asr_cer | ≤ 0.50 | ≤ 0.15 |
| duration | 0.3–30 s | 1.0–15 s |
| overlap | allowed (flagged) | excluded |
| speech_ratio | ≥ 0.50 | ≥ 0.85 |
| clip_ratio | ≤ 1% | ≤ 0.1% |
| lufs | — | −30 … −10 |
| min words | 1 | 3 |
| unscored rows | pass (metrics null) | excluded |

Thresholds live in `halolib/qc.py` (`ASR_GATE` / `TTS_GATE`). They are
starting points calibrated on one file; revisit once a real batch exists.
Because gates are applied at export time and nothing upstream is destroyed,
retuning them costs one `--stages export` run, not a reprocess.

## Resume & invalidation

- Each stage stamps `manifests/{id}.status.json` with a **config hash** of the
  parameters it depends on (model names, thresholds, pad values). Editing any
  of these re-runs only the affected stage on the next invocation — e.g.
  changing `ALIGN_CFG["pad"]` re-aligns but does not re-download or re-decode.
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
  "start0": 145.0, "end0": 152.1, "alignment0": "interpolated",  // parse baseline
  "alignment": "forced|interpolated|exact", "align_score": 0.71,
  "word_timings": [[146.6, 146.9], ...],          // after align
  "speech_ratio": 0.93, "asr_text": "...", "asr_cer": 0.12,
  "overlap": false, "lufs": -17.2, "clip_ratio": 0.0, "snr_db": 18.3,
  "split": "train", "audio_src": "...", "wav16": "..."
}
```

## Parallelism

| Stage | Bottleneck | Strategy |
|---|---|---|
| parse | ffmpeg subprocess | `ThreadPoolExecutor` across files (threads release the GIL during subprocess waits) |
| align | GPU | single process, one file at a time; VAD cached per file |
| qc | GPU | single process; model loaded once per run, freed on exit |
| export | disk + CPU | WAV-grouped slicing fanned across `ProcessPoolExecutor` |
| push | network | write → upload → unlink, ≤2 shards on disk |

GPU stages are deliberately serial and never co-resident — on an 8 GB card
the aligner (~1.6 GB) and Whisper large-v3 int8 (~4 GB) would fit together
only barely, and any desktop VRAM usage would OOM the run. Each scorer
explicitly `del`s its model and calls `torch.cuda.empty_cache()` on exit.

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
   median well below interpolated-CER median. That separation is the
   aligner's self-validation: if forced and interpolated segments scored the
   same, alignment would be adding nothing.
2. Spot-listen: 10 random TTS passers, 10 worst forced by `align_score`,
   5 overlap-flagged, 5 interpolated. Pass bar: 9/10 TTS passers have
   text matching audio end-to-end with no adjacent-speaker bleed.
3. After `--push`: `load_dataset("<repo>", "asr")` row count matches the
   export log; re-running the driver is a no-op.
