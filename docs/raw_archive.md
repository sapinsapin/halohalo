# Raw Source Archive (`halo-livestream-raw`)

`push_livestream_raw.py` publishes the **inputs** to the livestream pipeline:
the operator-supplied transcript JSON plus the recording's audio track.

[`sapinsapin/halo-livestream`](https://huggingface.co/datasets/sapinsapin/halo-livestream)
is derived data — ~30 s segments with alignment confidence, round-trip CER, SNR
and overlap flags. [`sapinsapin/halo-livestream-raw`](https://huggingface.co/datasets/sapinsapin/halo-livestream-raw)
is what those were derived *from*. Keeping it fixed and addressable is what
makes a pipeline change measurable rather than merely different: re-run any
future version over the same bytes and the delta is attributable to the code.

```bash
python push_livestream_raw.py --dry-run   # stage + report, upload nothing
python push_livestream_raw.py             # upload new recordings only
python push_livestream_raw.py --card-only # refresh the card
python push_livestream_raw.py --file-id <ID>
```

## Repo layout

```
audio/{file_id}.{m4a|mp3|opus|flac}   archival audio track
transcripts/{file_id}.json            operator transcript, as delivered
index.jsonl                           one row per recording
```

`find_pairs` reads this layout directly, so a downloaded snapshot feeds back
into `process_livestream.py` with no reshuffling.

## Why lossy audio is copied, not transcoded

The instinct for an archive is "store it losslessly, use FLAC." For a source
that is *already* lossy, that instinct is wrong in three separate ways.

Measured on the seed recording (26:56 of AAC 48 kHz mono):

| Path | Size | Decoded PCM checksum |
|---|---|---|
| source `.mp4` (AAC) | 9.0 MB | `8bfa107f…` |
| stream-copy → `.m4a` | 9.0 MB | `8bfa107f…` ✅ identical |
| encode → `.flac` | 113 MB | `b2f62cde…` ❌ different |

1. **It recovers nothing.** FLAC is lossless with respect to its *input* — and
   its input here is the AAC decoder's output. Everything the AAC encoder threw
   away is already gone; FLAC just preserves the wreckage at higher cost.
2. **It costs 12×.** 9 MB → 113 MB. Extrapolate to the multi-hour streams this
   dataset is built to receive and the archive becomes the expensive part of
   the project for no benefit.
3. **It is not even faithful.** The decoded checksums differ — ffmpeg's FLAC
   path applied its own sample-format conversion. The "lossless archival copy"
   is a different signal from the delivered one. The stream copy round-trips
   bit-exactly.

So `halolib/raw.normalize_audio` copies the stream whenever the codec is in
`COPY_CONTAINERS` (AAC, MP3, Opus, Vorbis, ALAC, FLAC) and only encodes when
the source is genuinely uncompressed PCM. There FLAC earns its place: the
compression is lossless, roughly halves the size, and dodges the 4 GB RIFF/WAV
ceiling that a long stream will otherwise hit mid-recording.

Video, when present, is dropped (`-vn`). The pipeline never reads it, and it is
the part of a livestream recording carrying the most personal data.

## Built for long streams arriving over time

- **Staging is idempotent.** An existing non-empty output is reused, so a
  re-run over a directory of processed streams costs one `ffprobe` each.
- **Writes are atomic.** ffmpeg writes `{stem}.part{suffix}` and the result is
  `rename`d into place, so an interrupted run never leaves a truncated file
  that a later run would mistake for finished work. The extension is kept last
  because ffmpeg infers the output format from it.
- **Uploads are incremental.** Anything already on the Hub at the expected byte
  size is not re-sent.
- **Large transfers resume.** Past 1 GiB staged (`LARGE_THRESHOLD_BYTES`) the
  script switches from `upload_folder` to `upload_large_folder`, which chunks,
  parallelizes, and picks up where it left off after an interruption.
- **Integrity is checkable.** `index.jsonl` carries `audio_sha256` and
  `transcript_sha256` per recording.

## Gating

The dataset is **gated**: publicly listed and documented, but the files require
an access request against the terms in the card.

The reason is the asymmetry in exposure. A 30-second segment stripped of
context is a very different disclosure from 27 unbroken minutes of named people
discussing their income. The processed dataset carries most of the research
value; the raw archive carries most of the risk. Gate them accordingly — and
note that "accordingly" can change, which is why nothing here hard-codes what
`halo-livestream`'s own setting happens to be today.

`push_livestream_raw.py` creates the repo **private**, uploads, pushes the card
containing `extra_gated_prompt`/`extra_gated_fields`, applies gating, and only
then flips it public. Creating it public first would leave a window in which
the audio was world-readable before the gate existed.

**A routine upload never changes gating.** `--gated` defaults to `keep`: a new
repo is created `manual`, and an existing one is left exactly as it is. Access
policy is usually adjusted in the dashboard, and a re-upload silently reverting
that would be a quiet security regression — the failure mode is a dataset that
looks gated in the console until the next `push` widens it. Pass `--gated
auto|manual|off` to change it deliberately.

Verify the gate is live:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -L \
  https://huggingface.co/datasets/sapinsapin/halo-livestream-raw/resolve/main/audio/<id>.m4a
# 401 — README.md at the same path returns 200
```

`--gated off` removes the gate entirely — only for a dataset that should be
openly downloadable.

## Provenance caveat

`metadata.file_properties` in the transcript is operator-reported and can
disagree with the container. The seed recording is declared "44.1 kHz, 16-bit,
stereo" and is actually 48 kHz mono AAC. `index.jsonl` is probed from the file
itself — trust it over the transcript, and prefer it when filtering.
