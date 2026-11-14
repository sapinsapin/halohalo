# Chika: repeatable speech curation

Tracking: https://github.com/sapinsapin/halohalo/issues/8
Target: approximately **250 unique retained hours each** of Filipino/Taglish and Hiligaynon, transcribed with our ASR service and reviewed by an LLM. Download every previously identified Hiligaynon archive where public access is available. Continuous radio uses bounded captures. Acquisition is currently prioritized over ASR by user instruction.

## Current implementation and limits

Implemented: RSS discovery, feed snapshots, resumable media acquisition, SQLite checkpoint/event ledger, independent media-duration checks, disk floor, per-show and total acquisition caps, PLD draft migration with content-bound text reviews, local embedded-FLAC Parquet export, private Hub backup with remote hash verification.

Still required: end-to-end ingestion of new audio into segmentation/ASR/LLM stages, speech/music/overlap filtering, PCM and near-duplicate checks, language verification, source-balanced frozen splits, and both 250-hour completed-transcription targets. Do not mistake discovery estimates or downloaded file runtimes for those targets. ASR authentication remains unresolved and is deliberately deprioritized. Do not deploy or restart ASR on GB10.

## Setup

From the halohalo repository root:

```sh
uv venv curation/.venv --python 3.11
uv pip install --python curation/.venv/bin/python -r curation/requirements.txt
```

Install ffmpeg/ffprobe on the host. RSS discovery/downloads require only Python's standard library plus ffprobe. YouTube/archive scripts additionally use yt-dlp when applicable. Credentials are environment variables or external private files, never source files or GitHub issue content.

## Discover and acquire

```sh
python3 -m curation.collect discover --sources curation/sources.json --root curation/outputs/rss
python3 -u -m curation.collect download --root curation/outputs/rss
python3 -m curation.collect status --root curation/outputs/rss

python3 -m curation.collect discover --sources curation/taglish_sources.json --root curation/outputs/taglish/shared
python3 -u -m curation.collect download --root curation/outputs/taglish/shared --max-hours 300 --max-source-hours 50
```

Hiligaynon defaults to the full available feed inventory. Taglish collects 300 candidate file hours, balanced by acquired runtime with a 50-hour/show cap, allowing room for later rejection. Caps may exceed by one episode. Feed availability limits discovery: a feed may expose only recent episodes, so its complete inventory is not necessarily a show's lifetime archive. Source registries include candidate language and evidence; neither filenames nor English show descriptions establish spoken language.

The worker holds an OS lock per root, preventing two writers from downloading the same queue. Completed items survive restarts. Interrupted downloads use HTTP Range only with an ETag/Last-Modified validator; a server returning a full response causes a clean overwrite. Files are renamed only after media probing, then hashed and checkpointed. Unavailable 401/403/404/410 items are logged without bypassing access restrictions. Other failures retry up to three attempts. Re-discovery updates URLs without erasing progress. Errors are queryable in SQLite. Downloaded totals include music, ads and silence until later QC.

The disk floor is 25 GiB, checked during writes. Other workers share that free space; monitor before adding concurrency. Inspect the actual PID and growing file/log before treating a job as running. A worker.json file alone does not prove liveness. RSS workers can be safely restarted with the same command after confirming the prior process has ended. Do not launch duplicates. Radio cannot recover past live intervals; every capture has its UTC interval and termination reason.

## Standard working schema

`curation/schema.py` defines `sapinsapin.speech.v1`; `curation.cli validate` enforces required fields and split checks. JSONL manifests retain:

- PLD-compatible `sentence`, `duration`, `speaker_id`, `speech_type`, `language`, `num_words`, `text_is_prompt`.
- Stable `id`, `source_id`, `recording_id`, original audio path and SHA256, `prompt` separate from transcription, `languages`, `split`, `review_status`, rights and provenance.
- Original model/config/source/split-spec hashes, segments and their timestamp semantics; synthetic provenance must remain explicit.

`sentence` is always spoken-text draft or verified transcript, never an elicitation question. Keep source questions out of both ASR and LLM review inputs. Hiligaynon–English and Filipino–English code-switching, including local affixes on English roots, are valid speech. Do not translate, purify or reject grammatical code-switches. The primary language label does not imply monolingual speech. Annotate additional languages only from evidence.

A text-only LLM review can flag uncertainty and propose exact-span alternatives; it cannot establish that words were spoken. Preserve raw ASR output and model responses. Only an attributable audio review bound to both audio and sentence hashes sets `audio_verified`. Never compute WER/CER using the LLM guess as ground truth. Source facts and opinions are speech content, not claims to fact-correct in transcripts.

## PLD import and export

```sh
python3 -m curation.cli import-pld \
  --drafts pld_spontaneous/outputs/llm_review \
  --reviews pld_spontaneous/outputs/chatgpt_review/complete \
  --output curation/outputs/pld-hil/manifest.jsonl
python3 -m curation.cli validate curation/outputs/pld-hil/manifest.jsonl
curation/.venv/bin/python -m curation.cli export \
  --manifest curation/outputs/pld-hil/manifest.jsonl \
  --audio-root curation/outputs/pld-hil/audio \
  --output curation/outputs/pld-hil/export-v1 --profile curation
```

`curation` exports private drafts at 16 kHz. `asr` exports 16 kHz and `tts` 24 kHz, requiring recorded training permission, audio verification and assigned splits. Upsampling does not improve original bandwidth; original rate is recorded. All profiles embed mono FLAC as `audio={bytes,path:null}` and keep the established core columns. Additive metadata lives in `provenance_json` and a full sidecar JSONL. Output directories are immutable; use a new version directory after correcting inputs. A summary.json is written only once all shards finish; its absence means incomplete export. Source-byte hashes are checked before export. Test reservations remain reserved, and known speakers/recordings/exact source files cannot straddle assigned splits. Unknown speakers are not proof of speaker-disjointness. Near-duplicate checks are still needed.

## Hub storage and restart

Destination: **private `sapinsapin/chika`** until source-specific redistribution rights are resolved. Public availability is not a license for commercial training or redistribution. Keep rights `unknown` when evidence is absent; do not assign a blanket permissive license. Private backup is storage, not a rights determination.

```sh
curation/.venv/bin/python -m curation.hub --root curation/outputs/rss --namespace hiligaynon-rss
curation/.venv/bin/python -m curation.hub --root curation/outputs/taglish/shared --namespace taglish-rss
```

Raw originals use content-addressed paths; each upload is verified against the remote LFS SHA256 and byte length at its returned commit. SQLite is backed up through the SQLite backup API, not by copying an active WAL database. Checkpoint receipts record immutable revisions. This command **does not evict** local audio. Do not remove a local original unless its remote object and acquisition metadata are verified and recoverable. Recovery and explicit verified eviction are available below; automatic storage supervision remains to be completed. At the floor workers pause safely.

GitHub stores code, source registries, aggregate STATUS.json and issue #8, never credentials, private transcripts/questions or audio. At handoff: regenerate status from each live ledger, record worker PID/command and actual liveness, reconcile HF receipts, commit only code/aggregate files, then update the issue. Resume completed stage identities using input bytes plus configuration/model identity, not filenames alone.

## Video archives and radio

```sh
python3 -m curation.acquire_archives --source discover
python3 -u -m curation.acquire_archives --source deditos
python3 -u -m curation.acquire_archives --source all-youtube
python3 -u -m curation.acquire_radio --station bombo-iloilo --seconds 1800
```

Finish the two-worker Deditos pass before adding the serial video archive worker. Discovery rebuilds each scoped public inventory; failure of one source does not erase others. A missing, empty or failed inventory leaves the all-source queue incomplete. The tracked `archive_sources.json` lists coverage and limits. Complete episodes are prioritized before clips, but all accessible scoped items remain queued. Completed recordings are checked by content hash before skipping. A failed YouTube attempt logs the error and does not imply permission to bypass login/geoblocking/access controls. The source-language candidates require later validation.

Radio stations supported by `acquire_radio.py`: `bombo-iloilo`, `bombo-bacolod`, `bombo-roxas`, `rmn-iloilo`, `rmn-bacolod`. Capture intervals are finite and independently recoverable Matroska files. Station endpoints can fail; preserve the failure and retry only when appropriate. Identical network programmes may appear on multiple stations and need acoustic deduplication. DYSI nutrition-school documentation is a discovery lead, not a recovered audio archive.

## Recovery drills

Restore an immutable RSS checkpoint into a **new empty directory**:

```sh
curation/.venv/bin/python -m curation.restore \
  --receipt curation/outputs/rss/hub-checkpoint.json \
  --root curation/outputs/recovered-rss --max-files 1
```

`--max-files 0` restores metadata only. Backed-up but unselected recordings remain `remote`, with no misleading local path. Files without a backup become pending for reacquisition, with original metadata retained in `restore_inventory`. Selected files require remote hash/size and local checksum verification before the recovered directory is atomically published. The command checks SQLite integrity and the free-space floor, defaults to a 2 GiB selected-media budget, and never alters the original root. Restored remote hours still count against acquisition caps; they do not count as locally available audio.

Archive discovery and per-file ledgers can be packed separately:

```sh
curation/.venv/bin/python -m curation.archive_checkpoint pack \
  --root curation/outputs/archive_sources --root curation/outputs/radio \
  --output curation/outputs/checkpoints/archives-v1.tar.gz
```

Use the command's `--help` for verify/restore syntax. The deterministic metadata bundle excludes audio, logs, temporary files and symlinks. It includes a member-level checksum manifest; restoration rejects unsafe paths and non-file entries. Each source file is stable when copied, but independently running workers are not transactionally frozen. All saved PID/status values are historical observations. The bundle belongs in private Hub storage.

## Preparing new recordings

```sh
python3 -m curation.prepare --rss-root curation/outputs/rss \
  --output curation/outputs/prepared-hil --max-recordings 2
python3 -m curation.cli validate curation/outputs/prepared-hil/manifest.jsonl
```

Preparation verifies original bytes, decodes to canonical 16 kHz mono PCM on disk, and creates bounded nonoverlapping FLAC windows with exact sample offsets. It records original hashes, full-recording PCM hashes and configuration/tool identity. Completed windows are reused only when derivative hashes still match; changing preparation configuration requires a new output directory. The ledger publishes a recording only after all its windows succeed.

These are **ASR transport windows**, not asserted speech regions, speaker turns or TTS training utterances. They have `sentence=""`, `asr_status="pending"`, unknown speakers and provisional source language. No ASR-completed hours are claimed. Empty pending text is not a verified silence transcript. VAD/music/overlap QC, diarization, contextual ASR, alignment and acoustic near-duplicate removal still need to run. Full-recording PCM equality catches exact decoded repeats only, not re-encoded clips or network simulcasts. Original media remains available for higher-rate derivatives.

## LLM text review

```sh
python3 -m curation.review --manifest curation/outputs/pld-hil/manifest.jsonl \
  --output curation/outputs/text-review --key-file ~/.nyo/pld-review.key --limit 3
```

The adapter defaults to the working NYO route `glm-5.3-flash-nebius`; base URL and model are configurable. The earlier `glm-5.3-flash` route had an upstream credit failure. Only ID, transcript and provisional source-language hint are sent; questions, speaker identity, local paths and audio are excluded. Empty/pending ASR windows are skipped. Raw responses and validated text reviews stay separate from the source manifest.

Review caches bind text, audio hash, segment offsets, model/base URL and prompt identity. An OS lock prevents duplicate concurrent local requests. A saved raw response can recover after an interrupted commit; malformed responses fail closed. Transport uncertainty can still cause duplicate provider billing on retry without provider-side idempotency support. Errors are logged without keys or response text; private raw responses remain available for inspection.

Prompt v1's live check overflagged a recognizable phonetic English code-switch. Version v2 adds explicit phonetic/code-switch examples and forbids lexical word replacements in suggestions. Suggestions may only alter spacing, punctuation or capitalization; original ASR text is never overwritten. Successful JSON validation establishes an API/schema contract, **not** linguistic accuracy. Small synthetic text-only instruction probes are kept outside the dataset, and cannot establish general Hiligaynon/Taglish quality. Every result remains `text_review_only`, `audio_verified=false`, `human_approved=false`. Audio-based adjudication is still required.


## Reclaiming verified RSS storage

When the acquisition worker has stopped (for example at its disk floor), first upload and verify completed files plus the SQLite checkpoint using `curation.hub`. Then inspect a bounded plan:

```sh
curation/.venv/bin/python -m curation.evict --root curation/outputs/rss --max-files 5
curation/.venv/bin/python -m curation.evict --root curation/outputs/rss --max-files 5 --target-free-gib 40 --apply
```

The default is dry-run. An active downloader's OS lock prevents any eviction; this tool never stops a worker. Every selected item must be present in a checksum-verified immutable checkpoint, whose backup coordinates agree with both the local ledger and current private Hub object. Local SHA256/size/inode checks precede deletion. A durable `remote` transition and journal precede unlink so interruption cannot produce a false claim that a deleted file is local. Interrupted journal entries can be reconciled safely. Recovery archives imported deletion intents as provenance instead of acting on old filesystem paths.

Only completed `.media` originals within the selected RSS root are eligible. Other source directories and derived FLAC windows are untouched. No original audio was evicted during the initial recovery/dry-run checks. Automatic threshold monitoring and archive-source reclamation are not yet implemented; until then, inspect status and run the documented steps when storage requires it. Resume the collector only after confirming the previous process ended; remote hours continue to count against its acquisition caps.
