# Chika: repeatable speech curation

[Architecture overview](workflow_architecture.md) · [Infographic PNG](diagrams/chika-workflow.png) · [Tracking issue](https://github.com/sapinsapin/halohalo/issues/8)

![Chika's six-stage bulk workflow, separate contributor route, recovery layer and quality gates.](diagrams/chika-workflow.svg)

Target: approximately **250 unique retained, transcribed hours each** for Filipino/Taglish, Hiligaynon, Cebuano, Bikolano and Ilocano: **1,250 hours overall, not achieved volume**. Download every previously identified Hiligaynon archive where public access is available. Continuous radio uses bounded captures. Acquisition takes priority when ASR has problems.

## Current implementation and limits

Implemented: source inventories and resumable acquisition; canonical PLD-compatible manifests; exact-sample preparation and pinned-model VAD; authenticated Filipino/Hiligaynon ASR with bounded live checks; text-only LLM review and content-bound attachment; embedded-FLAC Parquet export; verified private Hub publication and backup; RSS storage supervision; source-interval coverage accounting. The architecture overview maps these responsibilities to modules.

Still required at scale: language, music and overlap checks; acoustic near-duplicate handling; source-balanced frozen splits; broader audio verification; reliable large-scale transcription and review; and all five completed-hour targets. Cebuano, Bikolano and Ilocano ASR routes remain unverified. Filipino/Hiligaynon authentication works, but isolated server finalization timeouts persist. Do not mistake discovery estimates, raw runtime, padded VAD candidates or small integration checks for completed targets. Do not deploy or restart ASR on GB10.

The website/MCP contributor route is a separate workstream in development. It is intended to use contributors' own assistant transcription capabilities, with scoped submissions into the same schema and private staging; it is not an alternative ASR backend already deployed by this bulk workflow.

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

Only completed `.media` originals within the selected RSS root are eligible. Other source directories and derived FLAC windows are untouched. No original audio was evicted during the initial recovery/dry-run checks. Automatic RSS threshold monitoring is available through `curation.supervise`; archive-source reclamation is not yet implemented. Resume the collector only after confirming the previous process ended; remote hours continue to count against its acquisition caps.


## Storage supervisor

Run `curation/.venv/bin/python -m curation.supervise --once` to observe, or `curation/.venv/bin/python -u -m curation.supervise --apply` for the persistent worker. Check `curation/outputs/supervisor/heartbeat.json` before launching another instance. It backs up bounded RSS batches below 40 GiB free and reclaims only verified, idle-root originals toward 45 GiB free. Downloader locks prevent eviction during acquisition. Durable launch guards, cooldowns and retry limits protect restarts. Hiligaynon RSS is uncapped; Taglish retains its 300-hour overall / 50-hour-per-show acquisition caps. These are raw runtime caps, not verified speech totals.

## Authenticated Boses ASR

The deployment source is `aineolab/sapinsapin-voice` at commit `9f341a84a22e3af04f45adc5d8c9fee8cf457af1`. Its current upstream is `wss://d1ve4x2dfkyese.cloudfront.net/v1/asr`. Supply the existing authorized speech credential using `--key-file`; never commit it. The public browser trial relay is separate and is not used for bulk transcription.

```sh
curation/.venv/bin/python -m curation.asr \
  --manifest curation/outputs/prepared-hil/manifest.jsonl \
  --output curation/outputs/asr-client-hil --key-file /path/to/speech-key \
  --language hil --model fw/whisper-large-v3-pld-hil-norm --limit 2 --pace 0
```

For Filipino use language `fil` and model `fw/whisper-large-v3-pld-fil-norm`. The client checks prepared audio hashes, model handshake, complete temporal coverage and matching finalization before publishing draft transcripts. Raw responses and cache identities permit resumable work. Source questions are never sent. Unpaced delivery passed small live pilots; this does not establish sustained service capacity. Automatic server utterance cuts can split a 30-second transport window, so merged text remains a draft requiring segmentation, language checks and listening. Empty responses are recorded separately. ASR success grants neither audio verification nor training rights.

The first larger batch encountered repeated server `finalize_timeout` errors despite successful small pilots. It was stopped; successful draft caches remain reusable and failed windows do not count as transcribed. Resolve service finalization before resuming bulk ASR. The initial new-audio LLM check accepted one text review and rejected one invalid identity/triage response under the strict contract; original transcripts remain unchanged.

Finalized radio captures can be backed up with `curation/.venv/bin/python -m curation.media_backup --root curation/outputs/radio`. This bounded adapter validates local audio against capture receipts, uploads only audio and selected provenance to the existing private dataset, and verifies downloaded bytes at an immutable revision before publishing `hub-media-receipt.json`. It never deletes local audio. A running upload is not a verified backup.

## Use the same schema for archives and radio

```sh
curation/.venv/bin/python -m curation.import_media \
  --archive-root curation/outputs/archive_sources --radio-root curation/outputs/radio \
  --output curation/outputs/imported-media
curation/.venv/bin/python -m curation.prepare \
  --manifest /path/returned/by/import/manifest.jsonl \
  --output curation/outputs/imported-prepared --max-recordings 1 --window-seconds 28
```

The importer checks finalized metadata against actual file hashes, sizes, audio streams and durations. It writes immutable content-addressed snapshots and a report of exclusions and exact-byte duplicates. Rerunning discovers newly completed downloads without changing prior snapshots. Canonical rights, language, split and provenance survive preparation; changed annotations require a fresh preparation output root. These 28-second windows remain transport units, not detected speech or aligned TTS utterances. The first real import contained 29 recordings from 11 sources, 21.3561 measured media hours, with no claim of unique speech or linguistic verification.

## Join text review and export collected drafts

```sh
curation/.venv/bin/python -m curation.attach_reviews \
  --manifest curation/outputs/asr-client-hil/manifest.jsonl \
  --reviews curation/outputs/review-new-hil \
  --output curation/outputs/review-new-hil/joined-manifest.jsonl
curation/.venv/bin/python -m curation.cli export \
  --manifest curation/outputs/review-new-hil/joined-manifest.jsonl \
  --output curation/outputs/new-hil-export-v1 --profile curation
```

Joining requires matching input/configuration, sentence/audio hashes, and the saved provider response. It rechecks the audio bytes and exact review contract, attaches only validated reviews, and reports missing/ineligible rows. It never changes transcripts, rights or split assignments, and never grants audio verification. A changed input requires a new output snapshot; rerunning the same join is idempotent. Invalid model responses stay outside the attached reviews. Curation export includes unreviewed drafts; training export continues to require audio verification and source permission.

## Publish immutable curation exports

```sh
curation/.venv/bin/python -m curation.publish \
  --export-root curation/outputs/new-hil-export-v1 \
  --prefix curation/a-new-versioned-prefix \
  --receipt curation/outputs/publication-receipt.json
```

The target dataset must already exist and be private. The publisher stages only the export summary, canonical manifest, README and explicitly listed Parquet shards; logs and arbitrary files are excluded. It validates file hashes and row consistency, enforces file/byte limits, rejects an occupied prefix, and uses a parent-commit condition to detect concurrent updates. Remote bytes must match at the immutable resulting revision before a receipt is written. Register the new prefix in `docs/cards/chika.md` and upload that card after verification; raw backups should never be listed as train splits. If commit succeeded but verification was interrupted, the explicit `--verify-existing` mode checks that exact prefix without uploading or overwriting it.

The first collected pilot configurations were uploaded at revision `db224906b9ca13c8eb14b382443f9080c99c62d1`. All nine exported/card files passed remote checksum checks; all five rows decoded locally and both configurations loaded through Hugging Face streaming. This validates format/integration only.

## Five-language collection targets

`curation/targets.json` is the shared target and RSS supervision registry: 250 unique retained/transcribed hours each for Filipino/Taglish, Hiligaynon, Cebuano, Bikolano and Ilocano (1,250 hours overall). Raw acquisition limits provide headroom but are not success criteria. Hiligaynon's identified archives remain uncapped as requested. Cebuano uses 300 raw hours with a 75-hour per-show cap. Bikolano and Ilocano currently use 300-hour RSS limits without per-show caps because their discovered sources are scarcer; domain concentration must be measured during curation.

Use the same discover/download/prepare/review/export commands with `cebuano_sources.json`, `bikolano_sources.json`, or `ilocano_sources.json` and isolated `outputs/<language>/rss` roots. `curation.status` and the storage supervisor read all five registered roots. Bikolano is a collection label, not a claim of one uniform variety: retain source variety evidence and uncertainty. RSS preparation carries full source metadata into provenance. New-language Boses ASR routes have **not** been validated, so do not send those languages to the Hiligaynon or Filipino model merely because it accepts a connection.

Text-review prompt v3 adds Cebuano, Bikolano and Ilocano, preserves code-switching and prohibits imposing another language's grammar. Its changed prompt hash separates it from v2 caches. Existing v2 review attachments remain reproducible using their saved configuration and raw responses. This prompt update is not evidence of model accuracy in the added languages; linguistic/audio evaluation remains required.

## Speech-activity candidates

```sh
curation/.venv/bin/python -m curation.vad \
  --manifest curation/outputs/prepared-hil/manifest.jsonl \
  --output curation/outputs/vad-hil-batch-v1 --max-minutes 2 --max-rows 4
```

This stage runs CPU Silero ONNX inference, checks the pinned model SHA256, and creates padded nonoverlapping speech candidates with exact absolute source-sample offsets. It uses the MIT-licensed [upstream model at pinned revision 066409f](https://github.com/snakers4/silero-vad/tree/066409f3d9db45a30d21dfcbf9147826ffdd7052), with our own hysteresis and gap-aware cuts. The maximum is 27 seconds, leaving headroom below the observed Boses automatic cut near 28 seconds. Continuous speech can still require hard cuts; final word alignment remains separate.

Source audio is unchanged. Every derivative starts with empty pending transcription and no inherited text/audio approval. Parent metadata, source questions, rights and frozen splits remain in provenance/the appropriate fields. Output snapshots bind configuration, model/runtime, input manifest and parent hashes; changing any input requires a new output directory. Counts report padded candidate duration, which includes retained silence. VAD does not verify language, music overlap, speaker identity or transcription quality and is not benchmarked on these language groups.

Small real checks covered Hiligaynon, Filipino, Cebuano, Bikolano and Ilocano. Four short Hil/Fil candidates subsequently passed ASR finalization. This is bounded integration evidence, not proof that the earlier bulk server failure is resolved. No new-language ASR route is implied.

## Count source coverage without duplicate windows

```sh
curation/.venv/bin/python -m curation.account \
  --manifest hil=curation/outputs/review-new-hil/joined-manifest.jsonl \
  --manifest fil_taglish=curation/outputs/asr-client-fil/manifest.jsonl \
  --output curation/outputs/coverage-audit.json
```

The audit rechecks audio hashes and unions exact source intervals by original audio hash and language group. Repeated manifests and overlapping windows count once, and cross-group source hashes are flagged. Draft transcripts, recorded review status and audio-verified coverage are separate. Coverage still includes padding and cannot establish unique retained speech: re-encodings, excerpts and simulcasts require acoustic deduplication. Evaluation reservations are never converted to training rows. Target completion is deliberately not inferred from these partial metrics.

## Back up imported archives

```sh
curation/.venv/bin/python -m curation.archive_backup /path/to/imported/manifest.jsonl \
  --state-dir curation/outputs/archive-backup --max-files 4 --max-bytes 536870912
# Add --upload to run the bounded private-Hub backup.
```

The default is a local verification/plan. Upload rechecks full-recording audio hashes, sizes, duration and acquisition metadata, then writes content-addressed originals and sanitized source metadata. Pinned remote LFS hashes or downloaded bytes must verify before a receipt is issued. Local plans, metadata, per-object progress and immutable receipts permit restart after transport failures; already verified objects are reused without another upload. Concurrent repository changes fail the parent-commit condition safely. This adapter never deletes local originals; archive eviction remains a separate unfinished step.

A real 30.021-second Bikol archive backup passed remote verification at commit `9b41b04eca96af05c7d524e327f0af0580117d11`; a second run succeeded with commit creation disabled. Four radio recordings (63,062,567 bytes including metadata) also finished verification, with receipt revision `6bf32ac31fde5c3cb20c9d40012d1169f80a5392`. Local originals remain intact.

The bounded segmented ASR retry uses only the previously prepared Hil/Fil recordings (about two hours total), serial concurrency one, zero transport retries, a 60-second per-window timeout and a two-consecutive-failure stop. Its private state is `outputs/vad-asr-batch-v1/worker.json`; this is an experiment, not an unattended full-corpus ASR rollout. `curation.status` reports its actual process observation and unions its successful source intervals with earlier drafts, so retries do not inflate covered hours.

### ASR connection-reuse investigation

Read-only inspection of `/Users/timsan/nyo-serve` found sequential multi-finalize socket reuse in `bench/ten_e2e.py:148`. In `modules/asr_fw/server.py:198`, finalization clears finalized PCM but preserves the cumulative sample clock; emitted times remain session-relative. The gateway can synthesize `finalize_end` after an error, so a matching ID must never replace error and coverage checks. No wire reset command was found. Local support is not verification of the deployed revision.

A future reuse client must preserve raw wire events, a verified session handshake and exact session-start samples for each item. It must subtract the session offset before applying original-source offsets, exclude cached items from the sent sample clock, wait for each item's valid completion before sending the next, and discard an uncertain session. Receipt/config versions must distinguish this transport. The running batch deliberately retains the tested one-connection-per-item client. The later bounded persistent-client checks below establish protocol compatibility; sustained throughput gains remain unmeasured.

### Persistent ASR client

`curation.asr_session` implements a separate, versioned persistent transport. It defaults to two items, uses one fixed model/language session, and stops on the first uncertain or failed item without retrying. Both per-batch and per-item OS locks protect output. Raw wire events and the checked session handshake remain intact; exact session sample offsets distinguish successive clips from original-source offsets. Cached items send no audio and do not advance the session clock. A completed attempt can recover after an interrupted cache commit; uncertain attempts require reconciliation instead of automatic paid replay.

```sh
curation/.venv/bin/python -m curation.asr_session \
  --manifest /path/to/prepared-candidates.jsonl --output /path/to/new-results \
  --key-file /path/to/speech-key --language hil \
  --model fw/whisper-large-v3-pld-hil-norm --max-items 2
```

Two live Hil and two live Fil items passed on one connection per language, including cumulative-clock checks. Both caches then passed with network connections disabled. This establishes bounded deployed protocol compatibility, not sustained throughput or linguistic accuracy. The existing larger batch remains on the original client. Word timestamps stay in raw session coordinates and are not represented as source-aligned annotations by this client.

### LLM request identity

Prompt/contract v4 uses a deterministic short request identifier for long row IDs. A live v3 rejection was caused by the model mistyping the original 64-character ID. The response must match the short identifier exactly before it is mapped back to the full local ID; no approximate matching or repair is allowed. Full input/audio/text/configuration hashes remain the cache and attachment identity. Historic responses without the encoding field retain their original full-ID contract. Two repeated live checks and five new draft checks passed this contract; that is not an accuracy benchmark or audio verification.

### Additional-language ASR candidates

`curation/asr_candidates.json` records available SapinSapin Cebuano, Central Bikol and Ilocano checkpoints and pinned revisions from the Hub API. The current local serving registry only configures Fil/Hil routes. Candidate model cards declare `cc-by-nc-4.0`; the metadata is not commercial-use clearance. No candidate weights were downloaded or deployed, and no new-language ASR route is claimed. Central Bikol must not be silently used as a label/model for all Bikol varieties.

### First segmented curation snapshot

`hil_collected_vad_200_v1` on private Chika contains 200 ASR draft candidates totaling 489.686 padded seconds. Five validated text-only reviews are attached; no audio approvals were granted. All 200 embedded audio arrays matched their source derivative PCM exactly. The export's five files passed immutable-revision verification at `ae2abab424fcf6535672d3fdcf8a1f11117a07b9`. This snapshot overlaps earlier pilot sources and is not additive unique speech coverage.

Subsequent bounded-batch failures included four `finalize_timeout` responses on candidates lasting 0.588–2.508 seconds. Avoiding the 28-second boundary does not eliminate every server failure. Failed items remain outside completed manifests; no claim of a production server fix is made.

### Transfer and review circuit breakers

New YouTube archive/inventory workers monitor media-byte growth and free disk space during transfers. The default limits are five minutes without media growth, six hours total, and 25 GiB free disk. Logs and metadata do not count as download progress. Failures terminate the worker-owned transfer process group and retain partial downloads for resumption. Existing running workers retain their earlier timeout behavior until their next normal launch; do not restart healthy downloads just to pick up this change.

LLM review stops after two consecutive failures by default (`--max-consecutive-failures`). Successful cached or new reviews reset that counter. Failed reviews stay excluded, and no transcript is rewritten. Publish newly attached reviews under a new immutable export prefix rather than replacing a previously published snapshot.

Operational checkpoint (2026-11-14 UTC): the bounded Hiligaynon batch stopped early with 667 successful drafts (0.44349 padded hours) and nine failures; no automatic restart was made. A separate persistent Filipino session completed 13 drafts (0.00997 padded hours) before its first error and stopped. The 200-row Hiligaynon review run finished with 194 valid text reviews and six failures. A new local manifest at `curation/outputs/collected-hil-vad-200-reviewed-v2/manifest.jsonl` attaches those 194 reviews with zero transcript changes or audio approvals. It overlaps the already published 200-row snapshot and must not be added as new unique audio. The reviewed version is now published as `hil_collected_vad_200_reviewed_v2` in private `sapinsapin/chika`: data revision `040d715a16820092e20c1e5a4bb40fccecee0213`, dataset-card revision `c868cbd6adc8d6bf0394f44b90aeac899c9964ac`. All five remote artifact hashes and all 200 embedded PCM arrays were checked; audio and sentences are unchanged. Loading the pinned configuration through Hugging Face streaming returned exactly 200 rows: 194 `text_reviewed` and six `draft`. Acquisition remains the priority.

### Hiligaynon archive backup checkpoint

The immutable archive import `75c7187cddc5553c419da5b094bf1021f72f7489d126a0958c50ec4582389f53` contains 49 completed recordings from eight sources: 40.72797 measured media hours and 3,342,182,733 bytes. Two incomplete acquisition records were excluded. This is raw media, with no speech/language verification or training permission implied.

A detached backup job was launched under `curation/outputs/archive-backup-hil-49-v1/` with a 100-recording / 4 GiB bound. Inspect its `worker.json`, actual process, log and `state/receipts/` before any recovery; a running marker alone is insufficient. `curation.status` now reports this job's actual process observation and local receipt-file count. Do not equate receipt-file count with newly verified unique recordings. Local originals are retained, and no archive eviction is enabled.

To reproduce the bounded plan against that imported manifest, then upload after inspecting the plan:

```sh
curation/.venv/bin/python -m curation.archive_backup   curation/outputs/imported-media-latest/snapshots/75c7187cddc5553c419da5b094bf1021f72f7489d126a0958c50ec4582389f53/manifest.jsonl   --state-dir curation/outputs/archive-backup-hil-49-v1/state   --max-files 100 --max-bytes 4294967296
# The same command with --upload writes to the existing private Chika repository.
# Do not run it concurrently with the active worker.
```

Re-running after confirmed termination revalidates remote content-addressed objects and resumes missing work. New recordings require a fresh import snapshot; this frozen job does not include later downloads. Status source-interval accounting also includes the persistent Filipino drafts and the published 194-review Hiligaynon manifest, without adding overlapping segments twice.

### Ilocano radio expansion

The [network player](https://www.bomboradyo.com/live-stream) explicitly maps **Bombo Radyo Vigan** to `https://stream.zenolive.com/ebcyau9nz9duv`. Source evidence and rejected alternatives are recorded in `curation/ilocano_radio_sources.json`. The station is an Ilocano candidate; actual speech and code-switching remain unverified. A 30-second smoke capture completed (30.016 measured seconds, 241,268 bytes) and imported successfully with `--language ilo`. A separate two-hour capture was launched, with status under `curation/outputs/ilocano/radio/bombo-vigan/` and launch metadata under `curation/outputs/ilocano/vigan-capture-2h/`.

```sh
curation/.venv/bin/python -m curation.acquire_radio --station bombo-vigan   --seconds 7200 --output curation/outputs/ilocano/radio --min-free-gib 25
curation/.venv/bin/python -m curation.import_media   --radio-root curation/outputs/ilocano/radio --language ilo   --output curation/outputs/imported-ilocano-radio
```

Check actual live processes before starting another capture. Each invocation records a new time interval, not a resume of missed live audio. Keep radio roots separated by provisional language. Import rejects records whose declared acquisition language conflicts with the requested import language, preventing accidental default-Hiligaynon relabeling. Active or incomplete captures are excluded.

Discovery did not justify importing the station website's linked YouTube channel as Ilocano: it is the national Bombo network, with no Ilocano/Vigan/Laoag title matches in the latest 1,000 streams inspected. The local streaming page also embedded the already known Bacolod stream. Neither page location nor a search title is sufficient evidence of the audio's language. A terminated search-result channel was excluded without attempting access circumvention.

### Targeted archive recovery checkpoint

Aksyon recording `1JsYMPNU2z8` had a terminal fixed-one-hour timeout, while the main round-robin collector had moved to other recordings. After confirming no live transfer for that ID, one targeted retry was launched under `curation/outputs/archive-recovery-1JsYMPNU2z8/`. Its previous acquisition record is preserved in `prior-attempt.json`; its one-item inventory and worker state are alongside it. The retry uses the new media-progress/disk guard and existing yt-dlp partials. Other collectors were not restarted. This retry remains incomplete until its final acquisition record and media checksum are verified.

For a future recovery after confirming this retry has terminated, the existing CLI accepts the isolated inventory:

```sh
curation/.venv/bin/python -m curation.acquire_archives --source youtube   --inventory curation/outputs/archive-recovery-1JsYMPNU2z8/inventory.json   --source-id aksyon-streams
```

Do not launch a second transfer for the same ID. Retain the prior record and inspect the current error before retrying; live/upcoming exclusions require fresh discovery, not blind download retries. The default all-source queue does not automatically retry a failed item during the same pass.

The 49-recording archive backup has produced its first verified recording receipt: revision `b53ba777174932bceb4c3883b126460746ca43d1`, two objects totaling 159,443,623 bytes. A separate read-only audit revalidated both objects at that pinned revision. This proves one recording checkpoint, not completion of the full backup; remaining work stays with the existing live worker.

### Source-balanced preparation batch

`curation.prepare --round-robin-sources` cycles sorted source IDs while preserving the original within-source order. It balances recording selection, not hours, speakers or languages. Sources with fewer available recordings can be exhausted first. Without the flag, the previous input order is preserved. `--max-recordings` still limits newly prepared recordings in a run; cached records do not consume that budget, so rerunning a completed preparation command can add further recordings.

A bounded five-language worker was launched at `curation/outputs/five-language-preparation-v1/`, preparing up to ten new RSS recordings per language group, then VAD candidates. The selection audit found approximately 27.1 raw hours across 9 Filipino/Taglish, 8 Hiligaynon, 10 Cebuano, 3 Bikolano and 4 Ilocano sources. Counts and durations are provisional until stage summaries finish. This batch uses low process priority and a 45 GiB disk floor before stages and during preparation. VAD is capped at 600 input minutes per group and 100,000 input windows; it does not certify speech, language, speaker identity or rights. ASR is not invoked.

```sh
curation/.venv/bin/python -m curation.prepare   --rss-root curation/outputs/cebuano/rss   --output curation/outputs/five-language-preparation-v1/ceb/prepared   --max-recordings 10 --round-robin-sources --min-free-gib 45
curation/.venv/bin/python -m curation.vad   --manifest curation/outputs/five-language-preparation-v1/ceb/prepared/manifest.jsonl   --output curation/outputs/five-language-preparation-v1/ceb/vad   --max-minutes 600 --max-rows 100000
```

Do not run these commands concurrently with the active batch. For recovery, inspect actual process state and each stage's return code first; skip completed stages. If preparation is extended after VAD was created, use a new VAD output root because its input/configuration binding is immutable. `curation.status` reports actual worker observation and completed stage summaries. These candidates overlap earlier pilots; use source-interval accounting instead of summing configurations.

The targeted Aksyon retry `1JsYMPNU2z8` completed successfully, measuring 3,400.901 seconds (56.7 minutes). Its acquisition record now contains the completed media hash; the prior fixed-hour timeout record remains preserved in the recovery directory.

### Independent derivative audit

Use `curation.audit_vad` after a VAD stage completes. It reads the independently supplied parent manifest, rechecks parent and candidate file hashes, compares every decoded candidate sample with the exact parent slice, checks relative/absolute offsets and timestamps, rejects duplicate IDs and within-parent overlaps, and checks inherited language, prompt, rights and split metadata. It does not rerun the VAD heuristic or validate speech content, language, transcripts, speakers or rights claims. Only parents referenced by candidates are decoded; silence-only parents may have no candidate.

```sh
curation/.venv/bin/python -m curation.audit_vad   --parents curation/outputs/five-language-preparation-v1/fil_taglish/prepared/manifest.jsonl   --candidates curation/outputs/five-language-preparation-v1/fil_taglish/vad/manifest.jsonl   --report curation/outputs/five-language-preparation-v1/fil_taglish/vad-audit.json
```

Real-data checks passed on all 1,375 earlier Hiligaynon candidates (1.19318 padded hours) all 5,189 new Filipino/Taglish candidates (6.43737 padded hours), and all 2,755 new Hiligaynon candidates (3.52511 padded hours). The Filipino batch began with 859 transport windows / 7.12117 raw hours; 857 windows produced candidates. Candidate runtime is not unique verified speech. Audit reports bind both input manifests by SHA256. Regression tests include a wrong audio slice with a freshly updated checksum, altered timestamps, conflicting embedded parents and changed rights metadata.

### Completed five-language candidate checkpoint

The first source-balanced batch completed all ten preparation/VAD stages successfully. It contains 50 recordings, 27.07428 transport hours and **19,357 candidates / 24.40479 padded hours**. Every exported candidate passed the independent PCM/offset/metadata audit. The machine-readable [preparation checkpoint](../curation/PREPARATION_CHECKPOINT.json) records per-language counts, source counts and input/output manifest hashes.

| Group | Candidate segments | Padded candidate hours |
| --- | ---: | ---: |
| Filipino/Taglish | 5,189 | 6.43737 |
| Hiligaynon | 2,755 | 3.52511 |
| Cebuano | 5,800 | 7.83103 |
| Bikolano | 1,975 | 3.07390 |
| Ilocano | 3,638 | 3.53738 |

These are local pending-ASR candidates, not completed transcribed datasets. Source labels, music/overlap, speakers, rights and acoustic near-duplicates remain unresolved. Parent/source overlap with earlier pilots means these durations cannot be added as new unique retained hours. No additional ASR calls were made for this batch.

A separate text-only review worker at `curation/outputs/collected-hil-remaining-467-v1/` processes the 467 successful Hiligaynon ASR rows outside the earlier 200-row snapshot. Input SHA256 is `43a283cfd684605db7c24eba37aa82443ae792957b1da8c319b39bb2352fd444`. It retains drafts, uses one attempt per item and stops after two consecutive failures. It is still running; inspect actual process and `worker.json`/`worker.log` before recovery. Existing published exports remain unchanged. After termination, attach validated reviews to a new manifest and reconcile those rows with the earlier snapshot before publishing a new combined configuration.

See [ASR incident diagnosis and offline reconciliation](asr_incidents.md) for the distinction between Hiligaynon gateway timeouts and the corrected Filipino usage-telemetry parser failure.

### Embedded-audio publication checks

`curation.publish` now decodes every embedded Parquet audio value before any Hub request. It requires nonempty FLAC PCM16 mono at the profile sample rate (16 kHz curation/ASR, 24 kHz TTS), checks decoded duration against the manifest with the exporter's 0.1-second tolerance, and compares complete `provenance_json` with the manifest plus explicit original/export sample rates. Shard hashes alone do not establish this. Training-profile publication additionally enforces nonempty audio-verified text, recorded training permission and assigned splits, even for a manually constructed export.

Both existing 200-row Hiligaynon exports passed the stronger checks. Tests reject empty embedded bytes, wrong rate or duration, altered provenance and draft rows disguised as a training profile before network calls. These checks establish file/metadata integrity, not transcript accuracy or rights beyond recorded evidence.

The first 100-row Filipino review completed with 99 valid reviews and one unattached failure. All 100 exported PCM arrays match their local derivatives and all transcripts are unchanged. The private export is at `curation/collected-fil-vad-100-v1`, data revision `1250b65d6fe38f3face5f42576d9bff123a414c8`; it contains 292.476 padded seconds. This overlaps earlier Filipino pilot intervals and is not additional unique audio. Publication receipts and audio checks are under `curation/outputs/fil-review-100-v1/`.

The dataset card exposes `fil_collected_vad_100_v1` at revision `0a1441d12d549f03c63d625ad4a1a698d73cb81a`. Pinned Hugging Face streaming returned exactly 100 rows: 99 text-reviewed and one draft. The separate full Filipino pilot v3 subsequently finished all 482 segments with zero failures, 0.5346333 padded transcribed hours and no empty results; this larger set is not yet the published 100-row configuration.

### Backup recovery after a concurrent Hub commit

The 49-recording Hiligaynon backup stopped with an explicit HTTP 412 precondition failure after another dataset commit landed during an upload. Three recordings had already received verified receipts. The failed worker was confirmed absent before resuming the same immutable plan/state directory; its previous terminal state is retained in `curation/outputs/archive-backup-hil-49-v1/worker-attempt-1.json`.

`curation.archive_backup` now retries only explicit HTTP 412 commit conflicts, at most three commit attempts per object. Each retry fetches the current repository privacy/head, verifies whether the same content-addressed object already exists, and rechecks local bytes before committing. Conflicting remote bytes still fail; authentication, rate-limit and generic network errors are not automatically retried. Conflict evidence is recorded under `state/conflicts/`. The existing fixed plan remains 49 recordings / 4 GiB; no local deletion is authorized.

Resumption revalidated three existing audio objects totaling 304,916,101 bytes. It generated additional pinned receipts for those same objects, so six receipt files represented only three distinct recordings at that checkpoint. Deduplicate receipt audio hashes before counting protected recordings or bytes. Full backup completion remains pending with the resumed worker. Fourteen archive-backup tests passed, including rebasing a one-time 412 and stopping after three repeated conflicts.

### Cebuano news archive pilot

The delegated acquisition pilot downloaded two official **Balitang Bisdak** episodes: `TVve_CROifU` (1,891.521 seconds) and `xzryb6E-vK4` (1,733.281 seconds), totaling 1.006889 raw hours / 48,568,079 bytes. Both passed ffprobe and independent SHA256 revalidation. [GMA identifies the show as its Cebuano newscast](https://www.gmanetwork.com/regionaltv/specials/13621/gma-regional-tv-newscasts-balitang-bisdak-and-balitang-amianan-no-1-in-ratings/story/); each video was matched through its official episode page, rather than assigning Cebuano to the entire channel. Exact spoken language remains provisional, including possible English/Filipino/Waray segments. Rights remain unknown.

The [source registry](../curation/cebuano_archive_sources.json) points to preserved page evidence, inventory and download root. The verified receipt is `curation/outputs/cebuano/archive-expansion/pilot-receipt.json`. Canonical import snapshot `f5f8630fd1a176e181f66e5b661f71296df2b35e9a544a0de17a84ef02458a98` contains both recordings, with no exclusions or exact-byte duplicates:

```sh
curation/.venv/bin/python -m curation.import_media   --archive-root curation/outputs/cebuano/archive-expansion --language ceb   --output curation/outputs/imported-cebuano-archive
```

The archive importer expects source subdirectories, so use the parent of `audio/`, not `audio/` itself. The 30 URLs discovered on one publisher index page are candidate episodes/clips, not 30 measured full episodes. Expansion is delegated with a 30-recording / 20-hour ceiling and 45 GiB floor; keep full-episode/excerpt distinctions and overlap uncertainty. No additional hours are claimed until completed downloads are measured. No Cebuano ASR calls have been made.

### Combined Hiligaynon release and completed Cebuano expansion

The [release index](dataset_releases.md) records the new `hil_collected_vad_667_v1` configuration: 667 transcripts, 531 validated text reviews, 136 draft-only rows, 1,596.56 padded seconds. Both prior review directories were independently validated during attachment; sentences and audio stayed unchanged. All 667 embedded PCM arrays, ten remote artifacts and the pinned streamed row/status counts passed verification. This supersets the earlier 200-row Hiligaynon releases; it is not additive unique coverage.

The two review responses that triggered the earlier stop failed consistency checks (a no-issue verdict with flagged issues, and a quoted span absent from the transcript). Neither was repaired or retried. After confirming other review requests were succeeding, a separate bounded worker under `curation/outputs/collected-hil-remaining-125-v1/` started only the previously unattempted 125 rows. Keep its future attachments in a new immutable snapshot.

Cebuano expansion completed all 13 selected clips. Combined with the two full-episode pilots, 15 recordings total 1.463365 raw hours / 70,738,694 bytes. All hashes were rechecked; canonical import snapshot `339c11f38a1f671a472b19c9d905767d89362d63eea3d61066ceaaecd0fa87ce` has zero exclusions or exact-byte duplicates. Acoustic/episode overlap and actual language remain unverified. The [frozen public inventory](../curation/cebuano_archive_inventory.json) preserves all fifteen video IDs, publisher episode URLs, channel identities and full-episode/clip selection scope so reacquisition does not depend on private working files. Upcoming content and conservative same-date candidates remain excluded.


### Reviewed snapshot continuation and next Filipino batch (15 November)

The completed Hiligaynon review join under `curation/outputs/collected-hil-667-v2/` contains 667 rows: 654 validated text reviews and 13 rejected reviews left as drafts. The Filipino join under `curation/outputs/collected-fil-482-v1/` contains 482 rows: 471 validated text reviews and 11 draft-only rows. Both joins preserve transcript text and audio, with zero audio approvals. Their exports passed independent PCM comparisons for every row. See [release evidence](dataset_releases.md) for published configurations and pinned revisions; a local export alone is not publication.

A new bounded review worker under `curation/outputs/fil-review-next-500-v1/` selects 500 nonempty ASR drafts from the ongoing Filipino session, excluding IDs in every earlier Filipino review input. `selection.json` and the input hash freeze the selection. It uses one request attempt per item and stops after two consecutive invalid responses. Inspect the actual process, `worker.json`, and `reviews/summary.json` before recovery. This batch does not retry the eleven previously rejected reviews, and its future results must be attached to a new immutable snapshot.

When a publication receives an explicit HTTP 412, another writer has advanced the Hub branch. The publisher now makes at most three commit attempts, re-reading repository privacy/head and the destination prefix and rechecking staged bytes before every attempt. Successful receipts record rejected parent revisions in `parent_conflicts`. It never overwrites an occupied prefix and does not retry generic network, authorization, quota, or HTTP 409 failures. If the bound is exhausted, re-run only after checking the prior process is terminal and inspecting the failure; export validation runs again. If the commit succeeded but receipt verification failed, use `--verify-existing` instead. Never overwrite an existing snapshot or restart the acquisition/backup jobs merely to resolve a publication conflict.


### Next Filipino preparation batch and backup recovery (15 November)

`curation/outputs/fil-preparation-next-20-v1/` freezes 20 additional downloaded recordings across nine Filipino/Taglish candidate sources, totaling 15.977 raw hours. Selection cycles source IDs and excludes all ten original hashes in the preceding five-language Filipino preparation batch. It does not claim acoustic deduplication or verified language. `selected-recordings.json` is immutable input; `selection.json` records the bounds and excluded hashes.

The worker prepares separate 16 kHz windows, exports per-recording checkpoints, runs VAD with a 1,000-minute/100,000-row bound, then independently audits PCM and offsets against the parent manifest. It enforces a 45 GiB disk floor at preparation and before VAD. The currently running ASR batch's manifest is not edited. Inspect the worker's actual process, stage, and audit result before assigning these candidates to a subsequent ASR session. `curation.status` now observes continuation preparation workers separately.

The Hiligaynon full-recording backup's second attempt ended with explicit HTTP 409 (“another commit operation is in progress”). After verifying its process had exited and publication had completed, recovery preserved `worker-attempt-2.json` and resumed the same plan/state. Existing immutable remote objects are reverified before reuse. This was a deliberate recovery, not an automatic retry of an uncertain network failure; originals remain intact.


### Cebuano full-broadcast expansion completed (15 November)

Seven additional distinct dated Balitang Bisdak full broadcasts completed: 3.659496389 measured raw hours, 173,892,278 bytes. All seven source hashes were rechecked against the imported canonical manifest. The archive now contains 22 recordings (nine full broadcasts and thirteen clips), totaling 5.122861667 raw hours. The public selection and publisher links are preserved in `curation/cebuano_archive_inventory.json`.

Private evidence: `curation/outputs/cebuano/archive-expansion/full-batch/receipt.json`; canonical snapshot `05b4d9fb5cc9c3f0bcac5cac8a8ea5440892b849b1a2fe4d7a55bb2d2f94d101` under `full-batch/canonical/snapshots/`. Import the archive parent, not its `audio/` directory: the latter produced an empty snapshot during the automatic first import, which was preserved and superseded by the verified 22-row snapshot. Different dates avoid known same-episode overlap but do not prove acoustic uniqueness. No verified-language or transcribed hours are claimed.


### Audited Filipino continuation queued for ASR (15 November)

The twenty-recording continuation completed preparation and VAD: 10,118 candidate segments, 14.865410226 padded hours. Independent audit checked all derivative PCM/offsets and inherited metadata against 1,914 parent windows. The full selection, hashes and audit are recorded in `curation/FIL_NEXT_PREPARATION_CHECKPOINT.json`. This is not verified clean speech.

The supervised queue `curation/outputs/fil-asr-queue-v1/` waits for `fil-persistent-batch-v4` to finish successfully with all 5,189 rows and no failures. It checks the actual dependency process while waiting, then launches `fil-persistent-batch-v5` (10,000 rows) and `v6` (118 rows) sequentially. Each input is frozen and hash-checked; each stage checks a 45 GiB disk floor. Any failed dependency, missing process, changed input, or unsuccessful stage stops the queue for inspection. The queue does not retry ASR failures or rerun non-queued stages. Do not manually start v5/v6 while this queue is live.

### Ilocano archive pilot completed (15 November)

Five distinct dated TV Patrol Ilocos broadcasts from the verified ABS-CBN News channel were downloaded and imported: 2.670427352 raw hours, 133,950,519 bytes. All five hashes were independently rechecked against canonical snapshot `fce4019a7f2ffed23cf19cc8f42e5f6eb20beb879028d5a07de804daed2920d9` under `curation/outputs/ilocano/archive-expansion/canonical/snapshots/`. Receipt: `curation/outputs/ilocano/archive-expansion/receipt.json`; public source evidence: `curation/ilocano_archive_inventory.json`.

Publisher titles and Ilocano descriptions support provisional source selection; language and speech remain unverified, permissions unknown, and no ASR hours are claimed. Regional coverage alone was insufficient evidence to label GMA Balitang Amianan recordings Ilocano, so those were excluded from this pilot. The status report now includes the separately stored Ilocano archive.


### Review continuation after Hiligaynon timeout (15 November)

All 51 successful Hiligaynon continuation drafts received validated text-only reviews; the failed ASR item was excluded. They join the prior 667 rows in `collected-hil-718-v1`, giving 705 text-reviewed and 13 draft-only rows. Audio and sentences remain unchanged. See [release evidence](dataset_releases.md) for verified publication revisions.

The next Filipino 500-item review batch completed with 490 validated reviews and ten rejected responses. Its local join with the earlier 482 rows is `curation/outputs/collected-fil-982-v1/`: 982 rows, 961 text-reviewed, 21 draft-only, 3,730.272 padded seconds. Every embedded PCM array and sentence matched its source manifest. Upload completion is established only by its Hub receipt, not by export existence.

`curation/outputs/fil-review-next-1000-v1/` freezes the next 1,000 completed nonempty Filipino drafts, excluding all 982 IDs in previous review inputs. This does not retry rejected reviews. The same one-attempt/two-consecutive-failure policy applies; retain all raw responses and attach only content-bound validated results.

Ilocano acquisition is continuing through a separate 16-recording inventory (8.062 metadata-estimated hours), with a 45 GiB floor. `outputs/ilocano/archive-expansion/expansion_inventory.json` and `expansion_download.log` preserve the selection and attempts; a separate finalizer waits for actual acquisition-process termination before verifying media and creating a canonical union/partial receipt. Inspect both workers before recovery, and do not count pending metadata estimates as measured hours.


### Publication conflict handling verified (15 November)

The first 982-row Filipino publication attempt received an explicit HTTP 412 and exited without a receipt. This exposed the repeated manual-recovery cost of simultaneous archive backups and draft publication. `curation.publish` now handles only that explicit rejected-parent case with a three-attempt limit and renewed privacy/prefix/staged-byte checks. Four focused regressions cover a successful rebase, a newly occupied destination, changed privacy, and exhausted retries; all fourteen publication tests passed. A new bounded publication attempt was then started against the same immutable local export. This change does not restart active acquisition, ASR, or backup workers.


The 982-row Filipino publication retry completed and all thirteen artifacts were verified. Its configuration `fil_collected_vad_982_v1` was registered and read back at pinned revision `f24f5047cfa44c4a2644f2bd7c1ddb566036c5c5`, yielding exactly 982 IDs/sentences with 961 text-reviewed and 21 draft rows. Data revision: `2b04e79bc36bf65d7d020e9322a7ca529e7d84a4`. Receipts and streaming evidence are under `curation/outputs/collected-fil-982-v1/`. No publication worker remains pending for this snapshot.

### Reusable sequential ASR queue

`python -m curation.asr_queue` replaces one-off queue scripts for **new plans**. It supports the existing Filipino and Hiligaynon persistent client, 1–20 sequential stages, and at most 10,000 rows per stage. It does not migrate or restart the already-running `fil-asr-queue-v1` supervisor.

A plan has this structure; roots resolve relative to the plan file (absolute roots also work):

```json
{
  "version": 1,
  "dependency": {"root": "../previous-batch", "rows": 5189},
  "batches": [
    {"root": "batch-001", "rows": 500, "input_sha256": "SHA256_OF_INPUT_JSONL"}
  ]
}
```

Omit `dependency` when none is required. Each new batch root must contain a validated `input.jsonl` and `worker.json` containing `{"status":"queued"}`; its `asr/` directory must not exist. Compute the input hash with `curation.schema.digest`, and set `rows` to the exact manifest length. The queue state directory must be separate from every batch and dependency root.

```sh
curation/.venv/bin/python -m curation.asr_queue \
  --plan /absolute/path/to/plan.json \
  --state-dir /absolute/path/to/queue-state \
  --key-file /absolute/path/to/asr-key \
  --language fil --model fw/whisper-large-v3-pld-fil-norm \
  --min-free-gib 45
```

The queue checks all inputs before any stage, rejects duplicate IDs across stages, and checks frozen plan/input hashes again before launch and after completion. It requires successful dependency status, complete ASR summary, and output manifest; while a dependency is running it also checks the actual process. It does not infer completion from a missing process. Each stage checks the disk floor and proceeds only when the preceding output has the expected IDs/audio hashes and processed ASR states.

A queue lock and per-batch launch locks prevent competing launches. The batch enters `launching` **before** child creation, then records the actual PID. If interruption leaves `launching`, `running`, or an existing `asr/` directory, a subsequent run refuses to replay that batch. Inspect the child process and response ledger first; preserve uncertain paid attempts. The queue does not automatically kill a child on interruption, repair failures, or reset queued markers. Use the documented ASR cache-reconciliation procedure before constructing a new recovery plan.

Validation: nine queue regressions plus eleven persistent-client tests passed. Tests cover sequential execution, failure stopping, changed inputs, duplicate IDs, absent dependencies, wrong outputs, storage floor, interrupted launch, competing batch claim, plan mutation and separate state roots. A read-only preflight of the real 10,118-row pending Filipino plan validated both stages with zero network calls and no new worker launches. This preflight does not claim that the new generic queue has run those production stages; their existing supervisor retains ownership.


### Preemptive offload of completed Bikolano RSS acquisition (15 November)

The three registered Bikolano RSS sources reached 269 downloaded recordings, 120.908339333 raw hours, with no eligible remaining items in the current ledger. The downloader process had exited and its OS lock was free. This is finite-feed acquisition completion, not completion of the 250-hour retained/transcribed language target.

With global free space near 46 GiB, `curation/outputs/bikolano-preemptive-backup-v1/` started a single bounded offload: at most 20 recordings selected for backup (839,147,161 local bytes at selection), at most 20/1 GiB of verified local originals eligible for eviction, and a 50 GiB target. It uses the existing `Supervisor.step` backup/checkpoint/eviction sequence under the **same global supervisor lock**; the ordinary supervisor may report busy while it runs. Other acquisition and ASR workers continue. Its 50 GiB threshold applies only to this one-shot job; normal supervisor settings remain unchanged.

The job first verifies immutable remote audio hashes/sizes and the SQLite checkpoint. Eviction independently rechecks proof, requires an idle downloader, and journals the remote-ledger transition before unlink. No eviction is implied by a running job or uploaded-file count. Check `worker.json`, `step.json`, the actual process, `hub-checkpoint.json`, and the root's eviction journal before reporting reclaimed bytes or attempting recovery. A stopped backup retains local originals.

Storage verification: all 40 supervisor, eviction, and restoration tests passed, including live-downloader exclusion, missing/corrupt backup proof, public-repository rejection, crash recovery, bounded eviction, and restoring from the same checkpoint.

The tracked Ilocano inventory now includes all 21 explicit episode references, tagged by `pilot-five` or `expansion-sixteen` selection batch. It is a reproducible source inventory, not a download-completion claim. The expansion's worker/finalizer receipts remain the authority for actual acquired media.


### Ilocano bounded expansion verified complete (15 November)

All sixteen expansion episodes completed before the archive floor was reached: 8.059983693 new measured raw hours, 406,095,199 bytes. The combined archive has 21 recordings, 10.730411045 raw hours and 540,045,718 bytes. Parent verification independently rehashed all 21 files, checked receipt durations and byte counts, matched their IDs to the tracked source inventory, and confirmed the canonical rows remain empty-transcript provisional-language drafts.

Tracked checkpoint: `curation/ILOCANO_ARCHIVE_CHECKPOINT.json`. Canonical snapshot: `b72a802054b0ac9cc5c16d31b2921e8955c70fd47e9992f2137ed78214415fa3`; manifest SHA256 `d0338bc347a21951cd35d374e4c9fb9d97debccf29e848ce0b1c817b86dd4366`. Private receipts and independent audit are in `curation/outputs/ilocano/archive-expansion/`. No exclusions or exact-byte duplicates were reported; acoustic overlap, language, speech and transcripts remain unverified.

Free space subsequently fell below the 45 GiB archive/preparation floor. New archive work is discovery-only while the verified-backup job runs; do not lower an active downloader's guard or restart it on an observation timeout. Existing RSS acquisition retains its configured floor.

### Larger Cebuano and Ilocano discovery inventories (15 November)

Two discovery-only passes added reproducible candidate inventories without downloading media:

| Candidate source | Additional candidates | Duration evidence | Tracked files |
| --- | ---: | --- | --- |
| GMA Balitang Bisdak | 300 distinct IDs and broadcast dates | Publisher grid provides no durations; no hours estimate | `curation/cebuano_discovery_inventory.json`, `curation/cebuano_discovery_receipt.json` |
| ABS-CBN TV Patrol Ilocos | 300 distinct IDs and broadcast dates | 150.342222222 hours from flat-search metadata only; not measured media | `curation/ilocano_discovery_inventory.json`, `curation/ilocano_discovery_receipt.json` |

Both selections exclude acquired IDs and known broadcast dates; story footage may still recur across dates. Parent checks verified item counts, unique IDs/dates, baseline exclusions, URL/ID correspondence, inventory hashes and the Ilocano duration sum. No candidate has become verified speech or transcribed data through discovery.

For Cebuano, the actual publisher navbar maps `balitangbisdak` to program ID `2`; the publisher's grid JavaScript derives the corresponding `grid/videos/2.gz` endpoint. The retained private 32.50 MB grid contains 30,477 records. Selection requires a dated program title and full-episode marker, excludes clips/live-style titles and known dates, and caps the result at 300. The compact receipt preserves URLs, mapping evidence and raw-grid SHA256; raw grid/descriptions remain outside Git.

For Ilocano, three bounded official-channel flat searches returned 469 entries, yielding 314 additional unique dated candidates before the 300-item cap. The receipt preserves the exact `yt-dlp --flat-playlist --dump-single-json --skip-download` commands, query URLs, limits and snapshot hashes. Uploader identity/badge comes from flat search metadata; `per_video_full_metadata_verified` is explicitly false.

**These are not download-ready inventories.** Before selecting a bounded media batch, fetch full per-video metadata (without downloading media), require the expected official channel ID, reject live/upcoming/unavailable items, check positive finite duration, deduplicate known IDs/dates again, and freeze a new bounded inventory. Respect the 45 GiB archive floor. Do not pass all 300 discovery entries directly to acquisition or infer that their dated titles prove current availability. Preserve code-switching and keep source-language labels provisional.


### Per-request ASR storage guard and waiting-queue migration (15 November)

`curation.asr_session --min-free-gib` now checks available space before each item (default 25 GiB). A low-space stop flushes completed rows and records `StorageFloor`, `request_attempted=false`, and the observed free bytes. It does not submit that item or retry a paid request. Resume only after inspecting the result/attempt ledger; validated earlier caches are reused. The generic queue passes its configured floor through to the client, so its bound applies throughout each new stage.

Tests demonstrated one successful request followed by a storage stop, then a resume that reused the first result and submitted only the previously unsent row. All 22 queue/client/status tests passed. Existing running ASR processes retain their loaded code.

The old Filipino queue `fil-asr-queue-v1` was confirmed to be waiting, with no child process, both future batches still marked queued, and no `asr/` directories. Its state was preserved in `worker-before-migration.json`; it was deliberately retired before launching any paid stage. The current v4 ASR session was not stopped or restarted.

Replacement `curation/outputs/fil-asr-queue-v2/` uses the reusable runner, the same frozen two-stage input plan, and a **30 GiB ASR floor checked before every request**. This is appropriate for already-prepared audio: the observed 4,300-plus-row ASR output occupied about 84 MiB, while new archive/preparation jobs retain a separate 45 GiB floor. The configured ASR threshold is 5 GiB above the RSS downloader floor; it does not lower media/preparation guards. `migration.json` records old/new PIDs, plan ownership and dependency counts. The v2 supervisor now exclusively owns pending v5/v6. Do not revive v1 or launch either stage manually while v2 is live.

### Metadata-preflighted next archive batches

The first twenty candidates per new discovery registry were inspected with full per-video metadata, without downloading media. Cebuano selected 19 episodes totaling 9.705833333 metadata-estimated hours; one candidate required sign-in and was excluded without bypass. Ilocano selected 19 totaling 9.773611111 metadata-estimated hours; one otherwise-valid item was excluded solely to stay below the 10-hour cap.

Frozen selections and receipts are tracked as `curation/cebuano_next_acquisition_inventory.json`, `curation/cebuano_next_acquisition_receipt.json`, `curation/ilocano_next_acquisition_inventory.json`, and `curation/ilocano_next_acquisition_receipt.json`. Parent checks rehashed all 38 selected raw metadata files and verified IDs, official channel IDs, explicit completed status, titles, finite 20–60-minute durations, unique dates, source exclusions and aggregate limits. Metadata access does not grant source rights or establish spoken language.

These selections are ready for a bounded acquisition after available space again satisfies the **45 GiB media floor**; recheck availability when starting later. No media was acquired in preflight and these estimates add zero measured or transcribed hours. Raw metadata/logs remain private under each language's `outputs/.../preflight-next-20-v1/` directory.

### NYO offload preference (user instruction, 15 November)

Use NYO's requested model route `us.kimi-k3` for substantial planning, code review, implementation drafts, metadata reasoning and documentation drafting to reduce ChatGPT usage. Prefer the NYO `offload` connector for known-safe local files: it sends them directly to NYO and saves the answer locally, without copying whole source files into ChatGPT context. Keep local execution, factual checks, tests and acceptance of changes under the orchestrator's control. Do not send credentials or blindly execute model suggestions.

The first offload reviewed `asr_queue.py` and `asr_session.py`; the request used `us.kimi-k3`, while NYO reported the answering backend as `global.moonshotai.kimi-k3.1`. The answer is saved under `~/.nyo/answers/20261115-020202-70795.md`. Record requested and returned model identities separately; do not infer region from an alias. Findings are advisory until checked against the implementation and its intentional recovery/model-alias contracts.

This preference changes how future reasoning work is delegated. It does not restart active ASR/acquisition workers or silently replace the model/configuration of in-flight text-review batches. Existing validated caches and provenance remain intact. Future review-model changes require a bounded contract check and a distinct configuration/cache identity.

### Kimi transcript-review contract check (15 November)

A Kimi-drafted smoke script was locally inspected and adjusted to preserve isolated responses instead of deleting them. Two synthetic code-switch examples (Hiligaynon–English and Filipino–English) passed the existing `curation.review` strict validator using requested model `us.kimi-k3`; both responses identified `global.moonshotai.kimi-k3.1`. Each used one request attempt, an 800-token output limit, and a 90-second timeout. Both returned `no_obvious_text_issue`; neither result verifies audio, grants approval, or demonstrates broad language quality. Source questions were excluded from the request.

Tracked evidence: `curation/NYO_KIMI_CONTRACT_CHECKPOINT.json`; private script, raw responses and validated caches: `curation/outputs/nyo-kimi-contract-smoke-v1/`. Synthetic rows must never be published as acquired or transcribed data. The fixed isolated cache preserves successful responses on reruns; inspect attempt/error evidence before any failed-call recovery. Existing GLM jobs remain unchanged. A future real-data Kimi batch must retain its separate model/configuration cache identity and existing strict gates.

Filipino ASR v4 subsequently completed all 5,189 selected clips with zero failures and 19 empty transcripts. Reported transcribed duration is 6.434397882 padded hours, not independently verified speech or unique source coverage. The existing v2 queue owns the next frozen v5/v6 stages; do not launch duplicate workers.

### First real-data Kimi review batch

`fil-kimi-review-100-v1` selects 100 nonempty completed Filipino v4 transcripts, excluding all 1,982 IDs assigned to the four prior Filipino review inputs (including the still-running GLM batch). `curation/FIL_KIMI_REVIEW_CHECKPOINT.json` freezes source/input hashes and exclusions. Requested route is `us.kimi-k3`, with a separate output/cache identity, one attempt per item, 60-second timeout, 2,500-token cap and a stop after two consecutive failures. The first response passed strict validation; this checkpoint is not a batch-completion claim.

The status collector includes this worker. Private `worker.json`, `reviews/summary.json`, raw responses and cache records are the completion/provenance authority. Never automatically rerun rejected outputs or concatenate overlapping release snapshots. Only valid results may later be attached to their exact transcript/audio identities; no source question is submitted and no text review grants audio approval.

### Reusable offline review selection

Use `python -m curation.select_review --manifest SOURCE --exclude-manifest PRIOR_INPUT --output NEW_DIRECTORY --limit 500`, repeating `--exclude-manifest` for **every** previously assigned input, including running, failed, and merely prepared batches. For the first batch, explicitly supply an empty exclusion manifest. Serialize selection across workers: separate output directories do not coordinate claims. The selector makes no API calls and launches no worker.

It validates and hashes the same bytes read from each input, rejects duplicate IDs within an input, preserves source order and complete rows, excludes all assigned IDs regardless of review outcome, skips ineligible rows, and writes `input.jsonl` plus `selection.json`. Exclusive directory creation prevents overwriting an existing selection. An interrupted write may leave an incomplete directory; inspect it rather than treating it as a ready batch.

NYO `us.kimi-k3` drafted the implementation (reported backend `global.moonshotai.kimi-k3.1`); local inspection and five focused tests passed. Applied to completed Filipino v4 with all five existing assignment inputs, it prepared 500 new clips under `fil-kimi-review-next-500-v1`. Independent checks confirmed no assigned-ID overlap and exact input/exclusion hashes. Tracked receipt: `curation/FIL_KIMI_NEXT_SELECTION.json`. This is **prepared, not launched**; inspect the current 100-item Kimi run before starting it.

### Kimi real-data pilot complete; next 500 launched

The first 100 real Filipino transcript reviews completed with zero failures. All 100 were independently attached after validating raw response/configuration/transcript identities and rehashing local audio. No transcript changed and no audio approval was granted. Triage: 55 `no_obvious_text_issue`, 41 `needs_audio_review`, four `unintelligible_text`. These are text-only judgments, not an ASR quality benchmark. `curation/FIL_KIMI_REVIEW_COMPLETE.json` freezes the audit; the private reviewed manifest remains under `fil-kimi-review-100-v1/`.

After verifying terminal success and all frozen input/exclusion hashes, `fil-kimi-review-next-500-v1` was launched using the same requested `us.kimi-k3` route, separate cache, one attempt per item and two-consecutive-failure stop. `curation/FIL_KIMI_NEXT_SELECTION.json` now records launch status. Include both batches in all future assignment exclusions. This supersedes the preceding prepared-only checkpoint; the 500-row batch is not yet claimed complete or published.

### Additional Bikol archive preflight

Kimi-assisted discovery led to a bounded 50-result Balitang Bicolandia metadata search. Three completed, public episodes were independently fetched with full metadata and matched official GMA Regional TV channel `UCVoawkpW12kgtXgInbPCrhQ`: March 1, 2021; August 26, 2022; August 3, 2023. Combined metadata duration is 5,950 seconds (1.652777778 hours). `curation/bikolano_bicolandia_pilot_inventory.json` freezes IDs, URLs, durations and raw metadata hashes. No media has been downloaded, language/variety is unverified, and rights remain unknown. New media acquisition retains the 45 GiB floor.

The official RMN Naga page links to channel `UC37sBCLnQ4Dvsdr06BGra9g`, but bounded discovery returned neither a streams tab nor a videos tab. Ownership evidence alone does not establish a usable archive. Private command/result logs remain under `outputs/bikolano/rmn-discovery-v1/`, `rmn-videos-discovery-v1/`, and `bicolandia-discovery-v1/`. Do not repeatedly retry the missing tabs.

### Bikol archive batch expanded to 17 preflighted episodes

Full metadata for twenty additional Balitang Bicolandia candidates was fetched successfully. Combining them with the three-episode pilot and applying the ten-hour cap selected **17 episodes / 9.729444444 metadata-estimated hours**; six otherwise-valid episodes were excluded by the cap. All selected IDs and broadcast dates are distinct, titles contain explicit dates, channel identity matches official GMA Regional TV, status is completed, and durations are between twenty and sixty minutes. Independent rehashes matched every saved metadata file.

Frozen inventory: `curation/bikolano_next_acquisition_inventory.json`. It includes the earlier three-episode pilot; do not acquire both inventories separately. This remains metadata-only: zero downloaded hours, unverified language/variety, unknown rights. Download only after the 45 GiB media floor is met and current availability is rechecked.

### Larger Bikol discovery and completed GLM review checkpoint

A bounded 300-result metadata-only Balitang Bicolandia search yielded 203 additional unique, dated, 20–60-minute candidates after excluding the prior preflight IDs and selected dates. Flat metadata estimates 110.917222222 hours; these are neither downloaded nor verified speech hours. `curation/bikolano_discovery_inventory.json` retains the candidate URLs, dates, estimates and exact raw-search hash. Uploader identity is deliberately unverified: every candidate needs full metadata preflight and a bounded selection before acquisition.

The existing Filipino GLM 1,000-item batch is terminal after processing all items: 952 accepted and 48 rejected strict outputs. Independent attachment revalidated transcript/configuration/raw-response identity and local audio hashes for all accepted reviews. Triage among accepted rows: 553 no-obvious-text-issue, 354 needs-audio-review, 45 unintelligible-text. No transcripts changed, rejected outputs were not repaired or replayed, and no audio approvals were granted. `curation/FIL_GLM_1000_COMPLETE.json` freezes the audit; private `fil-review-next-1000-v1/reviewed-manifest.jsonl` is ready for a later combined snapshot. It is not included in the already-frozen 5,671-row upload.

### Optional monitored archive byte budget

New `curation.acquire_inventory` invocations can specify `--max-transfer-bytes N`. The transfer guard rejects nonpositive/noninteger budgets, checks existing matching-media partials before launch, sums observed per-inode high-water sizes while the owned process runs, and checks the final files even after a successful child exit. A breach records `transfer_budget`, stops/reaps the owned process when still running, and preserves partials for inspection. Renames do not double-count an inode; separate raw/converted media do contribute separately. Without the option, no byte budget is imposed.

This is **monitoring, not a hard filesystem quota or reserved space**. Writes can overshoot between the five-second polls; a file created and removed between observations is not accounted. Metadata/log files are not included. The existing disk floor and runtime/idle guards remain necessary. Do not use this change alone to lower the 45 GiB archive/preparation floor. Active workers were neither changed nor restarted.

NYO `us.kimi-k3` drafted the guard/tests (reported backend `global.moonshotai.kimi-k3.1`); local integration completed CLI propagation and added cumulative-inode/rename checks. All 19 transfer/inventory tests passed, covering prelaunch rejection, running-child termination/reaping, oversized successful exit rejection, unchanged defaults, invalid budgets and CLI-function propagation.

### Next Kimi review input prepared from newly completed ASR

`fil-review-source-1284-v1` freezes the 780 completed v5 rows, 500 completed v7 rows and four offline cache-reconciled rows. The reusable selector excluded every prior/current review assignment, skipped eight empty transcripts and prepared 1,000 rows in `fil-kimi-review-next-1000-v1` from 1,276 eligible candidates. All selected IDs are disjoint from earlier assigned inputs. This batch is **not launched**; finish and audit `fil-kimi-review-next-500-v1` first. Include this new reserved input in future exclusions.

Tracked selection: `curation/FIL_KIMI_NEXT_1000_SELECTION.json`; input hash `4dc8710e647d9b3b6a41efbea2b0e3223f339303101c0e27351b9c6bc0663ee1`. The source is a combination of already-retained clips, not new acquisition or unique-hour evidence. No empty transcript was submitted for review.

### Kimi 500 complete; next 1,000 running

The Kimi 500-row batch processed all rows: 491 accepted strict reviews and nine rejected responses. All 491 were independently attached after response/configuration/transcript validation and audio rehashes; no transcripts changed or audio approvals were granted. Triage: 285 no-obvious-text-issue, 193 needs-audio-review, 13 unintelligible-text. Tracked audit: `curation/FIL_KIMI_500_COMPLETE.json`. Rejected responses remain unattached and were not retried.

After rechecking the frozen input and all exclusion hashes, `fil-kimi-review-next-1000-v1` was launched using the same requested `us.kimi-k3` route, one attempt per row and two-consecutive-failure stop. Its tracked selection now records launch status.

### First Bikolano backup-and-eviction batch verified complete

The bounded preemptive job completed after backing up and checkpointing twenty Bikolano originals. It unlinked 839,147,161 bytes; the observed free-space increase during eviction was 804,147,200 bytes because other work was concurrent. Independent audit confirmed all twenty local paths are absent, SQLite now has twenty `remote` plus 249 `downloaded` rows, and twenty backup entries remain. No recording metadata was discarded.

Recovery checkpoint: `checkpoints/bikolano-rss/3fdfaa2b03acef120203f08c2fa3a9b96908f9c331c0dd9060bc444a1732db6c.sqlite` at private Chika revision `aaa52bab7b224da5d6cab4398893552832da836d`. Tracked evidence: `curation/BIKOL_BACKUP_EVICTION_CHECKPOINT.json`; detailed eviction journal/receipts remain in the source and job output roots.

The existing global supervisor subsequently entered Taglish checkpointing. It retains the supervisor lock during uploads, so do not run competing backup/eviction work against the same roots or restart the supervisor solely for latency. Active downloader roots may be backed up but are not evicted until idle. Storage floors remain unchanged; this 0.78 GiB nominal offload does not establish sufficient space for new archive work.

### Retained totals after offloading

RSS status now distinguishes `local_raw_hours`, `backed_up_remote_raw_hours`, `unbacked_remote_raw_hours`, and `retained_raw_hours` (local plus backed remote). Remote rows count as backed only when their audio hash has a backup record with nonempty repo/path/revision/verification timestamp and positive bytes. `EXISTS` avoids multiplying item hours from duplicate backup entries. Legacy databases without a backup table classify remote rows as unbacked. The existing `measured_downloaded_hours` field remains local-only for compatibility.

These are ledger-derived raw runtimes, not fresh remote availability checks or unique speech totals. Offloading a verified original therefore does not erase retained acquisition progress. Kimi drafted the helper/tests; local integration and all three status tests passed, including incomplete receipts, duplicate backup records and missing-table behavior.

### Refreshed canonical archive snapshots

Completed acquisition records were re-imported without copying media: Hiligaynon 152 recordings / 128.966757593 raw hours across eight sources; Bikolano DZGB 96 recordings / 76.644920871 raw hours. The importer verified each audio hash, file size, audio stream, duration and unchanged metadata/file identity during the check. Both snapshots have zero exact-byte duplicates internally, empty transcripts, pending ASR and provisional language/rights. These snapshots overlap prior imports and may overlap RSS or other recordings acoustically; do not sum them as unique speech hours.

`curation/ARCHIVE_CANONICAL_REFRESH_CHECKPOINT.json` records immutable snapshot/manifest hashes and sizes. Hiligaynon excluded four incomplete acquisition entries. Bikolano excluded five metadata entries: three incomplete acquisition records plus worker/smoke-inventory metadata, not five lost recordings. The importer expects archive records one directory below `--archive-root`: use `outputs/bikolano`, not its `dzgb` child. An initial child-root scan produced an empty snapshot; it is not the authoritative Bikolano result. The nonempty checkpoint above supersedes it.

The finite Hiligaynon sources remain in scope: Deditos11, IlonggaTalks1, RMNBacolod8 and known Kuan2 were represented; larger GMA/Aksyon/Bombo inventories remain partially acquired with the original archive process still live. No source was declared exhaustively complete from runtime totals alone.

### Canonical archive metadata checkpointed remotely

The refreshed Hiligaynon and Bikolano manifests/reports (four files, 319,666 bytes) are saved under `checkpoints/canonical-archives/{snapshot}/` in private `sapinsapin/chika`, revision `dc6fcae14ce8d6dfff0739f589538f1f422cd18c`. All four were freshly downloaded at that immutable revision and compared byte-for-byte. `curation/ARCHIVE_CANONICAL_REMOTE_CHECKPOINT.json` records exact remote paths and hashes.

This is **metadata-only recovery evidence**. It uploads no audio and grants no eviction eligibility for referenced originals. Full media backup proofs/checkpoints remain separately required.

### Offloaded Bikol audio restored and independently checked

An isolated recovery test freshly downloaded the immutable Bikolano SQLite checkpoint and one backed original, with a one-file/64 MiB cap and 25 GiB floor. It recovered 33,977,796 bytes / 2,100.870998 seconds. Independent SHA256, size, ffprobe audio/duration and SQLite integrity checks passed. The original source ledger still marks the item remote and was not altered.

Tracked proof: `curation/BIKOL_RESTORE_PROOF_CHECKPOINT.json`; isolated restored root: `outputs/bikolano-restore-proof-v1/`. The isolated ledger correctly marks one recording local, nineteen backed-but-not-restored recordings remote and 249 unbacked recordings pending. This is a recovery test, not new acquisition or unique audio hours; do not add the test root to production accounting.


### Consolidated Filipino staging checkpoint (2026-11-15)

Combined the reviewed 5,671-row snapshot with persisted ASR v5/v7/v8/v9 responses: 7,946 IDs, 2,504 text-reviewed and 5,442 draft, including 27 empty transcripts. Rehashed every local audio file; all 7,946 matched. No audio was copied or new ASR requested. The manifest is staged only; see `curation/FIL_CONSOLIDATED_MANIFEST_CHECKPOINT.json`.

Exact audio hashing finds 7,460 distinct audio hashes across those IDs. Hold the consolidated export until duplicate groups and transcript differences are reconciled; do not count the 34,841.484375 padded seconds as unique verified speech. Existing snapshots remain unchanged.


### Exact-audio deduplication (2026-11-15)

Run `python -m curation.deduplicate --manifest INPUT --output NEW_DIRECTORY` to derive a unique-byte staging view. The command validates the schema, rehashes every source audio file, rejects conflicting text, split, rights, language, duration, speaker, synthetic status or speech type, and selects by existing review status then ID. It preserves selected rows verbatim and writes every excluded row in `aliases.jsonl`; no media is deleted. Review status selection grants no new approval. Keep the alias file with the canonical manifest for provenance and future split assignment.

Applied to the 7,946-row Filipino snapshot: 7,460 canonical rows and 486 alias rows. Independent reconstruction proved the outputs preserve every original row exactly; all canonical audio hashes are distinct. This resolves the exact-byte export hold in the preceding checkpoint, but acoustic deduplication, audio review, permissions and final export remain pending. See `curation/FIL_DEDUPLICATED_CHECKPOINT.json`. Three focused tests cover conflict rejection, preservation and distinct audio with identical text. NYO requested `us.kimi-k3`, returned `global.moonshotai.kimi-k3.1`, supplied advisory rules; local code and data checks enforce them.

The deduplicated manifest, full alias rows and audit are backed up as deterministic gzip files in private `sapinsapin/chika` at revision `d8afa809fbff479d90d904b591fcf6bd99e4e210`. All three files (6,450,730 compressed bytes) were freshly downloaded at that revision, hash checked and decompressed to exact original bytes. `curation/FIL_DEDUPLICATED_REMOTE_CHECKPOINT.json` records paths and both hashes. Restore by downloading these pinned paths and decompressing; audio paths still require the corresponding media backups. This metadata-only backup grants no audio eviction eligibility.


### Next storage recovery pass queued (2026-11-15)

Bikol RSS was independently observed idle, with no remaining downloaded rows having Hub backup proof. A second preemptive pass is queued behind the existing supervisor lock (`outputs/bikolano-preemptive-backup-v2`, PID 78938), bounded to 100 backups and 8 GiB of verified eviction. It uses the existing remote checkpoint and eviction verifier, preserves active downloaders, and stops on errors. The launch checkpoint is `curation/BIKOL_BACKUP_V2_CHECKPOINT.json`; launch is not evidence of successful backup or reclaimed space. Inspect the actual process, worker/step receipts and ledger before claiming recovery.


### Prevent duplicate review spending

`curation.select_review` now excludes previously assigned content under alternate IDs using exact audio hash, exact transcript, primary language and language list, and selects only one matching input within a batch. Existing ID exclusions still apply, including failed assignments. Changed transcript/language inputs remain distinct. No reviews are transferred to another ID. The receipt reports `content_aliases_excluded`; callers must still enumerate every prior/running/prepared assignment and serialize selection. Six selector tests pass. `curation/FIL_REVIEW_ALIAS_AUDIT.json` records a read-only audit against enumerated Filipino review assignment inputs.


### Kimi 1,000-row automatic completion

A locked finalizer watches the existing review worker in `outputs/fil-kimi-review-next-1000-v1`. On successful terminal completion it validates the frozen input hash and full processed count, revalidates each accepted review against its raw response/config/text/audio bytes using `attach_reviews`, and writes an exclusive `reviewed-manifest.jsonl` plus `join-audit.json`. Rejected reviews remain unattached. It makes zero model requests, changes no transcripts and grants no audio approval. Missing/failed workers or validation failures stop it for inspection without replay. The tracked `curation/FIL_KIMI_1000_FINALIZER_CHECKPOINT.json` includes its exact script for recovery; inspect the actual process and finalizer lock before restoring or running it.

### Archive backup coverage and eviction gap (2026-11-15)

`curation/HIL_ARCHIVE_BACKUP_COVERAGE.json` reconciles local backup receipts against the fixed 152-row Hiligaynon snapshot. At observation, 43 audio hashes (2,788,509,587 bytes) had matching-size receipts and 109 (5,228,252,983 bytes) did not. Repeated receipts are counted once per audio hash. This is a local receipt inventory during a live backup, not fresh remote verification; newer acquisitions are outside this denominator.

Archive receipts retain `local_deletion_authorized=false`. The archive pipeline still needs a durable remote/evicted lifecycle, acquisition logic that does not redownload evicted media, cooperative reader/writer locks, preserved acquisition provenance and a verified restore path before archive originals can be removed. RSS eviction cannot be applied unchanged to these files. NYO requested `us.kimi-k3` and returned `global.moonshotai.kimi-k3.1` for a review of these lifecycle gaps. No archive media was deleted; existing RSS backup/eviction remains the current space-recovery path.


### Archive snapshots after storage-floor stop (2026-11-15)

Refreshed canonical imports after both archive workers exited: Hiligaynon 169 completed recordings / 146.854601204 raw hours across eight sources; Bikol DZGB 110 recordings / 88.735163649 raw hours. The importer checked hashes, byte sizes, audio streams, durations and acquisition metadata; no within-snapshot exact byte duplicates were found. Incomplete/other metadata entries were excluded. `curation/ARCHIVE_FLOOR_SNAPSHOT_CHECKPOINT.json` pins the manifests and reports. These supersede the earlier 152/96 inventory for future planning, overlap older snapshots and remain untranscribed, provisionally language-labeled and unapproved for training. No media copies or new ASR requests were made.

The Hiligaynon169 and Bikol110 frozen manifests and reports are now recoverable from private `sapinsapin/chika` revision `40019c3d6aa82bb0a566e988dc74be85fdbe4763`. All four files were freshly downloaded at that revision and matched both SHA-256 and exact local bytes. Paths and hashes are in `curation/ARCHIVE_FLOOR_REMOTE_CHECKPOINT.json`. This is a metadata-only checkpoint; it does not upload the corresponding recordings or authorize eviction.

### Partial-checkpoint RSS recovery (2026-11-15)

The RSS evictor now accepts repeated `--only-sha256 HASH` arguments (or `only_sha256` in Python). This restricts candidates to an explicit set while retaining download-lock, private-repository, immutable-checkpoint, remote hash/size, local fingerprint and journal checks. An empty API set selects nothing. It does not skip failed proofs for selected files or infer permission from a hash. This allows an earlier verified checkpoint to offload its covered files while a live backup adds newer receipts.

The Taglish intermediate checkpoint initially failed the unrestricted dry run because the live ledger gained an eighth backup. Filtering to its seven proven hashes then passed; application offloaded seven originals totaling 602,607,050 bytes. Independent checks confirmed all seven local paths absent and their ledger entries `remote` with no local path. Restore coordinates remain in `curation/TAGLISH_PROGRESS_CHECKPOINT.json`. The ongoing backup was not restarted. All 33 eviction/restore tests passed, including new subset and proof-rejection cases. ASR stays stopped until the 30 GiB free-space requirement is satisfied.

### Sequential Kimi continuation (2026-11-15)

Prepared 1,000 additional draft clips from the 7,460-row exact-audio-unique manifest, excluding all seven enumerated prior/current assignment inputs plus all already text-reviewed canonical rows. Selection excluded 3,140 assigned rows and 27 empty/ineligible rows, leaving 4,293 candidates before the limit. Independent checks found no matching content keys in prior assignments and no duplicate content within the new batch.

`outputs/fil-kimi-review-next-1000-v2` is queued behind v1. The locked runner checks the actual prior process while waiting, requires its complete 1,000-row summary without early stopping, checks the frozen input hash, then uses `us.kimi-k3` with one attempt per row and a two-consecutive-failure stop. It runs the content-bound finalizer after completion. No failed reviews are retried and no transcript edits/audio approvals are granted. `curation/FIL_KIMI_NEXT_1000_V2_CHECKPOINT.json` preserves source/exclusion hashes and exact runner/finalizer code for recovery. Inspect worker state and lock before recovery; do not blindly relaunch a started job.

### Kimi rejection audit (2026-11-15)

Re-parsed the eleven recorded failures in the running 1,000-row batch offline against each original input/config: three proposed lexical changes requiring listening, three had invalid request identity, one completion did not finish normally, one contradicted its no-obvious-issue triage, two had invalid issue structure, and one had no saved response. `curation/FIL_KIMI_1000_REJECTION_AUDIT.json` records response hashes and validation categories. This is an observation during a running batch, not its terminal failure total. No requests, retries, prompt changes or transcript edits were made. The strict contract remains in place; text review must not invent words or reject valid code-switching by normalizing it into another language.

### Kimi 1,000 v1 completed; terminal-status wrapper corrected

The batch processed all 1,000 inputs without early stopping: 988 accepted, 12 rejected. `curation.review` returned exit code 1 because individual rows failed; the conservative wrapper labeled this `stopped_with_failures`, causing both the finalizer and queued continuation to stop. Inspection proved full coverage from the terminal summary and absent worker processes. No requests were replayed.

Finalization now accepts either successful or failed-row terminal status only after verifying full processed coverage and no early stop. It attached all 988 accepted reviews after content-bound validation; independent checks confirmed all 1,000 IDs, transcripts and audio hashes unchanged. `curation/FIL_KIMI_1000_COMPLETE.json` pins the reviewed manifest. Rejected rows remain drafts; no audio approvals were granted.

The next batch had no reviews directory and had never issued requests. After preserving its stopped state, the corrected wrapper was launched for those 1,000 new inputs. Recovery scripts/checkpoints were updated. Future workers must distinguish per-row rejection from interrupted/early-stopped batches; exit code alone is insufficient to decide whether an entire batch was processed.

### Deduplicated Filipino review merge (2026-11-15)

Merged 984 of the completed batch's 988 accepted reviews by exact existing ID and matching transcript/audio/source/interval/split/rights/language fields into a new 7,460-row staging manifest. Every incoming audio file was rehashed. The result contains 3,067 text-reviewed rows and 4,393 drafts, with one row per exact audio hash and no transcript changes. Four reviews belong to alternate IDs; their matching audio/transcript canonical IDs are recorded separately and reviews were not silently transferred. `curation/FIL_7460_V2_CHECKPOINT.json` pins inputs/output and the alias mapping. This snapshot is staged only, with zero audio approvals; no new Parquet export or publication is claimed.

### Hiligaynon 49-recording archive backup complete (2026-11-15)

The archive backup worker finished all 49 frozen recordings: 3,342,182,733 bytes and 40.727967871 raw hours. An independent pass verified both remote object entries at each receipt's immutable revision, checked audio size/SHA-256 against the canonical manifest, and freshly downloaded all 49 metadata files to verify hashes and recording/audio identities. `curation/HIL_ARCHIVE_49_BACKUP_COMPLETE.json` records all pinned recovery objects. The audio proof uses remote LFS hash/size (downloaded hash fallback for non-LFS objects), not a claim that every audio file was downloaded again. No local archive files were deleted. This older 49-recording snapshot overlaps the 169-recording inventory and must not be added as independent hours.

The updated 7,460-row manifest, its merge audit and the completed 1,000-row review manifest (including the four alternate-ID reviews) are remotely checkpointed at private `sapinsapin/chika` revision `6c5d5a5af848effc0b3d0d667925613354302494`. All three deterministic gzip files were freshly downloaded, hash-checked and decompressed to exact original bytes. `curation/FIL_7460_V2_REMOTE_CHECKPOINT.json` records recovery paths and hashes. This metadata backup does not publish a new dataset configuration or upload audio.

### Remote acquisition on dwnd-01 (2026-11-15)

Authenticated as `iotim` using the existing local `~/.ssh/nyo-iotim` identity. The host has about 1.8 TB free. Installed ffmpeg/ffprobe and an isolated uv environment with yt-dlp under `/home/iotim/chika-curation`; copied only the acquisition modules and three preflighted finite inventories. No credentials, local audio or ASR serving configuration were copied/changed.

A one-item Cebuano smoke download succeeded (`AGzm7fuNMiA`, 1,680.321 seconds). Three independent persistent workers now acquire the preflighted Cebuano, Bikol/Bicolandia and Ilocano inventories in separate directories. Each keeps a 100 GiB floor, 10-hour batch cap and monitored 1 GiB per-item transfer budget. `curation/DWND_ACQUISITION_CHECKPOINT.json` records remote paths, PIDs, commands, inventory hashes and actual process observations. Remote acquisition rows and audio remain on that host; local status totals do not yet include these jobs. Inspect processes and per-item records before any resume. Launch markers prevent duplicate launches. These are provisionally labeled raw recordings, not approved training data or transcribed hours.

### Additional offload to dwnd-01 (2026-11-15)

Expanded remote work with three Hiligaynon continuation workers (Ratsada/One Western Visayas, Aksyon streams, Aksyon videos), excluding every previously attempted local ID. Local failed/excluded records are retained for separate inspection, not silently replayed. The mixed-program GMA inventory is restricted to recognizable Ratsada/One Western Visayas title matches; 6,839 other titles remain in a deferred review inventory. Each remote continuation is bounded to 100 raw hours / 10,000 items, a 100 GiB disk floor and monitored 1 GiB per-item budget. Larger finite backlogs remain saved; these caps do not imply all sources are finished.

Remote Cebuano import validated 18 completed recordings / 9.187271667 raw hours with no exact byte duplicates. A separate CPU preparation worker decodes those recordings and emits 27-second transport windows entirely on dwnd-01, with a 100 GiB floor. It makes no ASR calls and grants no language, speech or training approvals. Remote views keep each language's import separate. `curation/DWND_OFFLOAD_CHECKPOINT.json` records launch commands, paths, process observations and selection hashes. Existing local backup/review workers and production ASR serving were not changed.

### Verified remote acquisition and preparation checkpoint (2026-11-15)

`curation/DWND_VERIFIED_PROGRESS_CHECKPOINT.json` records a fresh remote observation, per-recording paths, SHA-256 hashes, sizes and durations. All recorded downloaded files were independently rehashed on dwnd-01. Completed regional batches contain 19 Cebuano recordings (9.706 raw hours), 17 Bikolano recordings (9.730 raw hours), and 19 Ilocano recordings (9.773 raw hours). These counts are remote-only; do not add them to earlier snapshots without identity reconciliation.

The 18-recording Cebuano preparation snapshot completed successfully: 1,233 fixed transport windows, 9.187175017 window hours, zero transcribed hours. Every window audio hash was checked against the manifest; its SHA-256 is `01709dcb12f43ce258454ea3ccb3a194bcf58dc8aac812d086d22d756b14c018`. The nineteenth acquired recording is outside this preparation snapshot. No speech, language, rights or training approval is implied.

At observation, Ratsada and Aksyon streams processes were confirmed live, with 34 and 15 downloaded recordings respectively. Aksyon videos stopped with 17 downloaded and two failed items; failures remain pending inspection, not silently retried. This is a moving acquisition snapshot, not completion of the saved Hiligaynon backlog. Remote free storage was approximately 1,832 GiB. `reth5et` SSH port 22 timed out before authentication; no work was moved there.

### Remote Bikolano and Ilocano preparation (2026-11-15)

Verified canonical imports now cover all 17 remote Bikolano recordings (9.729593611 raw hours) and 19 Ilocano recordings (9.772842486 raw hours). Independent persistent preparation workers run on dwnd-01, using 27-second transport windows and a 100 GiB free-space floor. `curation/DWND_BIK_ILO_PREPARATION_CHECKPOINT.json` preserves exact commands, input hashes, worker identities, live process observations and recovery scripts. These processes do not call ASR. Keep the provisional `bik` label; do not silently map it to a Central Bikol (`bcl`) model. Confirm terminal worker receipts and rehash generated audio before claiming preparation completion. Do not relaunch solely because a monitoring call times out.

### Further regional source preflight on dwnd-01 (2026-11-15)

A persistent metadata-only worker checks 50 additional candidates per language (Cebuano, Bikolano and Ilocano). Selection excludes IDs in the other tracked language inventories; source discovery inventories remain intact. Publisher channel and video ID must match, live status must be finite, and duration must be 5–120 minutes. Failed candidates are retained with reasons. `curation/DWND_REGIONAL_PREFLIGHT_V2_CHECKPOINT.json` preserves inputs, exact recovery code and a live process observation. Results remain candidates until terminal receipts and overlap checks are inspected; this worker does not download media or approve language/rights.

### Remote Cebuano VAD (2026-11-15)

Installed a separate Python 3.11 compute environment on dwnd-01 with ONNX Runtime, NumPy and SoundFile. The existing hash-pinned Silero model passed a four-window smoke run (15 candidate segments). A persistent CPU worker now processes all 1,233 prepared Cebuano windows, bounded to 600 minutes and 1,233 rows. The smoke output overlaps the full run and must not be added to its totals. `curation/DWND_CEB_VAD_CHECKPOINT.json` records commands, dependencies, recovery code and live process evidence. Outputs remain untranscribed speech candidates, with no language or training approval. The corrected alternate host `carrot@reth5et.tail36638e.ts.net` also timed out on SSH port 22 before authentication.

### Verified Bikolano/Ilocano preparation and VAD launch (2026-11-15)

Both remote preparation workers completed with return code zero. Revalidated every generated window against its audio SHA-256 and the canonical schema: Bikolano 1,308 windows / 9.729509323 hours; Ilocano 1,312 windows / 9.772801701 hours. Full-batch CPU Silero jobs are now running in separate output roots on dwnd-01, with the same pinned model and 600-minute bounds as Cebuano. `curation/DWND_REGIONAL_VAD_CHECKPOINT.json` records verified input manifest hashes, terminal preparation summaries, launch commands, recovery code and actual process observations. VAD output remains pending verification, and all these batches have zero transcribed/approved hours.

### Second remote regional acquisition batches (2026-11-15)

The 150-candidate metadata preflight finished: 47 Cebuano, 49 Bikolano and 50 Ilocano candidates passed publisher/video identity, finite-live-status and duration checks. All saved metadata hashes were rechecked, and no accepted IDs overlap the first remote batches. Selection had already excluded IDs in other tracked language inventories; acoustic duplication remains unresolved. Three persistent download workers now run on dwnd-01 in `*-next-v2` directories, with 35-hour batch caps, one attempt per item, a 100 GiB free-space floor and a monitored 1 GiB per-item transfer budget. Metadata duration estimates are 23.64, 27.84 and 23.94 hours respectively; these are not downloaded hours. `curation/DWND_REGIONAL_DOWNLOAD_V2_CHECKPOINT.json` preserves the accepted inventories, rejection reasons, metadata hashes, commands, live process observations and recovery code. Inspect per-item receipts after a worker exits; exit zero alone is not proof every item downloaded.

### Regional VAD outputs independently audited (2026-11-15)

All three full remote VAD runs completed successfully: Cebuano 3,663 segments / 7.579527778 candidate hours; Bikolano 4,185 / 8.269753333; Ilocano 3,464 / 8.541022778. `curation.audit_vad` checked every derivative against an independently supplied preparation manifest: schema, unique IDs, audio hash/size/format, exact decoded PCM slice equality, sample offsets, segment bounds/non-overlap and inherited metadata. `curation/DWND_REGIONAL_VAD_COMPLETE.json` pins all manifests and reports; audit reports also remain beside the remote outputs. These 11,312 segments are untranscribed speech candidates, not verified language or training-ready hours. The smoke run is overlapping and excluded. Production ASR still lacks confirmed Cebuano/Bikolano/Ilocano routes, so no silent substitution with Filipino or Hiligaynon ASR was made.

### Filipino/Taglish RSS continuation on dwnd-01 (2026-11-15)

Delegated 132 never-attempted pending RSS episodes across nine sources to dwnd-01, approximately 111.21 advertised hours. Under the local download lock and a SQLite transaction, these local rows were marked `delegated`; the normal local collector only selects pending/failed rows, preventing concurrent redownload. A pre-delegation SQLite backup remains at `curation/outputs/taglish-dwnd-v1/before-delegation.sqlite`. The remote queue contains only selected pending rows and source metadata, with no acquired audio or remote-backup claims. Initial queue hashes matched across hosts.

The persistent remote collector runs with 132-item/125-measured-hour limits, one attempt per item and a 100 GiB storage floor. `curation/DWND_TAGLISH_DOWNLOAD_CHECKPOINT.json` records every delegated original row, hashes, command, worker identity and recovery code. Remote successes must be reconciled by ID/hash and location before changing local ownership; never reset delegated rows to pending while the remote worker may own them. Recover individual failures only after inspecting terminal receipts. These new hours remain provisional raw audio, not transcribed or verified Taglish.

### Filipino processing follows remote acquisition (2026-11-15)

Started remote preparation from a consistent SQLite backup of the Filipino RSS queue. The frozen downloaded rows and original hashes/paths are recorded in `curation/DWND_FIL_PREPARATION_CHECKPOINT.json`; new downloads continue in the original ledger independently. Preparation runs entirely on dwnd-01 with 27-second windows and a 100 GiB floor. Later snapshots must exclude these selected IDs or reuse verified cached preparation to avoid duplicate processing/counting. No ASR credentials were copied, no transcription was started, and the frozen queue is an input snapshot rather than a second downloader.

### Regional v2 launch failure and verified recovery (2026-11-15)

All 146 v2 regional items initially failed with `transfer_monitor_error`: noninteractive SSH PATH omitted the yt-dlp executable installed beside the virtualenv Python. No media was downloaded by those first attempts. `acquire_archives.executable` now also searches the running interpreter directory. A focused fallback check and both inventory tests passed. A real retry of Cebuano `ZLQNhI-I14Q` succeeded (2,108.321 seconds), confirming the fix beyond unit checks. After verifying the old wrappers were terminal and absent, three recovery wrappers were launched in the same roots with max-attempts increased from one to two; the successful smoke item is cache-skipped. Original failure records and old worker receipts remain available. `curation/DWND_REGIONAL_V2_RECOVERY.json` records new PIDs, exact commands and smoke evidence. Use recovery-launch.json for current worker identity, rather than the original launch.json. Filipino RSS acquisition was unaffected.

### Filipino remote preparation verified; VAD and audit started (2026-11-15)

The eight-recording Filipino preparation snapshot completed with 1,017 windows / 7.600932049 decoded window hours. Every window passed schema validation and its stored audio SHA-256 check. A persistent dwnd-01 CPU job now runs pinned Silero VAD across the full snapshot, then automatically runs the independent exact-PCM/offset/inherited-metadata audit. An audit failure has a distinct terminal status and cannot be reported as completion. `curation/DWND_FIL_VAD_CHECKPOINT.json` records input hashes, command, required PYTHONPATH, recovery code and live process evidence. No ASR request has been made for these new remote recordings.

### Remote Filipino ASR client ready (2026-11-15)

Installed the existing bounded persistent ASR client on dwnd-01, matching local websockets 16.1.1. Provisioned the existing authorized Boses bulk credential over SSH to an owner-only file under `.credentials` (directory 0700, file 0600); no secret is committed or printed. `curation/DWND_ASR_CLIENT_READY.json` records module hashes, endpoint/model, permissions and the live VAD dependency. No remote ASR request has been sent. Require terminal successful VAD and its exact-PCM audit before the two-item smoke; use `fw/whisper-large-v3-pld-fil-norm`, language `fil`, and preserve the uncertain-request stop/no-replay contract. Production serving remains unchanged.

### Filipino remote transcription started (2026-11-15)

Remote VAD and independent PCM audit completed for all 5,400 segments / 7.001574444 candidate hours. A two-item Boses smoke succeeded with two nonempty draft transcripts and no failures. A persistent worker now transcribes the full 5,400-row input using the existing Filipino model, 100 GiB floor and checkpoint every 25 rows. It deliberately reuses `remote-fil-asr-smoke-v1` as the response cache so the smoke requests are not duplicated; worker lifecycle files live in `remote-fil-asr-batch-v1`. `curation/DWND_FIL_ASR_BATCH_CHECKPOINT.json` pins the audit, smoke result, commands and recovery code. Any uncertain request stops the batch for inspection; a running worker is not a claim of full transcription completion. Text review and audio verification remain separate.

### Second remote Filipino preparation snapshot (2026-11-15)

A second consistent RSS snapshot selects 49 additional downloaded recordings / 29.865554864 raw hours, explicitly excluding all eight first-batch IDs. Preparation runs independently on dwnd-01 with the same 27-second windows and 100 GiB floor. `curation/DWND_FIL_PREPARATION_V2_CHECKPOINT.json` records every selected/excluded ID, input snapshot hash, command and process observation. The original acquisition ledger is unchanged. The current remote-only acquisition/ASR snapshot is `curation/DWND_STATUS.json`; pending recovery items may still carry their historical failed status until retried, so those counts are not terminal new failures. Do not add prepared/VAD/ASR durations to acquisition totals.

### Remote Hiligaynon canonical import and preparation (2026-11-15)

Imported a frozen snapshot of 175 completed remote Hiligaynon recordings from three source roots, totaling 141.653670833 raw hours. Import verified file hashes, sizes, probe durations and canonical schema; no exact byte duplicates were found within this snapshot. Twelve other entries were excluded by the importer. These are provisional language labels and do not establish unique speech or source rights.

Preparation now runs for the entire snapshot on dwnd-01, using 27-second windows and a 100 GiB floor. Acquisition continues independently; later snapshots must reconcile IDs against this frozen batch. `curation/DWND_HIL_PREPARATION_CHECKPOINT.json` preserves the manifest hash, import report, included IDs, command, live process observation and recovery code. No additional Hiligaynon ASR or TTS training is implied by this preparation launch.

### Reusable remote status and first Filipino ASR stop (2026-11-15)

Run `python -m curation.remote_status --root curation/outputs` on an acquisition host to inspect worker identities, recovery launches, archive receipts, read-only RSS ledgers and preparation/VAD/ASR summaries. It emits counts without transcript text or process command strings, reports malformed-file observations, and distinguishes process presence from matching the job root. Run over SSH and save stdout to `curation/DWND_STATUS.json`; no job is restarted or media rehashed. Counts across pipeline stages overlap.

The first full remote Filipino ASR run stopped after 160 completed rows (one empty), approximately 0.417786111 nonempty draft-transcribed hours. The failed row is `c6506e60e8131953927fe7598cc9035ff979c740e9b8aef414b947aa49c83f3b`; its attempt cache key is `760b0a12eb5b32c77f5f54ed4e5378ebb49cfefb43c04245b94e6a5cef806bec`. The saved attempt reports ValueError inside the session, surfaced as RuntimeError by the batch. This is an uncertain sent request, not permission to retry it. All 160 partial results and response receipts remain on dwnd-01. Downloads and preparation continue independently; ASR diagnosis is lower priority per the acquisition-first instruction.

### Partial remote Filipino transcripts backed up (2026-11-15)

The 160-row draft ASR manifest and terminal stop summary are backed up as deterministic gzip objects in private `sapinsapin/chika` at revision `fd5ddb7001090975252780158bb9697f2e422779`. Both objects were freshly downloaded from that immutable revision, checked by compressed SHA-256 and decompressed to exactly the local source bytes. `curation/DWND_FIL_ASR_PARTIAL_REMOTE_CHECKPOINT.json` pins recovery paths. This is metadata-only preservation: no new audio upload, dataset configuration, text review or audio approval. The uncertain request remains excluded from automatic retries.

### Bounded Filipino ASR continuation after timeout inspection

The first remote stop was a service `finalize_timeout`. A new 100-row batch contains only never-attempted audio, excludes the timed-out ID/hash, and preserves the 160 validated prior results separately. Its single-row canary passed before launch. See the continuation selection/launch checkpoints and `docs/asr_incidents.md`; do not combine stages or replay quarantine entries. NYO us.kimi-k3 supplied the recovery review.

### Remote Filipino continuation verified and second VAD batch launched (2026-11-15)

The bounded 100-row ASR continuation completed without failures. All row IDs match its frozen input; saved raw response protocol, resulting transcript equality and audio hashes were independently checked. `curation/DWND_FIL_ASR_100_COMPLETE.json` pins the result. It is disjoint from the earlier 160-row batch by attempt fingerprint/audio selection; the timed-out segment remains quarantined.

The second Filipino preparation snapshot completed with 4,007 windows / 29.866669514 hours. Every input audio hash was checked before starting full VAD followed automatically by exact-PCM auditing. `curation/DWND_FIL_VAD_V2_CHECKPOINT.json` records launch/recovery details. The latest remote status also shows regional v2 downloads terminal: Cebuano 47 recordings /23.639785278 raw hours, Bikolano 48 /27.314391111 with one failed item, and Ilocano 50 /23.939346569. These last acquisition counts are receipt observations, not a fresh full-file rehash.

### Regional second batches: verified imports and staged processing

All 145 completed v2 regional downloads were rehashed, size/duration-probed and imported into frozen canonical manifests: Cebuano 47 /23.639785278 raw hours, Bikolano 48 /27.314391111, Ilocano 50 /23.939346569. Each snapshot has zero internal exact-byte duplicates. Noncompleted/metadata-only entries remain excluded, including the one failed Bikolano item.

Three persistent workers now chain preparation, generated-window hash validation, VAD and independent exact-PCM/offset/metadata auditing. Stage status is written atomically; an exception stops that language worker. Preparation and the VAD transition enforce the 100 GiB floor. No ASR calls or approval changes are included. `curation/DWND_REGIONAL_PROCESSING_V2_CHECKPOINT.json` preserves frozen manifest hashes, import reports, commands, complete recovery code and launch observations. Inspect worker.json plus terminal audit before claiming completion; do not restart on observation timeouts.

### Filipino remote acquisition complete and remaining preparation launched

All 132 delegated Filipino RSS episodes completed: 111.861506029 raw hours. Independently rechecked every downloaded file size and SHA-256 against the remote ledger. `curation/DWND_FIL_DOWNLOAD_COMPLETE_PREP_V3.json` preserves all recording identities, hashes and remote paths. The local ledger remains delegated rather than pretending these files exist locally or are backed up to Hugging Face.

A third frozen preparation snapshot selects the remaining 75 recordings / 74.395227855 raw hours, excluding all 57 IDs from the prior two batches. The worker is live on dwnd-01 with 27-second windows and a 100 GiB floor. All 132 recordings are now assigned exactly once across preparation snapshots; this does not mean all preparation or transcription is complete. No local files were deleted.
