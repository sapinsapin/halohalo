# ASR failure diagnosis and recovery

## Saved-response investigation, 15 November 2026

The two bounded batches stopped for different reasons. Neither error should be inferred from the wrapper's generic `RuntimeError` alone; inspect the preserved per-attempt event ledger.

| Batch | Evidence | Action |
| --- | --- | --- |
| Hiligaynon serial batch | All nine failed attempts contain server `finalize_timeout` errors stating that finalization did not finish within 3,000 ms. | Remains deferred. A longer client timeout does not change the gateway deadline. No production serving configuration was changed. |
| Filipino persistent batch | Fourteenth attempt contains a `usage` event, complete final result and matching `finalize_end`; the client rejected `usage` as an unexpected event. | Client now preserves and accepts usage telemetry, without counting it as audio coverage or bypassing error/truncation checks. |

The local serving source's `serve/gateway.py` `ASRSession.send_usage` emits `usage` with cumulative `audio_ms` and credits. Accounting messages can occur before a result or between a result and finalization. Regression tests cover those placements, incomplete transcript coverage despite usage counters, and truncation flags. All existing model, handshake, finalize-ID and sample-clock checks remain active.

## Offline reconciliation

The original failed Filipino attempt under `curation/outputs/fil-persistent-batch-v1/asr/` remains unchanged. An explicitly reconciled copy at `curation/outputs/fil-reconciled-14-v1/` records the original attempt hash and why its complete response is acceptable. Replaying the first fourteen items with `websockets.connect` patched to raise completed fourteen drafts and made **zero network calls**. `reconciliation-audit.json` records that check. This was not a retry and does not imply audio/transcript accuracy.

Do not automatically change failed attempt statuses. Server errors, missing results, truncated coverage, mismatched IDs, unknown model binding and uncertain paid attempts must remain rejected. A telemetry-only parser defect is recoverable only when the complete saved exchange independently satisfies all other checks.

## Bounded continuation

`curation/outputs/fil-persistent-batch-v2/` was seeded with the fourteen validated response/result caches. Its worker selects the first 100 original Filipino VAD rows, so it can make at most 86 new requests; cached rows make none. It uses zero retries, a 60-second client bound, and stops on the first uncertainty/error. Check actual process state, `worker.json`, and `asr/summary.json` before recovery. It finished successfully with 100 completed rows, no failures and 0.0812433 padded transcribed hours. All outputs remain drafts.

Acquisition, archive backup, the Vigan capture and Hiligaynon text review continue independently. No Hiligaynon server-timeout batch was restarted, and no new Cebuano/Bikolano/Ilocano route is claimed.

## Full pilot continuation and review

After confirmed completion of v2, `curation/outputs/fil-persistent-batch-v3/` was seeded with its 100 response/result caches and launched with a 482-item ceiling, matching the entire original Filipino VAD pilot. It can send at most 382 new items, still with zero retries and stop-on-first-error behavior. It completed all 482 items with zero failures, no empty results and 0.5346333 padded transcribed hours. The source-interval accounting includes v3 and unions its overlaps with earlier snapshots.

`curation/outputs/fil-review-100-v1/` holds a frozen copy of the first 100 drafts and an independent text-review worker. It preserves code-switching and transcripts, uses one attempt per row, and stops after two consecutive failures. Hiligaynon's remaining-467 review continues separately. Neither worker provides audio verification.

`curation.status` now reports these persistent-ASR and text-review job states, selected/completed/failed counts, and actual process observations. `process_matches_job` also checks that the observed command mentions the expected job directory; a stored running marker or a recycled PID alone does not establish that the intended worker is running. Inspect logs and response evidence before any restart. Private commands, keys, transcripts and per-row review lists are not copied from worker/summary files into the status report.

## Larger audited Filipino set

`curation/outputs/fil-persistent-batch-v4/` now processes the audited 5,189-row Filipino/Taglish candidate manifest (6.43737 padded hours). All 482 pilot responses matched by audio hash and configuration and were copied as reusable caches; the ceiling is therefore 4,707 new requests. Input hash: `d1c0b1079e7ca889c973b137c941c29276d83571caf00ba3e40e0fc293e88b06`. The worker still stops on the first error and never retries uncertain attempts. This is a running job, not completed coverage.

`curation/outputs/fil-review-382-v1/` reviews the completed pilot rows outside the published first 100. Its frozen input hash is `853c67fa5818b9465999b5e4bca77be9c661e94ebb3a00bf0f70c637695901b5`. One attempt per item, two-consecutive-failure stop, no transcript edits or audio approvals. Existing published configurations stay immutable.

## Manifest checkpoint cadence

Future persistent-client launches default to `--checkpoint-every 25`. Every individual response/result still reaches its atomic cache before an item is counted complete; only the combined manifest is batched. It flushes at the cadence, on the first failure, and on normal or interrupted exit. Abrupt process termination can leave the aggregate manifest behind the response caches, which are revalidated and reused on recovery. `summary.json` distinguishes completed responses from `manifest_rows`; source-interval accounting reads the last available manifest, not an estimated future result.

This avoids rewriting an increasingly large manifest after every row. Tests cover final partial batches, failure flushes and replay with networking forbidden. The already running v4 process retains its earlier per-row behavior; do not restart it merely to load this optimization. No cache fingerprint or model configuration changed.
