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

`curation/outputs/fil-persistent-batch-v2/` was seeded with the fourteen validated response/result caches. Its worker selects the first 100 original Filipino VAD rows, so it can make at most 86 new requests; cached rows make none. It uses zero retries, a 60-second client bound, and stops on the first uncertainty/error. Check actual process state, `worker.json`, and `asr/summary.json` before recovery. It was still running at this checkpoint, with 27 completed rows and no failures observed. All outputs remain drafts.

Acquisition, archive backup, the Vigan capture and Hiligaynon text review continue independently. No Hiligaynon server-timeout batch was restarted, and no new Cebuano/Bikolano/Ilocano route is claimed.
