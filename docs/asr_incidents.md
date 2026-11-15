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


## Hiligaynon persistent route: new-source pilot and bounded continuation (15 November)

A ten-segment pilot on previously unattempted original recordings completed successfully through `fw/whisper-large-v3-pld-hil-norm`: all ten nonempty transcripts, zero errors, 82.448 padded seconds. The input deliberately excludes the two original recording hashes used by the earlier timeout-affected serial batch. This is evidence of current service on these requests, not proof the prior timeout defect is fixed. Failed earlier items were not retried.

The follow-on `curation/outputs/hil-persistent-new-batch-v1/` freezes 1,380 VAD candidates (2.331931111 padded hours), including the ten successful pilot segments. All ten caches were validated with network exchange forbidden before launch; the maximum new-request count is 1,370. Input SHA256: `1ecbeb6ec2444b24a3282906ab48082620dda2541354acc08c336e5dd33da733`. Inspect `selection.json`, `worker.json`, actual process, and `asr/summary.json` before recovery. One persistent session, no request retries, stop on first error, combined-manifest checkpoints every 25 completions. Per-item evidence is durable between checkpoints.

The pilot root is `curation/outputs/hil-persistent-new-10-v1/`; its ten rows overlap the larger batch and must not be summed twice. These remain source-labeled machine drafts, not audio-verified transcripts.


### Follow-on Hiligaynon session stopped on recurring server deadline

The bounded continuation stopped after 51 completed, nonempty segments (0.082330556 padded hours), with one failed request. The server explicitly returned `finalize_timeout`: its engine did not finish finalization within 3,000 ms. The failed row ID is `e96017e8026cf02fac594c0c93eb64c9e3abd8d7757d4c2258d67a0c196e044c`; the raw attempt remains in `hil-persistent-new-batch-v1/asr/`. The process is terminal, `stopped_early=true`, and the combined manifest contains exactly the 51 successes. The initial ten-row pilot is included in those 51.

This confirms the earlier successful pilot did not establish that the server deadline issue was resolved. No retry was issued for the failed request and the remaining candidates stay unattempted. Acquisition remains the priority. The successful drafts alone were selected for a bounded text-review job under `curation/outputs/collected-hil-new-51-review-v1/`; it cannot approve audio or recover a missing ASR transcript. Filipino's separate active route and queued stages are unaffected.

### Filipino v5 connection closure (15 November)

The v5 persistent session terminated after 780 completed rows (four empty; 1.005215556 padded transcribed hours). Failed row: `bfb22fee67856b5e8bda0a86d2e9aea64352f07dae37e76e240f799f3ceda1a2`. Its saved attempt `f917f4a231f44309d3ec9eac7507e8590e59c992da2296c70f77d8973f378c2f.attempt-146e1c0edb5442b2a67ee8f176d5a5fe.json` records `ConnectionClosedError` and no received events. This does not prove the server did not process or charge the request. Do not replay it automatically.

The v2 queue stopped after this error and did not launch v6. Both v5 and queue terminal states were confirmed against missing process handles. Completed caches remain preserved; prioritize acquisition while ASR is degraded. This was not a storage-floor stop.

#### Bounded continuation without replay

After confirming v5 and queue v2 were terminal and their processes absent, all 781 prior attempt fingerprints were enumerated: 780 completed plus the uncertain connection-closed request. A new immutable 500-row v7 selection excludes every attempted fingerprint and completed/failed ID. It reads already-prepared audio, uses a separate fresh session and output directory, retains the 30 GiB per-request floor, and stops on any error. No failed request was retried. `curation/FIL_ASR_CONTINUATION_CHECKPOINT.json` records the frozen input and exclusions.

**Do not restart v5 or queue v2.** v7 now owns these 500 formerly unsubmitted rows; subsequent recovery must exclude its input/attempts as well. v6 remains unlaunched. This bounded continuation does not resolve or clear the uncertain v5 attempt, and source acquisition remains the priority.
