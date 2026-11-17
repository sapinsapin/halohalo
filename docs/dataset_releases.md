# Chika draft dataset snapshots

These configurations are in the private [sapinsapin/chika dataset](https://huggingface.co/datasets/sapinsapin/chika). They follow the shared `sapinsapin.speech.v1` manifest and PLD-compatible embedded-FLAC Parquet format. They are curation drafts, not approved training splits.

| Configuration | Rows | Text-reviewed | Padded audio seconds |
| --- | ---: | ---: | ---: |
| `fil_collected_vad_982_v1` | 982 | 961 | 3,730.272 |
| `hil_collected_vad_718_v1` | 718 | 705 | 1,892.95 |
| `hil_collected_vad_667_v2` | 667 | 654 | 1,596.56 |
| `fil_collected_vad_482_v1` | 482 | 471 | 1,924.68 |
| `hil_collected_vad_667_v1` | 667 | 531 | 1,596.56 |
| `hil_collected_vad_200_reviewed_v2` | 200 | 194 | 489.686 |
| `fil_collected_vad_100_v1` | 100 | 99 | 292.476 |

The 718-row Hiligaynon configuration includes the 667-row snapshots plus 51 new segments from different original recordings. The two 667-row Hiligaynon configurations contain the same audio and transcripts and include the entire 200-row set. The 982-row Filipino configuration includes the 482-row set, which includes the 100-row set. Both also overlap earlier collected pilots. Do not concatenate configurations or sum their durations as unique hours. All listed rows are unassigned, with source rights unknown and zero audio approvals. Text review preserves code-switching and does not prove fidelity to audio.

For the older 667-row v1 snapshot, 136 rows remain draft-only: eleven rejected reviews and 125 not attempted at snapshot time. Later reviews are tracked separately and do not mutate this release. Audio and transcripts were unchanged during attachment. All 667 embedded PCM arrays were compared against local source derivatives; all ten remote artifacts were hash-verified. Pinned Hugging Face streaming returned 667 rows with exactly 531 `text_reviewed` and 136 `draft` statuses.

Older v1 data revision: `cad388ea5b586ef63907d196d271d1ea6b969843`. Dataset-card/config revision: `80641c19769f064f7e3a757cbabb9fe0af6c6e80`.

The Hiligaynon v2 snapshot adds the final completed reviews: 654 valid text reviews and 13 rejected reviews left draft-only. The Filipino 482-row snapshot has 471 valid reviews and 11 rejected reviews left draft-only. Every embedded PCM array and sentence was checked against the corresponding manifest (667 and 482 matches); all ten Hiligaynon and eight Filipino remote artifacts were hash-verified. Neither snapshot grants audio approval or changes the ASR text.

Pinned streaming checks returned exactly 667 Hiligaynon rows (654 text-reviewed, 13 draft) and 482 Filipino rows (471 text-reviewed, 11 draft), with matching ID sets, sentences and review states.

Shared dataset-card/config revision: `aa3b647061474e2d66e8677ebc80e66e26372dca`.

Data revisions: Hiligaynon v2 `42532f747e25ca9f6edf7530c1104117eed199a8`; Filipino 482 `1eeac74a04cca6eac16aeb2e4cd5209611e31adb`.

```python
from datasets import Audio, load_dataset
rows = load_dataset(
    "sapinsapin/chika", "fil_collected_vad_982_v1",
    revision="f24f5047cfa44c4a2644f2bd7c1ddb566036c5c5",
    split="unassigned", streaming=True, token=True,
).cast_column("audio", Audio(decode=False))
```

The existing PLD configurations and earlier pilots remain reproducible; see the dataset card for their schema migration and reserved-test restrictions. Operational steps, review failure policy, and source-interval accounting are in the [curation playbook](dataset_curation.md). The [preparation checkpoint](../curation/PREPARATION_CHECKPOINT.json) describes the separate five-language pending-ASR candidates; those are not these transcribed releases.

The 718-row Hiligaynon release combines the previous 667-row snapshot with 51 new nonempty ASR drafts, all of which passed text-review validation. The persistent ASR batch then stopped on a server finalization timeout; the failed request is excluded. All 718 PCM arrays and sentences matched the local source manifest. Eleven remote artifacts were hash-verified at data revision `ab48197c220bf6cbc4c5ea99ae2891be0f4d1102`. It remains a private unassigned draft with unknown rights and zero audio approvals.

The 718-row configuration was read back at card revision `40a461007fee00cf77169f51eea21f537d86535f`: exact ID set, sentences, and review statuses matched (705 text-reviewed, 13 draft).

The 982-row Filipino release contains the prior 482-row snapshot plus 500 new nonempty drafts, with 961 validated text reviews and 21 draft-only rows overall. All 982 PCM arrays and sentences matched local source derivatives; all thirteen remote artifacts were hash-verified at data revision `2b04e79bc36bf65d7d020e9322a7ca529e7d84a4`. Configuration/card revision: `f24f5047cfa44c4a2644f2bd7c1ddb566036c5c5`. Source permissions remain unknown, splits unassigned, audio approvals zero.

Pinned streaming at `f24f5047cfa44c4a2644f2bd7c1ddb566036c5c5` returned exactly 982 rows, matching IDs, unchanged sentences and expected statuses (961 text-reviewed, 21 draft).

### Filipino 5,671-clip export — publication pending

The next immutable curation snapshot combines completed Filipino v4 with the prior 982-clip release and 100 completed Kimi reviews. It contains 5,671 unique clip IDs, 1,061 text-reviewed rows and 4,610 drafts, including 19 empty transcripts. Total padded duration is 25,099.226375 seconds (6.972007326 hours), not unique or verified speech. All overlapping IDs matched transcript/audio/ASR identities; all 5,671 exported PCM arrays and sentences matched local inputs exactly. No audio approvals or training permissions were granted.

Tracked checkpoint: `curation/FIL_5671_EXPORT_CHECKPOINT.json`. Private artifacts: `outputs/collected-fil-5671-v1/`. Upload was started at `data/fil_collected_vad_5671_v1`; publication receipt, remote verification, card configuration and streaming verification are still pending. Inspect `publisher.json` and the actual process before recovery. This snapshot overlaps older Filipino configurations; never sum or concatenate them as independent data.

A one-shot finalizer now waits for this publisher's verified private receipt, then runs `outputs/publish-fil-5671-card.py` to register the configuration and verify all streamed rows at the resulting immutable revision. Its state/log live beside `publisher.json` as `finalizer.json` and `finalizer.log`. It has a per-release lock and checks the actual publisher process while waiting. Publisher failure, uncertain process state, one-hour observation deadline or card/stream failure stops the finalizer without restarting or replaying the publisher. Inspect receipts before recovery, especially if a card commit succeeded but streaming verification failed. The finalizer's launch does not establish publication completion.

### Filipino 5,671 v2 manifest staged only

Completed GLM952 and Kimi491 reviews were joined into the existing 5,671-clip manifest, producing 2,504 text-reviewed rows and 3,167 drafts. Every incoming row matched the prior audio/transcript/ASR identity; transcripts are unchanged. `curation/FIL_5671_V2_MANIFEST_CHECKPOINT.json` freezes source and output hashes. This is only a staged manifest under `outputs/collected-fil-5671-v2/`: no Parquet export or upload has started, and it does not alter v1's in-flight publication.

### Filipino 5,671 v1 data and card verified; stream audit running

All sixty remote export artifacts were freshly downloaded and hash/size verified at data revision `d1daed197a5ea2fbaae3eb25e2b52802f356a418`. The private dataset card now registers `fil_collected_vad_5671_v1`; exact README readback passed at `e3d9886dbcb63f8dc784f2e37d8638fdde05f0f3`. The finalizer is still streaming all rows at the card revision; full release verification is not yet claimed. The tracked export checkpoint now separates these completed data/card stages from the pending stream audit.


### Filipino 5,671 v1 release verification complete

The finalizer completed successfully. Streaming at immutable revision `e3d9886dbcb63f8dc784f2e37d8638fdde05f0f3` matched all 5,671 IDs, exact transcripts and review statuses (1,061 text-reviewed / 4,610 draft), with nonempty embedded audio and unassigned splits. Data, card readback and complete stream checks have now all passed.

There are 5,189 distinct audio hashes among these 5,671 IDs; 482 rows repeat exact audio bytes. Publication verification does not imply acoustic uniqueness or training approval. The later 7,460-row deduplicated staging manifest supersedes this collection for future export planning, but has not yet been published as a dataset configuration.

### Filipino 4,000-clip remote snapshot verified (`fil_dwnd_asr_4000_v1`)

Remote ASR queues v4 and v5 (2,000 rows each, zero failures) were merged on dwnd-01 into one immutable manifest after re-running every per-batch offline audit (40 audits, no server requests). The merge requires 4,000 unique IDs and audio hashes, rehashes every audio file, and shares **no ID or audio hash** with `fil_dwnd_asr_2000_v1`. 4,000 rows, 3,985 nonempty, 17,245.148 padded seconds (4.79 nonempty hours). Manifest SHA256 `5ea756c565fb6d1d64e0bab564753777294f34bcd0bb3405cbea51393bb0b335`.

Exported to 40 PLD-style curation shards, transferred and re-hashed (40/40), published privately at data revision `6b502458b733670e02619f2ef4f06812752ef763`, card registered at `186e68d52dbea3f5aef3bc2060857e1ac9264f62`. Full streaming at the card revision matched all 4,000 IDs, sentences, review states, provenance and audio bytes. An independent check matched all 43 remote sizes and 41 LFS hashes. Evidence: `curation/DWND_FIL_4000_PUBLICATION_COMPLETE.json`; scripts in `curation/job_scripts/2026-11-15/`. Draft, unassigned, rights unknown, zero audio approvals. It does not overlap the 2,000-clip remote snapshot, so the two may be combined after provenance-hash deduplication; it may still overlap Mac-collected `fil_collected_*` audio from the same podcasts at the source-interval level.

### Hiligaynon 1,181-clip remote snapshot verified (`hil_dwnd_asr_1181_v1`)

Completed rows from remote Hiligaynon queues v1-v3 (Aksyon and GMA Ratsada broadcasts) merged on dwnd-01 with `curation/job_scripts/2026-11-15/merge_export_queue.py`: 29 complete batches with the full offline audit, plus 40 partial batches whose completed rows are exactly the input prefix before the quarantined row (audited against a derived subset input). 1,181 rows, 1,180 nonempty, 5,313.585 padded seconds (1.48 nonempty hours); manifest SHA256 `7ac9143d326e3755f6bd0409d78164274a7e1935a7ef88b5099affa25c30c7f6`. No audio hash or source is shared with any earlier Hiligaynon configuration. The 42 quarantined requests are excluded and must not be replayed. Exported to 12 shards and re-hashed after transfer. Publication attempts 1-5 were refused by the Hub's 128-commits-per-hour repository limit (used by the storage supervisor's archive commits) before any upload; a guarded retry succeeded on attempt 6. Published privately at data revision `546895f63eecea86150df009e9c36b6728439d43`, card registered at `0f1da56d98911db0ac49404423a79b91ba46ad64`. Full streaming at the card revision matched all 1,181 IDs, sentences, review states, provenance and audio bytes; an independent check matched all 15 remote sizes and 12 LFS hashes. Evidence: `curation/DWND_HIL_1181_PUBLICATION_COMPLETE.json`. Draft, unassigned, rights unknown, zero audio approvals.

### Filipino 9,927-clip remote snapshot verified (`fil_dwnd_asr_9927_v1`)

Remote Filipino queue v6 (10,000 frozen rows, 100 batches of 100) merged on dwnd-01 with `curation/job_scripts/2026-11-15/merge_export_queue.py`: 99 complete batches with the full offline audit and batch-013, whose 27 completed rows are exactly the input prefix before its quarantined finalize timeout (audited against a derived subset input; the 72 later rows were never attempted). 9,927 rows, 9,881 nonempty, 44,233.244 padded seconds (12.28 nonempty hours); manifest SHA256 `56e9ba5a9b78708820d3f10409f966fa6e5456e35bbaf0fe41efe1f5aa871f1c`. The merge refused any ID or audio overlap with the first1000, smoke, 2000 and 4000 Filipino manifests and found none. The one quarantined request is excluded and must not be replayed. Exported to 100 shards; all 103 export files were sha256-identical after transfer. The first publication attempt stopped before upload because `curation.publish` bounds an export at 100 files by default; it was rerun with `max_files=110` (now the optional fifth argument of `publish_export.py`). Published privately at data revision `4fd3e512670aa2145067cdabdbcd9fe7928ced0e`, card registered at `401cae1c237dacb1b9c735d3fccbd2130c3f6e50`. Full streaming at the card revision matched all 9,927 IDs, sentences, review states, provenance and audio bytes; an independent check matched all 103 remote sizes and 101 LFS hashes. Evidence: `curation/DWND_FIL_9927_PUBLICATION_COMPLETE.json`. Draft, unassigned, rights unknown, zero audio approvals.
