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
