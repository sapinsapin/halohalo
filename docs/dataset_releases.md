# Chika draft dataset snapshots

These configurations are in the private [sapinsapin/chika dataset](https://huggingface.co/datasets/sapinsapin/chika). They follow the shared `sapinsapin.speech.v1` manifest and PLD-compatible embedded-FLAC Parquet format. They are curation drafts, not approved training splits.

| Configuration | Rows | Text-reviewed | Padded audio seconds |
| --- | ---: | ---: | ---: |
| `hil_collected_vad_667_v1` | 667 | 531 | 1,596.56 |
| `hil_collected_vad_200_reviewed_v2` | 200 | 194 | 489.686 |
| `fil_collected_vad_100_v1` | 100 | 99 | 292.476 |

The 667-row Hiligaynon configuration includes the entire 200-row set. Both also overlap earlier collected pilots. Do not concatenate configurations or sum their durations as unique hours. All listed rows are unassigned, with source rights unknown and zero audio approvals. Text review preserves code-switching and does not prove fidelity to audio.

For the 667-row snapshot, 136 rows remain draft-only: eleven rejected reviews and 125 not attempted at snapshot time. Later reviews are tracked separately and do not mutate this release. Audio and transcripts were unchanged during attachment. All 667 embedded PCM arrays were compared against local source derivatives; all ten remote artifacts were hash-verified. Pinned Hugging Face streaming returned 667 rows with exactly 531 `text_reviewed` and 136 `draft` statuses.

Data revision: `cad388ea5b586ef63907d196d271d1ea6b969843`. Dataset-card/config revision: `80641c19769f064f7e3a757cbabb9fe0af6c6e80`.

```python
from datasets import Audio, load_dataset
rows = load_dataset(
    "sapinsapin/chika", "hil_collected_vad_667_v1",
    revision="80641c19769f064f7e3a757cbabb9fe0af6c6e80",
    split="unassigned", streaming=True, token=True,
).cast_column("audio", Audio(decode=False))
```

The existing PLD configurations and earlier pilots remain reproducible; see the dataset card for their schema migration and reserved-test restrictions. Operational steps, review failure policy, and source-interval accounting are in the [curation playbook](dataset_curation.md). The [preparation checkpoint](../curation/PREPARATION_CHECKPOINT.json) describes the separate five-language pending-ASR candidates; those are not these transcribed releases.
