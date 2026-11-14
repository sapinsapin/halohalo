---
language:
- fil
- tgl
- hil
- eng
license: other
license_name: source-specific-rights-unresolved
pretty_name: Chika — Filipino and Hiligaynon Speech Curation
multilinguality:
- multilingual
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- philippines
- taglish
- hiligaynon
- code-switching
- work-in-progress
configs:
- config_name: hil_pld_curation_v1
  data_files:
  - split: unassigned
    path: curation/pld-hil-v1/curation-unassigned-*.parquet
  - split: reserved_test
    path: curation/pld-hil-v1/curation-reserved_test-*.parquet
---

# Chika

A work-in-progress collection of conversational Filipino/Taglish and Hiligaynon speech, preserving natural English code-switching. The target is approximately **250 unique retained, transcribed hours per language group**. This target is not a claim about currently completed data.

Tracking and workflow: [sapinsapin/halohalo issue #8](https://github.com/sapinsapin/halohalo/issues/8). The repository is private while source rights and curation are unresolved.

## Available configuration

`hil_pld_curation_v1` contains **229 PLD spontaneous recordings, 2.9522 hours**, with machine-generated transcript drafts and text-only review. There are **190 unassigned and 39 reserved-test recordings**. These recordings are an initial schema migration/curation set; they do not establish completion of the new 250-hour Hiligaynon collection. No audio-verified training labels are claimed. PLD source restrictions remain applicable.

```python
from datasets import load_dataset
rows = load_dataset("sapinsapin/chika", "hil_pld_curation_v1", split="unassigned", streaming=True)
row = next(iter(rows))
print(row["sentence"], row["review_status"])
```

`raw/` contains source audio backups; `checkpoints/` contains acquisition ledgers. Neither is an approved training split. Runtime totals in acquisition ledgers include music, ads and silence. Feed estimates, downloaded hours, retained speech, ASR-completed hours, LLM-reviewed hours and human/audio-verified hours must be reported separately. Simulcasts, excerpts, reposts and alternate encodings require acoustic deduplication before unique-hour accounting.

## Format

Following PLD, each Parquet row embeds `audio` as `{bytes, path}` with 16 kHz mono FLAC, plus `sentence`, `duration`, `speaker_id`, `speech_type`, `language` and `num_words`. Added fields are `id`, `split`, `schema_version`, `review_status` and `provenance_json`. A complete manifest sidecar preserves source questions, source hashes, original timestamps, review evidence, rights and original sample rate.

For spontaneous PLD speech, `sentence` is the draft transcription, **never the question**. The elicitation question remains separate in provenance and is excluded from ASR and text-review inputs. Frozen test reservations must not be used for training. Source recording and known speaker grouping governs future splits; unknown speakers do not imply speaker-disjointness.

## Language and review

Filipino–English and Hiligaynon–English code-switching are valid speech, including local affixes on English roots. No language purification or translation is applied. Source labels are provisional until the audio is checked. An English title or feed-level language tag does not determine the recording's spoken language.

LLM review checks textual plausibility and flags uncertainty. It does not prove fidelity to audio, approve a correction, or provide a ground-truth WER/CER reference. Raw drafts and suggestions remain distinguishable. TTS use additionally requires speaker consistency, speech-only segments and accurate text/audio alignment.

## Sources and rights

Sources include public podcast enclosures, regional news/video archives, bounded radio captures, and the existing PLD curation pilot. Exact source URLs, retrieval metadata, hashes and restrictions remain attached to each record. Public download availability does not grant commercial training or redistribution rights. There is **no blanket permissive license** for this collection. Unknown rights remain unknown; restricted sources must be excluded from any incompatible release or training profile. Private storage does not resolve these rights.

Synthetic audio, if later included, will have explicit model/version/license provenance and will be counted separately from unique real recorded hours. No synthetic hours currently contribute to the acquisition target.
