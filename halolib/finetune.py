"""
Dataset adapters for finetuning workflows — normalize the speech corpora to a
single schema so TTS/ASR trainers can swap datasets with a flag.

Both sources come out as a DatasetDict with:
  audio      — Audio(sampling_rate=16000)  (resampled on the fly if needed)
  text       — transcription string
  speaker_id — corpus-wide unique speaker key

Sources:
  fsc        — sapinsapin/filipinospeechcorpus (studio read + spontaneous speech)
  livestream — sapinsapin/halo-livestream (diarized Kumu streams; the tts/asr
               configs are pre-gated by the processing pipeline's QC stage)
"""

from datasets import Audio, DatasetDict, load_dataset

TARGET_SR = 16000

_LIVESTREAM_FILES = {
    "tts": "data/tts/{split}/*.parquet",
    "asr": "data/asr/{split}/*.parquet",
}

# TTS wants clean, bounded utterances; ASR tolerates the full range.
_FSC_FILTERS = {
    "tts": lambda r: (r["speech_type"] == "read"
                      and 1.0 <= r["duration"] <= 15.0
                      and len(r["sentence"].split()) >= 3),
    "asr": lambda r: 0.3 <= r["duration"] <= 30.0,
}


def load_speech_dataset(
    name: str,
    task: str,
    livestream_repo: str = "sapinsapin/halo-livestream",
    fsc_repo: str = "sapinsapin/filipinospeechcorpus",
    max_samples: int | None = None,
    token: str | None = None,
    num_proc: int = 1,
) -> DatasetDict:
    """Load `fsc` or `livestream` normalized to (audio@16k, text, speaker_id)."""
    if task not in ("tts", "asr"):
        raise ValueError(f"task must be tts|asr, got {task!r}")

    if name == "fsc":
        ds = load_dataset(fsc_repo, token=token)
        ds = ds.filter(_FSC_FILTERS[task], num_proc=num_proc)
        ds = ds.rename_column("sentence", "text")

    elif name == "livestream":
        from fnmatch import fnmatch

        from huggingface_hub import HfApi
        repo_files = HfApi(token=token).list_repo_files(
            livestream_repo, repo_type="dataset")

        # the test split exists only once the corpus is large enough — build
        # data_files from what is actually in the repo
        pattern = _LIVESTREAM_FILES[task]
        data_files = {}
        for split in ("train", "test"):
            glob = pattern.format(split=split)
            if any(fnmatch(f, glob) for f in repo_files):
                data_files[split] = f"hf://datasets/{livestream_repo}/{glob}"
        if "train" not in data_files:
            raise FileNotFoundError(
                f"{livestream_repo} has no {task} train shards — run "
                f"process_livestream.py --stages export --push first")

        ds = load_dataset("parquet", data_files=data_files, token=token)
        ds = ds.rename_column("sentence", "text")

    else:
        raise ValueError(f"unknown dataset {name!r} (expected fsc|livestream)")

    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    keep = {"audio", "text", "speaker_id"}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    if "test" not in ds:
        ds = DatasetDict({
            "train": ds["train"],
            "test": ds["train"].select(range(min(50, len(ds["train"])))),
        })

    if max_samples:
        for split in ds:
            n = min(max_samples if split == "train" else max(50, max_samples // 10),
                    len(ds[split]))
            ds[split] = ds[split].shuffle(seed=42).select(range(n))

    return ds
