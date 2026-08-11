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

from pathlib import Path

from datasets import Audio, DatasetDict, load_dataset

TARGET_SR = 16000

# hub naming: <base model basename>-<corpus suffix>
_DATASET_SUFFIX = {"fsc": "fsc", "livestream": "halohaloLS", "pld": "pld"}
_DATASET_REPOS = {"fsc": "sapinsapin/filipinospeechcorpus",
                  "livestream": "sapinsapin/halo-livestream",
                  "pld": "sapinsapin/pld"}
_TASK_TAGS = {"tts": "text-to-speech", "asr": "automatic-speech-recognition",
              "s2s": "audio-to-audio"}

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

# PLD's index has no duration (that would mean decoding 334k headers); length
# is bounded later by the trainers' token/mel caps instead.
_PLD_FILTERS = {
    "tts": lambda r: (r["speech_type"] == "read"
                      and not r["text_is_prompt"]
                      and r["num_words"] >= 3),
    "asr": lambda r: not r["text_is_prompt"],
}


def load_speech_dataset(
    name: str,
    task: str,
    livestream_repo: str = "sapinsapin/halo-livestream",
    fsc_repo: str = "sapinsapin/filipinospeechcorpus",
    max_samples: int | None = None,
    token: str | None = None,
    num_proc: int = 1,
    language: str | None = None,
) -> DatasetDict:
    """Load `fsc`, `livestream` or `pld` normalized to (audio@16k, text, speaker_id).

    `language` filters `pld` to one ISO 639-3 code (e.g. "bcl"); it is ignored
    for the single-language corpora.
    """
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

    elif name == "pld":
        # Local-first: index the raw corpus on disk (seconds) instead of
        # pulling 24GB of parquet back off the Hub.
        import os
        import random

        from datasets import Dataset

        from halolib.pld import index_corpus

        root = Path(os.environ.get(
            "PLD_RAW", "/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD"))
        if not root.exists():
            raise FileNotFoundError(
                f"PLD raw corpus not found at {root}; set PLD_RAW or load "
                f"from the Hub dataset sapinsapin/pld instead")

        entries, _ = index_corpus(root)
        filt = _PLD_FILTERS[task]
        rows = [{"audio": str(e["wav_path"]),
                 "text": e["sentence"],
                 "speaker_id": e["speaker_id"]}
                for e in entries
                if filt(e) and (language is None or e["language"] == language)]
        if not rows:
            raise ValueError(f"no PLD rows for task={task} language={language!r}")
        random.Random(42).shuffle(rows)     # session order → mixed speakers

        # A handful of WAVs in the corpus are unreadable ("Format not
        # recognised"). datasets.Audio decodes lazily and raises mid-training,
        # so header-check the rows we are about to use and drop the bad ones.
        # Only the candidate slice is checked — validating all 334k files
        # would cost minutes for no benefit.
        import soundfile as sf

        need = (max_samples + 400) if max_samples else len(rows)
        keep, bad = [], 0
        for r in rows:
            if len(keep) >= need:
                break
            try:
                sf.info(r["audio"])
            except Exception:
                bad += 1
                continue
            keep.append(r)
        if bad:
            print(f"  skipped {bad} unreadable wav(s)")
        if not keep:
            raise ValueError(f"no readable PLD audio for language={language!r}")
        ds = DatasetDict({"train": Dataset.from_list(keep)})

    else:
        raise ValueError(f"unknown dataset {name!r} (expected fsc|livestream|pld)")

    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    keep = {"audio", "text", "speaker_id"}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    if "test" not in ds:
        n = len(ds["train"])
        if n > 200:
            # disjoint held-out slice
            k = min(200, n // 10)
            ds = DatasetDict({"train": ds["train"].select(range(k, n)),
                              "test": ds["train"].select(range(k))})
        else:
            # corpus too small to give rows away (e.g. livestream seed):
            # overlap train so training still works, eval is nominal
            ds = DatasetDict({"train": ds["train"],
                              "test": ds["train"].select(range(min(50, n)))})

    if max_samples:
        for split in ds:
            n = min(max_samples if split == "train" else max(50, max_samples // 10),
                    len(ds[split]))
            ds[split] = ds[split].shuffle(seed=42).select(range(n))

    return ds


def push_model_to_hub(
    final_dir: str | Path,
    base_model: str,
    dataset_name: str,
    task: str,
    token: str | None = None,
    metrics: dict | None = None,
    train_summary: str | None = None,
    sample_files: list[str | Path] | None = None,
    license: str = "mit",
    namespace: str | None = None,
    suffix: str | None = None,
    lang_code: str = "tl",
    extra_tags: list[str] | None = None,
) -> str:
    """Upload a finetuned model dir as <ns>/<base basename>-<corpus suffix>.

    `base_model` must be the canonical hub id of the base (not a resumed local
    checkpoint path) so the repo name stays stable across continued runs.
    Models default to the same namespace as the corpora so the project stays
    in one place; falls back to the token's user if that org isn't writable.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    me = api.whoami()
    if namespace is None:
        corpus_org = _DATASET_REPOS[dataset_name].split("/")[0]
        namespace = (corpus_org if corpus_org in
                     [o["name"] for o in me.get("orgs", [])] else me["name"])
    repo_id = (f"{namespace}/{base_model.split('/')[-1]}-"
               f"{suffix or _DATASET_SUFFIX[dataset_name]}")
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    tag_lines = "".join(f"- {t}\n" for t in
                        (extra_tags if extra_tags is not None
                         else ["filipino", "tagalog"]))
    metric_lines = "".join(f"| {k} | {v:.4f} |\n" for k, v in (metrics or {}).items())
    card = f"""---
language: {lang_code}
license: {license}
library_name: transformers
pipeline_tag: {_TASK_TAGS[task]}
base_model: {base_model}
datasets:
- {_DATASET_REPOS[dataset_name]}
tags:
- {_TASK_TAGS[task]}
{tag_lines}---

# {repo_id.split('/')[1]}

[`{base_model}`](https://huggingface.co/{base_model}) finetuned on
[`{_DATASET_REPOS[dataset_name]}`](https://huggingface.co/datasets/{_DATASET_REPOS[dataset_name]}).

{train_summary or ""}

{f"| metric | value |\n|---|---|\n{metric_lines}" if metric_lines else ""}

Trained with `finetune_{task}.py` from the
[halohalo](https://github.com/sapinsapin/halohalo) pipeline; the dataset
adapter normalizes each corpus to `(audio@16k, text, speaker_id)` so corpora
are swappable with a `--dataset` flag.
"""
    (Path(final_dir) / "README.md").write_text(card, encoding="utf-8")

    api.upload_folder(folder_path=str(final_dir), repo_id=repo_id,
                      commit_message=f"Upload {task} finetune ({dataset_name})")
    for f in sample_files or []:
        api.upload_file(path_or_fileobj=str(f), repo_id=repo_id,
                        path_in_repo=f"samples/{Path(f).name}",
                        commit_message=f"Add sample {Path(f).name}")
    url = f"https://huggingface.co/{repo_id}"
    print(f"Pushed: {url}")
    return url
