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
_DATASET_SUFFIX = {"fsc": "fsc", "livestream": "halohaloLS", "pld": "pld",
                   "fsc+pld": "fsc-pld"}
_DATASET_REPOS = {"fsc": "sapinsapin/filipinospeechcorpus",
                  "livestream": "sapinsapin/halo-livestream",
                  "pld": "sapinsapin/pld",
                  "fsc+pld": "sapinsapin/pld"}      # namespace lookup only
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


def normalise_text(text: str) -> str:
    """Lowercase, drop stress accents and punctuation, single spaces.

    PLD's transcripts mark stress (ganína, ihúnong) and keep punctuation; 35%
    of Cebuano reference words carry one or the other. Neither is part of how
    the languages are ordinarily written, and an ASR model is not being asked
    for them. Measured 2026-09-21 on the frozen ceb split, scoring the same
    hypotheses with and without them moved whisper-large-v3 from 36.9 to 24.2
    WER and omni-1B from 51.5 to 39.5 — twelve points of "error" in both that
    were orthography. Apostrophes stay: they are letters here (mo'y, di').

    A view over the corpus, not a replacement for it: the raw text is what a
    stress-marking model would need.
    """
    import re
    import unicodedata

    t = unicodedata.normalize("NFD", (text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = unicodedata.normalize("NFC", t).replace("’", "'").replace("‘", "'")
    return " ".join(re.sub(r"[^\w\s']", " ", t).split())


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

    if "+" in name:
        # Combined corpora, e.g. "fsc+pld": load each part (already cast to
        # the shared schema), give each an equal share of max_samples so a
        # 305k-row corpus cannot drown a 50k-row one, then concatenate.
        # Motivated by the FSC/PLD cross-evaluation in docs/pld_models_plan.md
        # §3.2b: each per-corpus model collapses on the other corpus.
        from datasets import concatenate_datasets
        parts = name.split("+")
        share = max_samples // len(parts) if max_samples else None
        loaded = [load_speech_dataset(p, task, livestream_repo=livestream_repo,
                                      fsc_repo=fsc_repo, max_samples=share,
                                      token=token, num_proc=num_proc,
                                      language=language) for p in parts]
        return DatasetDict({
            split: concatenate_datasets([d[split] for d in loaded]).shuffle(seed=42)
            for split in ("train", "test")})

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

        # Cloud path: no raw corpus on the machine, so read the published
        # parquet instead. Set PLD_SOURCE=hub explicitly, or let it fall back
        # automatically when the raw tree is absent (a Nebius VM). The Hub
        # rows carry the same speech_type / text_is_prompt / num_words fields
        # the local index does, so _PLD_FILTERS applies unchanged.
        if os.environ.get("PLD_SOURCE") == "hub" or not root.exists():
            filt = _PLD_FILTERS[task]
            keep_cols = {"audio", "text", "speaker_id"}

            # Selecting one language means three passes over PLD's ~300k rows
            # (the task+language filter, then train and test assignment). That
            # is minutes of idle GPU at the start of *every* run, and the
            # bake-off runs many arms per language. Cache the filtered, split
            # corpus once per (task, language, split kind) and reuse it.
            split_kind = ("random" if os.environ.get("PLD_SPLIT") == "random"
                          else "frozen")
            cache_root = Path(os.environ.get(
                "PLD_DS_CACHE", Path(os.environ.get("PLD_WORK_DIR", ".")) / "ds_cache"))
            cached = cache_root / f"pld_{task}_{language or 'all'}_{split_kind}"
            if cached.is_dir():
                from datasets import load_from_disk
                print(f"  dataset cache: {cached}")
                ds = load_from_disk(str(cached))
                if max_samples:
                    for split in ds:
                        n = min(max_samples if split == "train"
                                else max(50, max_samples // 10), len(ds[split]))
                        ds[split] = ds[split].shuffle(seed=42).select(range(n))
                return ds

            hub = load_dataset(_DATASET_REPOS["pld"], token=token)

            def _keep(r):
                return filt(r) and (language is None or r["language"] == language)

            hub = hub.filter(_keep, num_proc=num_proc)
            # Same frozen split as the local path: rebuild train/test from the
            # pooled rows rather than trusting the published random split,
            # whose speakers and prompts overlap.
            if language and os.environ.get("PLD_SPLIT") != "random":
                try:
                    from datasets import concatenate_datasets

                    from halolib.splits import assign, load_spec
                    spec = load_spec(language)
                    pooled = concatenate_datasets([hub[s] for s in sorted(hub)])
                    hub = DatasetDict({
                        w: pooled.filter(lambda r, w=w: assign(r, spec) == w,
                                         num_proc=num_proc)
                        for w in ("train", "test")})
                    print(f"  split: frozen speaker+prompt-disjoint "
                          f"({len(hub['train'])} train / {len(hub['test'])} test)")
                except FileNotFoundError:
                    print("  split: published random split — speakers and "
                          "prompts overlap; in-domain numbers only")
            hub = hub.rename_column("sentence", "text")
            hub = hub.remove_columns(
                [c for c in hub["train"].column_names if c not in keep_cols])
            if not len(hub["train"]):
                raise ValueError(
                    f"no PLD rows on the Hub for task={task} language={language!r}")
            ds = hub
            # fall through to the shared cast / max_samples handling below
            ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
            # cache before subsetting, so runs with different --max-samples
            # share it. Written to a temp dir first: a job killed mid-write
            # must not leave a half-corpus that later runs would trust.
            try:
                tmp = cached.with_name(cached.name + f".tmp{os.getpid()}")
                ds.save_to_disk(str(tmp))
                tmp.rename(cached)
                print(f"  dataset cached: {cached}")
            except Exception as e:      # a cache is an optimisation, never a hard failure
                print(f"  dataset cache skipped ({type(e).__name__}: {e})")
            if max_samples:
                for split in ds:
                    n = min(max_samples if split == "train"
                            else max(50, max_samples // 10), len(ds[split]))
                    ds[split] = ds[split].shuffle(seed=42).select(range(n))
            return ds

        if not root.exists():
            raise FileNotFoundError(
                f"PLD raw corpus not found at {root}; set PLD_RAW, or set "
                f"PLD_SOURCE=hub to read sapinsapin/pld from the Hub")

        entries, _ = index_corpus(root)
        filt = _PLD_FILTERS[task]

        # Frozen speaker- AND prompt-disjoint split when one exists for this
        # language. PLD is a prompt corpus, so a speaker-only split still
        # trains on every test sentence and the resulting CER is in-domain
        # only (docs/pld_sota_track.md §6). Falls back to the historical
        # random split when no spec is built, so older runs stay reproducible.
        spec = None
        # PLD_SPLIT=random forces the historical split. Needed whenever a run
        # must stay comparable with a model already published on that split,
        # because compare_and_push_asr.py gates against the published CER.
        if language and os.environ.get("PLD_SPLIT") != "random":
            try:
                from halolib.splits import assign, load_spec
                spec = load_spec(language)
                s = spec["stats"]
                print(f"  split: frozen speaker+prompt-disjoint "
                      f"({s['train']} train / {s['test']} test rows available)")
            except FileNotFoundError:
                print("  split: random over utterances — no frozen spec, so "
                      "speakers and prompts overlap; in-domain numbers only")

        rows = []
        for e in entries:
            if not filt(e) or (language is not None and e["language"] != language):
                continue
            row = {"audio": str(e["wav_path"]), "text": e["sentence"],
                   "speaker_id": e["speaker_id"]}
            if spec is not None:
                where = assign(e, spec)
                if where is None:          # the disjointness remainder
                    continue
                row["_split"] = where
            rows.append(row)
        if not rows:
            raise ValueError(f"no PLD rows for task={task} language={language!r}")
        random.Random(42).shuffle(rows)     # session order → mixed speakers

        # A handful of WAVs in the corpus are unreadable ("Format not
        # recognised"). datasets.Audio decodes lazily and raises mid-training,
        # so header-check the rows we are about to use and drop the bad ones.
        # Only the candidate slice is checked — validating all 334k files
        # would cost minutes for no benefit.
        import soundfile as sf

        if spec is None:
            budget = {"train": (max_samples + 400) if max_samples else len(rows)}
        else:
            n = max_samples or len(rows)
            budget = {"train": n, "test": max(50, n // 10)}
        keep = {"train": [], "test": []}
        bad = 0
        for r in rows:
            where = r.pop("_split", "train")
            if len(keep[where]) >= budget.get(where, 0):
                if all(len(keep[k]) >= v for k, v in budget.items()):
                    break
                continue
            try:
                sf.info(r["audio"])
            except Exception:
                bad += 1
                continue
            keep[where].append(r)
        if bad:
            print(f"  skipped {bad} unreadable wav(s)")
        if not keep["train"]:
            raise ValueError(f"no readable PLD audio for language={language!r}")
        ds = DatasetDict({"train": Dataset.from_list(keep["train"])})
        if keep["test"]:
            ds["test"] = Dataset.from_list(keep["test"])

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


def latest_checkpoint(out_dir: str | Path) -> str | None:
    """Newest checkpoint that was fully written, or None.

    A VM that dies mid-save leaves a checkpoint directory holding only the
    config files; handing that to trainer.train() fails with "Can't find a
    valid checkpoint" and the run never restarts. Skip back to the newest
    one that has both weights and trainer state.
    """
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for c in ckpts:
        weights = any((c / f).exists() for f in (
            "model.safetensors", "model.safetensors.index.json",
            "pytorch_model.bin", "pytorch_model.bin.index.json"))
        if weights and (c / "trainer_state.json").exists():
            return str(c)
        print(f"resume: skipping incomplete {c.name}")
    return None


_LANG_NAMES = {"bcl": "Central Bikol", "ceb": "Cebuano", "eng": "Philippine English",
               "fil": "Filipino", "hil": "Hiligaynon", "ilo": "Ilocano",
               "pag": "Pangasinan", "pam": "Kapampangan", "tsg": "Tausug",
               "war": "Waray", "tl": "Filipino"}


def _model_card(repo_id, base_model, dataset_name, task, lang_code, license,
                extra_tags, metrics, train_summary, suffix) -> str:
    """The README for a published model. Every section is there so a reader
    who never opens the repo can answer: what is this, what was it trained on,
    how was the number measured, how do I run it, and what should I not
    conclude from it."""
    name = repo_id.split("/")[1]
    lang = _LANG_NAMES.get(lang_code, lang_code)
    norm = suffix.endswith("-norm")
    is_whisper = "whisper" in base_model
    is_ctc = "omniASR" in base_model or "w2v" in base_model
    is_orpheus = "orpheus" in base_model
    frozen = "frozen" in train_summary or "disjoint" in train_summary
    tag_lines = "".join(f"- {t}\n" for t in
                        (extra_tags if extra_tags is not None else ["filipino", "tagalog"]))
    metric_lines = "".join(f"| {k} | {v:.4f} |\n" for k, v in metrics.items())
    ds_repo = _DATASET_REPOS[dataset_name]

    if frozen:
        split_text = (
            "**Frozen speaker- and prompt-disjoint split** of PLD (`splits/pld_*.json` "
            "in the repo): no test speaker and no test sentence appears anywhere in "
            "training. Numbers on this split are not comparable with the in-domain "
            "figures on the dataset card, which share both, and are typically "
            "several times higher for that reason.")
    else:
        split_text = (
            "Random split of the corpus, which shares speakers and prompt sentences "
            "between train and test. Optimistic; not comparable with speaker-disjoint "
            "numbers.")

    if task == "tts":
        metric_text = (
            "Round-trip intelligibility: 50 frozen test sentences are synthesized, "
            "transcribed by an ASR judge (`whisper-large-v3-pld-<lang>` where it "
            "exists, else `whisper-small-pld-<lang>`), and scored against the text. "
            "`cer`/`wer` are character and word error rates after lowercasing and "
            "stripping punctuation and accents. `spk_sim` is ECAPA cosine similarity "
            "to the human recording of the same sentence (1.0 = same voice). The "
            "human recordings themselves score 1-3% CER through the same judge; "
            "that is the floor.")
    elif norm:
        metric_text = (
            "`cer` and `wer` are character and word error rates on the whole test "
            "split, hypothesis and reference both lowercased, whitespace-normalised, "
            "and with stress accents and punctuation removed "
            "(`halolib.finetune.normalise_text`). PLD marks stress on about a third "
            "of words and an ASR model is not asked for it: scoring the same "
            "hypotheses with and without them moved whisper-large-v3 on Cebuano "
            "from 36.9 to 24.2 WER. `eval_loss` is the training objective on the "
            "test split.")
    else:
        metric_text = (
            "`cer` and `wer` are character and word error rates on the whole test "
            "split, hypothesis and reference both lowercased and whitespace-"
            "normalised. Accents and punctuation in the reference count as errors "
            "when missing, which is strict: PLD marks stress on about a third of "
            "words. The `-norm` variant of this model uses the other convention. "
            "`eval_loss` is the training objective on the test split.")

    if is_whisper:
        usage = f"""```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch, soundfile as sf

proc = WhisperProcessor.from_pretrained("{repo_id}")
model = WhisperForConditionalGeneration.from_pretrained("{repo_id}").eval()
wav, sr = sf.read("clip.wav")          # 16 kHz mono
feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features
with torch.no_grad():
    ids = model.generate(feats, task="transcribe", max_new_tokens=200)
print(proc.batch_decode(ids, skip_special_tokens=True)[0])
```
Languages other than Filipino and English were trained under Whisper's `<|tl|>`
token, the closest one it has; do not pass a `language=` argument."""
        if norm:
            usage += "\nOutput is lowercase with no punctuation or accents."
    elif is_ctc:
        usage = f"""```python
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC
import torch, json, soundfile as sf
from huggingface_hub import hf_hub_download

repo = "{repo_id}"
extractor = AutoFeatureExtractor.from_pretrained(repo)
model = Wav2Vec2ForCTC.from_pretrained(repo).eval()
vocab = json.load(open(hf_hub_download(repo, "vocab.json")))
id2unit = {{i: u for u, i in vocab.items()}}

wav, sr = sf.read("clip.wav")          # 16 kHz mono
x = extractor(wav, sampling_rate=16000, return_tensors="pt").input_values
with torch.no_grad():
    ids = model(x).logits[0].argmax(-1).tolist()
out, prev = [], None                    # greedy CTC: collapse repeats, drop blank (0)
for i in ids:
    if i != prev and i != 0:
        out.append(id2unit[i])
    prev = i
print("".join(out).replace("|", " ").strip())
```
A 4-gram word LM over PLD's training transcripts, fused with `pyctcdecode`,
lowers WER by a further 4-7 points (`scripts/ctc_lm_eval.py` in the repo)."""
    elif is_orpheus:
        units = suffix.split("-")[0]
        usage = f"""This is a LoRA adapter on `{base_model}`; audio is SNAC codec tokens.
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = "{base_model}"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, "{repo_id}").eval()
```
Then follow `finetune_orpheus.py` in the halohalo repo: build the prompt with
`frontend_text(text, "{units}")` -- **the adapter was trained on `{units}` text and
must be prompted the same way** -- wrap it in the SOH/EOT/EOH/SOAI/SOS framing,
generate, and decode the tokens with `decode_tokens` through SNAC 24 kHz.
`scripts/tts_eval.py --stage synth` does all of this."""
    else:
        usage = f"See `finetune_{task}.py` in the halohalo repo."

    caveats = []
    if task == "tts":
        caveats += ["50 sentences per language resolves a gap of a few CER points, "
                    "not of one.",
                    "The judge is our own ASR model, trained on the same corpus; an "
                    "independent judge is still owed before these numbers are cited.",
                    "Read, prompted speech only. Spontaneous or noisy input is out of "
                    "domain."]
    else:
        caveats += ["Trained on read, prompted speech; accuracy drops on spontaneous "
                    "or noisy audio.",
                    "One corpus, one recording setup. Cross-corpus tests on Filipino "
                    "showed large drops for models of this kind."]
    if not frozen:
        caveats.append("Scored on an overlapping split; expect much worse on unseen speakers.")
    caveat_lines = "".join(f"- {c}\n" for c in caveats)
    kind = "text-to-speech" if task == "tts" else "speech recognition"

    return f"""---
language: {lang_code}
license: {license}
library_name: {"peft" if is_orpheus else "transformers"}
pipeline_tag: {_TASK_TAGS[task]}
base_model: {base_model}
datasets:
- {ds_repo}
tags:
- {_TASK_TAGS[task]}
{tag_lines}---

# {name}

**{lang} {kind}**: [`{base_model}`](https://huggingface.co/{base_model})
finetuned on the [Philippine Language Dataset](https://huggingface.co/datasets/{ds_repo})
(PLD), read speech collected by the UP Diliman Digital Signal Processing
Laboratory. Part of the [halohalo](https://github.com/sapinsapin/halohalo) project.

## Method

{train_summary}

## Evaluation

{split_text}

{metric_text}

| metric | value |
|---|---|
{metric_lines}
## How to use

{usage}

## Caveats

{caveat_lines}
## Licence

`{license}`. PLD is CC-BY-NC and research-only, so every model trained on it
inherits that regardless of the base model's own licence. The base model's
terms apply in addition.

Trained with `finetune_{task}.py` from halohalo; the dataset adapter
normalises each corpus to `(audio@16k, text, speaker_id)`.
"""


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
    cards_only: bool = False,
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
    card = _model_card(repo_id, base_model, dataset_name, task, lang_code,
                       license, extra_tags, metrics or {}, train_summary or "",
                       suffix or "")
    (Path(final_dir) / "README.md").write_text(card, encoding="utf-8")

    if cards_only:
        # refresh the README of a repo whose weights are already up
        api.upload_file(path_or_fileobj=str(Path(final_dir) / "README.md"),
                        path_in_repo="README.md", repo_id=repo_id,
                        commit_message="Refresh model card")
        url = f"https://huggingface.co/{repo_id}"
        print(f"Card refreshed: {url}")
        return url
    api.upload_folder(folder_path=str(final_dir), repo_id=repo_id,
                      commit_message=f"Upload {task} finetune ({dataset_name})")
    for f in sample_files or []:
        api.upload_file(path_or_fileobj=str(f), repo_id=repo_id,
                        path_in_repo=f"samples/{Path(f).name}",
                        commit_message=f"Add sample {Path(f).name}")
    url = f"https://huggingface.co/{repo_id}"
    print(f"Pushed: {url}")
    return url
