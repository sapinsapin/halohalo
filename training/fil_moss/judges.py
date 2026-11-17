"""ASR judges, ECAPA speaker similarity and trimmed duration for Filipino MOSS evaluation and rewards (GPU).

Same three families as the Hiligaynon bake-off (``bench/hil_tts/eval/judge_asr.py`` on dwnd), switched to
Filipino: PLD Whisper (in-family), MMS-1b-all with the ``tgl`` adapter and Omnilingual CTC-1B through the same
port (both independent). Differences from the bake-off, on purpose:

* Whisper decodes greedily without the temperature fallback, batched, so every evaluation is deterministic in
  the judge and fast enough to run every ``eval_every_updates``.
* No UTMOS: it is an English naturalness proxy and no gate reads it.

A judge that fails to load is reported, not fatal: worst-judge CER is then taken over the remaining families
(recipe ``judges.fallback``), and every report names the judges that actually scored it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

JUDGE_SR = 16000
TOP_DB = 40.0


def resample(wav, sr_from, sr_to=JUDGE_SR):
    import torchaudio
    if sr_from == sr_to:
        return np.asarray(wav, dtype=np.float32)
    return torchaudio.functional.resample(torch.as_tensor(wav, dtype=torch.float32), sr_from, sr_to).numpy()


def trimmed_seconds(wav, sr):
    """Seconds after trimming leading/trailing audio quieter than 40 dB below peak (bake-off rule)."""
    import librosa
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0 or float(np.abs(wav).max()) == 0.0:
        return 0.0
    y, _ = librosa.effects.trim(wav, top_db=TOP_DB)
    return len(y) / sr


def _snapshot(repo, revision):
    """The cached snapshot directory at the pinned revision, without network and without huggingface_hub's
    completeness check (the same rule as the Hiligaynon bake-off's ``common.hf_snapshot``: only the files a judge
    uses are cached, e.g. mms-1b-all's safetensors and the needed adapters). ``snapshot_download`` would instead
    fetch the whole repository, which stalled the first GB10 benchmark."""
    import os
    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    base = Path(os.environ.get("HF_HUB_CACHE", str(home / "hub"))).expanduser() / ("models--" + repo.replace("/", "--"))
    d = base / "snapshots" / revision
    if not d.is_dir():
        raise FileNotFoundError("%s@%s is not in the local Hugging Face cache (%s)" % (repo, revision, d))
    return str(d)


def make_mms(cfg, device="cuda"):
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    d = _snapshot(cfg["model"], cfg["revision"])
    proc = AutoProcessor.from_pretrained(d, target_lang=cfg["target_lang"])
    model = Wav2Vec2ForCTC.from_pretrained(d, target_lang=cfg["target_lang"], ignore_mismatched_sizes=True)
    model = model.to(device).eval()
    if model.config.vocab_size != len(proc.tokenizer):
        raise AssertionError("MMS adapter vocabulary mismatch for %s" % cfg["target_lang"])

    def transcribe(wav16):
        x = proc(wav16, sampling_rate=JUDGE_SR, return_tensors="pt").input_values.to(device)
        with torch.inference_mode():
            ids = model(x).logits.argmax(-1)[0]
        return proc.decode(ids)
    return transcribe


def make_omni(cfg, device="cuda"):
    code = str(Path(cfg["port_code"]).expanduser())
    if code not in sys.path:
        sys.path.insert(0, code)
    from omni_port import load_omni
    model, sp, blank = load_omni(Path(cfg["port_dir"]).expanduser(), device, torch.bfloat16)

    def transcribe(wav16):
        x = torch.tensor(wav16, dtype=torch.float32)
        x = torch.nn.functional.layer_norm(x, x.shape)
        with torch.inference_mode():
            ids = model(x[None].to(device, torch.bfloat16)).logits[0].float().argmax(-1)
        keep = torch.ones_like(ids, dtype=torch.bool)
        keep[1:] = ids[1:] != ids[:-1]
        ids = [int(i) for i in ids[keep].tolist()]
        return sp.DecodeIds([i for i in ids if not (sp.IsControl(i) or i == blank)])
    return transcribe


def make_whisper(cfg, device="cuda"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    d = _snapshot(cfg["model"], cfg["revision"])
    proc = WhisperProcessor.from_pretrained(d)
    model = WhisperForConditionalGeneration.from_pretrained(d, dtype=torch.bfloat16).to(device).eval()

    def batch(wavs16):
        feats = proc.feature_extractor(list(wavs16), sampling_rate=JUDGE_SR, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(feats.to(device, torch.bfloat16), language=cfg["language_token"],
                                 task="transcribe", do_sample=False, num_beams=1, max_new_tokens=224)
        return [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]

    def transcribe(wav16):
        return batch([wav16])[0]
    transcribe.batch = batch
    return transcribe


MAKERS = {"mms": make_mms, "omni": make_omni, "whisper": make_whisper}


def load(judge_cfg, only=None, device="cuda"):
    """{name: transcribe(wav16) -> text} for one language's judges, plus {name: error} for those that failed."""
    judges, failed = {}, {}
    for name, cfg in judge_cfg.items():
        if only is not None and name not in only:
            continue
        try:
            judges[name] = MAKERS[name](cfg, device)
        except Exception as exc:  # reported in every result; worst-judge falls back to the rest
            failed[name] = "%s: %s" % (type(exc).__name__, exc)
    if not judges:
        raise RuntimeError("no judge loaded: %s" % failed)
    return judges, failed


def transcribe_all(judges, wav16):
    if len(wav16) < JUDGE_SR // 10:  # under 0.1 s: nothing a judge can hear
        return {name: "" for name in judges}
    return {name: fn(wav16) for name, fn in judges.items()}


def transcribe_batch(judges, wavs16):
    """Like ``transcribe_all`` over many clips; judges with a ``.batch`` method (Whisper) decode them together.
    Clips under 0.1 s get "" from every judge, as in ``transcribe_all``."""
    out = [{} for _ in wavs16]
    audible = [i for i, w in enumerate(wavs16) if len(w) >= JUDGE_SR // 10]
    for i in range(len(wavs16)):
        if i not in audible:
            out[i] = {name: "" for name in judges}
    for name, fn in judges.items():
        if not audible:
            break
        if hasattr(fn, "batch"):
            texts = fn.batch([wavs16[i] for i in audible])
        else:
            texts = [fn(wavs16[i]) for i in audible]
        for i, t in zip(audible, texts):
            out[i][name] = t
    return out


def make_pool(recipe, lang="fil", placement=None):
    j = recipe["judges"]
    return JudgePool(j[lang], j.get("placement") if placement is None else placement, j.get("remote"))


class JudgePool:
    """Local judges plus judges served by ``judge_server`` on a second GPU, behind one interface.

    ``submit(wavs16)`` starts the remote part in a background thread and returns a handle; ``collect(handle)``
    runs the local judges in the calling thread and merges. So a caller can submit step t's audio, do GPU work,
    and collect later: the remote judges run while the GB10 generates or trains.

    Placement comes from ``recipe.json`` ``judges.placement``. If the server is unreachable at start, or fails
    three times in a row mid-run, its judges are loaded locally and the run continues with the same judges; the
    switch is recorded in ``events``. Judges and decoding are identical either way.
    """

    def __init__(self, judge_cfg, placement=None, remote=None, device="cuda"):
        import concurrent.futures
        placement = placement or {}
        self.cfg, self.device, self.events = judge_cfg, device, []
        self.remote_names = [n for n in judge_cfg if placement.get(n) == "remote"] if remote else []
        self.remote = remote
        if self.remote_names:
            try:
                self.token = Path(remote["token_file"]).expanduser().read_text().strip()
                health = self._call("GET", "/healthz", None, timeout=10)
                missing = [n for n in self.remote_names if n not in health["judges"]]
                if missing:
                    raise RuntimeError("server does not serve %s" % missing)
                self.events.append({"event": "remote_ok", "judges": self.remote_names, "server": health})
            except Exception as exc:
                self.events.append({"event": "remote_unavailable_at_start", "error": str(exc)})
                self.remote_names = []
        local_names = [n for n in judge_cfg if n not in self.remote_names]
        self.local, self.failed = load(judge_cfg, only=local_names, device=device) if local_names else ({}, {})
        self.executor = concurrent.futures.ThreadPoolExecutor(1)
        self.strikes = 0

    @property
    def names(self):
        return sorted(list(self.local) + self.remote_names)

    def _call(self, method, path, body, timeout):
        import urllib.request
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.remote["url"].rstrip("/") + path, data=data, method=method,
                                     headers={"Authorization": "Bearer " + self.token,
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _remote(self, wavs16, names):
        from .judge_server import encode_audio
        last = None
        for attempt in range(3):
            try:
                out = self._call("POST", "/transcribe", {"judges": names, "audio_b64": [encode_audio(w) for w in wavs16]},
                                 timeout=self.remote.get("timeout_seconds", 120))
                return out["hyps"]
            except Exception as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("remote judges failed 3 times: %s" % last)

    def submit(self, wavs16):
        wavs16 = list(wavs16)
        names = list(self.remote_names)
        fut = self.executor.submit(self._remote, wavs16, names) if names else None
        return wavs16, names, fut

    def collect(self, handle):
        wavs16, names, fut = handle
        out = transcribe_batch(self.local, wavs16)
        if fut is not None:
            try:
                remote = fut.result()
            except Exception as exc:
                self.events.append({"event": "remote_failed_loading_locally", "judges": names, "error": str(exc)})
                extra, failed = load(self.cfg, only=names, device=self.device)
                self.failed.update(failed)
                self.local.update(extra)
                self.remote_names = [n for n in self.remote_names if n not in names]
                remote = transcribe_batch(extra, wavs16)
            for d, r in zip(out, remote):
                d.update(r)
        return out

    def transcribe_many(self, wavs16):
        return self.collect(self.submit(wavs16))


class Speaker:
    """ECAPA (speechbrain/spkrec-ecapa-voxceleb, pinned) embeddings and cosine similarity. speechbrain 1.1.2 (the
    bake-off's version) is installed in the MOSS environment on dwnd-01 with ``uv pip --no-deps`` (2026-11-15,
    logged in ``environment-additions.jsonl``); 3.1 ms per 3.6 s clip on the GB10."""

    def __init__(self, cfg, device="cuda"):
        from speechbrain.inference.speaker import EncoderClassifier
        src = _snapshot(cfg["model"], cfg["revision"])
        self.enc = EncoderClassifier.from_hparams(source=src, savedir=str(Path(cfg["savedir"]).expanduser()),
                                                  run_opts={"device": "cuda:0" if device == "cuda" else device})
        self.device = device

    def embed(self, wav16):
        with torch.inference_mode():
            e = self.enc.encode_batch(torch.as_tensor(wav16, dtype=torch.float32)[None].to(self.device))
        return torch.nn.functional.normalize(e.squeeze().float(), dim=0).cpu()

    def centroid(self, wavs16):
        if not wavs16:
            raise ValueError("no clips for centroid")
        return torch.nn.functional.normalize(torch.stack([self.embed(w) for w in wavs16]).mean(0), dim=0)

    @staticmethod
    def cosine(a, b):
        return float(torch.dot(a, b))
