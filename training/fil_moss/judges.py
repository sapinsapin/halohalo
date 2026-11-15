"""ASR judges, ECAPA speaker similarity and trimmed duration for Filipino MOSS evaluation and rewards (GPU).

Same three families as the Hiligaynon bake-off (``bench/hil_tts/eval/judge_asr.py`` on dwnd), switched to
Filipino: PLD Whisper (in-family), MMS-1b-all with the ``tgl`` adapter and Omnilingual CTC-1B through the same
port (both independent). Differences from the bake-off, on purpose:

* Whisper decodes greedily without the temperature fallback. Rewards need thousands of transcripts per hour
  and a fallback would make the reward stochastic in the judge as well as the policy.
* No UTMOS: it is an English naturalness proxy and no gate reads it.

A judge that fails to load is reported, not fatal: worst-judge CER is then taken over the remaining families
(recipe ``judges.fallback``), and every report names the judges that actually scored it.
"""
import sys
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
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision)


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

    def transcribe(wav16):
        feats = proc.feature_extractor(wav16, sampling_rate=JUDGE_SR, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(feats.to(device, torch.bfloat16), language=cfg["language_token"],
                                 task="transcribe", do_sample=False, num_beams=1, max_new_tokens=224)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
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


class Speaker:
    """ECAPA (speechbrain/spkrec-ecapa-voxceleb, pinned) embeddings and cosine similarity. Needs speechbrain in
    the MOSS environment: ``pip install speechbrain==1.1.2`` (the bake-off's eval environment version)."""

    def __init__(self, cfg, device="cuda"):
        from speechbrain.inference.speaker import EncoderClassifier
        src = _snapshot(cfg["model"], cfg["revision"])
        self.enc = EncoderClassifier.from_hparams(source=src, savedir=str(Path(cfg["savedir"]).expanduser()),
                                                  run_opts={"device": device})
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
