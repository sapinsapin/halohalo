"""GPU runtime for MOSS-TTS-Realtime on dwnd-01: model loading, token log-probabilities, generation, codec.

Reuses the pinned upstream code and weights of the Hiligaynon pilot (``$MOSS_PILOT_ROOT``) as a read-only
dependency; nothing here writes into that directory. Run inside its ``.venv``.

The stage 2 objective needs per-token log-probabilities of sampled audio codes. Upstream exposes them only
through the local transformer's logits during teacher forcing, so ``token_logprobs`` rebuilds a rollout as an
SFT record, runs the same forward the training loss uses, and gathers the label log-probabilities. Stage 1's
parity check proves this equals upstream's own loss before anything is trained with it.
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("MOSS_PILOT_ROOT", str(Path.home() / "work/hil-moss-pilot"))).expanduser()
MODEL_DIR = ROOT / "models/MOSS-TTS-Realtime"
CODEC_DIR = ROOT / "models/MOSS-Audio-Tokenizer"
for p in (ROOT / "upstream", ROOT / "upstream/moss_tts_realtime"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from moss_tts_realtime.finetuning.sft import build_processor, configure_torch_backends  # noqa: E402
from moss_tts_realtime.finetuning.dataset import MossTTSRealtimeSFTDataset  # noqa: E402
from moss_tts_realtime.mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime  # noqa: E402
from moss_tts_realtime.mossttsrealtime.configuration_mossttsrealtime import MossTTSRealtimeConfig  # noqa: E402

AUDIO_EOS = 1026
SAMPLE_RATE = 24000


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def new_output_dir(path):
    path = Path(path)
    if path.exists():
        raise SystemExit("refusing to overwrite existing output %s" % path)
    path.mkdir(parents=True)
    return path


def setup():
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
    configure_torch_backends()


def processor():
    return build_processor(str(MODEL_DIR))


def dataset(records, proc):
    return MossTTSRealtimeSFTDataset(records, proc, n_vq=16)


def batch(ds, record, device="cuda"):
    """Collate one record (not necessarily in ``ds``) into model inputs."""
    return {k: v.to(device) for k, v in ds.collate_fn([ds._pack_record(record)]).items()}


# --------------------------------------------------------------------------------------------- model loading

def _weight_map(path):
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        return {k: path / v for k, v in json.loads(index.read_text())["weight_map"].items()}
    single = path / "model.safetensors"
    if single.exists():
        from safetensors import safe_open
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return {k: single for k in f.keys()}
    raise FileNotFoundError("no safetensors weights under %s" % path)


def verify_loaded(model, path, dtype, names=("language_model.layers.0.self_attn.q_proj.weight",
                                             "local_transformer.model.layers.0.self_attn.q_proj.weight")):
    """Compare loaded parameter slices with the files after casting to the requested ``dtype``. The pilot found
    that the nested language config silently loads FP32 checkpoints as BF16; this catches that and any other
    silent cast or mismatch."""
    from safetensors import safe_open
    wmap = _weight_map(path)
    params = dict(model.named_parameters())
    for name in names:
        with safe_open(str(wmap[name]), framework="pt", device="cpu") as f:
            expected = f.get_slice(name)[:16]
        actual = params[name].detach()[:16].cpu()
        if actual.dtype != dtype:
            raise AssertionError("%s loaded as %s, requested %s" % (name, actual.dtype, dtype))
        if not torch.equal(expected.to(dtype), actual):
            raise AssertionError("loaded weights differ from %s for %s" % (path, name))


def load_model(path=MODEL_DIR, dtype=torch.float32, train=True, device="cuda"):
    """Load MOSS-TTS-Realtime with every nested config forced to ``dtype`` and verify against the files."""
    config = MossTTSRealtimeConfig.from_pretrained(path)
    config.dtype = dtype
    config.language_config.dtype = dtype
    config.local_config.dtype = dtype
    model = MossTTSRealtime.from_pretrained(path, config=config, torch_dtype=dtype, attn_implementation="sdpa")
    model = model.to(device=device, dtype=dtype)
    verify_loaded(model, path, dtype)
    model.train(train)
    if not train:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def freeze(model, names):
    params = dict(model.named_parameters())
    for name in names:
        params[name + ".weight" if name + ".weight" in params else name].requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not all(p.dtype == torch.float32 for p in trainable):
        raise AssertionError("trainable parameters must be FP32")
    return trainable


def adamw8bit(params, lr, cfg):
    import bitsandbytes as bnb
    return bnb.optim.AdamW8bit(params, lr=lr, betas=tuple(cfg["betas"]), eps=cfg["eps"],
                               weight_decay=cfg["weight_decay"])


def save(model, proc, path):
    model.save_pretrained(path, max_shard_size="5GB")
    proc.tokenizer.save_pretrained(path)


# ----------------------------------------------------------------------------------------- log-probabilities

def token_logprobs(model, inputs):
    """Per-token log-probabilities of the audio labels, from the same forward the training loss uses.

    Returns (logp [N, 16], valid [N, 16] bool, upstream loss). ``-logp[valid].mean()`` equals upstream's loss
    (stage 1 parity check). Runs under the caller's autocast; logits are upcast to FP32 as upstream's loss does.
    """
    out = model(**inputs)
    audio_labels = inputs["labels"][:, :, 1:]
    train_mask = ~(audio_labels == -100).all(dim=-1)
    labels = audio_labels[train_mask]
    logits = out.local_logits.float()
    if logits.shape[:2] != labels.shape:
        raise AssertionError("local logits %s do not match labels %s" % (tuple(logits.shape), tuple(labels.shape)))
    valid = labels != -100
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return logp, valid, out.loss


def wrapper_loss(model, inputs):
    logp, valid, loss = token_logprobs(model, inputs)
    return -logp[valid].mean(), loss


def drop_eos_label(inputs):
    """For a rollout that hit the frame cap: the SFT packer appends an EOS label the policy never produced.
    Remove it so the update does not credit a stop that did not happen."""
    labels = inputs["labels"].clone()
    labels[..., 1][labels[..., 1] == AUDIO_EOS] = -100
    inputs = dict(inputs)
    inputs["labels"] = labels
    return inputs


def record(text, codes, ref_codes=None):
    rec = {"conversations": [{"role": "assistant", "text": text, "audio_codes": np.asarray(codes).tolist()}]}
    if ref_codes is not None:
        rec["ref_audio_codes"] = np.asarray(ref_codes).tolist()
    return rec


# ------------------------------------------------------------------------------------------------ generation

def load_codec(device="cuda"):
    from transformers import AutoModel
    return AutoModel.from_pretrained(CODEC_DIR, trust_remote_code=True).eval().to(device)


def encode_wav(codec, wav, n_vq=16, device="cuda"):
    """24 kHz mono float waveform -> codes [T, n_vq] (same call as the pilot's encode scripts)."""
    with torch.inference_mode():
        enc = codec.encode(torch.as_tensor(wav, dtype=torch.float32)[None, None, :].to(device), num_quantizers=n_vq)
    codes = enc["audio_codes"][:, 0, :int(enc["audio_codes_lengths"][0])].T.cpu().numpy()
    if codes.shape[1] != n_vq:
        raise AssertionError("expected %d codebooks, got %s" % (n_vq, codes.shape))
    return codes


def decode(codec, codes, device="cuda"):
    if len(codes) == 0:
        return np.zeros(0, dtype=np.float32)
    with torch.inference_mode():
        audio = codec.decode(torch.as_tensor(np.asarray(codes), device=device).T, chunk_duration=8)["audio"][0]
    return audio.detach().float().cpu().numpy().reshape(-1)


def engine(model, codec, max_length=1000):
    from transformers import AutoTokenizer
    from inferencer import MossTTSRealtimeInference
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    return MossTTSRealtimeInference(model, tok, max_length=max_length, codec=codec, codec_sample_rate=SAMPLE_RATE)


def max_frames(seconds, gen_cfg):
    return max(gen_cfg["min_max_frames"],
               int(math.ceil(seconds * gen_cfg["frames_per_second"] * gen_cfg["max_frames_per_reference_second"])))


def generate(eng, texts, ref_codes, frame_cap, gen_cfg, do_sample=True, device="cuda"):
    """Generate codes for each text with its reference-voice codes (or None). Returns [(codes [T,16],
    truncated)], truncated meaning no EOS before ``frame_cap``. Uses the upstream inferencer's own prefill and
    sampling, with reference codes passed directly instead of re-encoding a wav file."""
    ids = [eng.processor.make_ensemble(None if r is None else np.asarray(r)) for r in ref_codes]
    text_ids = [eng.tokenizer(t)["input_ids"] for t in texts]
    out = eng._generate_from_ids(
        input_ids=ids, text_ids=text_ids, text_lengths=[len(t) for t in text_ids], max_length=int(frame_cap),
        temperature=float(gen_cfg["temperature"]), top_p=float(gen_cfg["top_p"]), top_k=int(gen_cfg["top_k"]),
        do_sample=bool(do_sample), repetition_penalty=float(gen_cfg["repetition_penalty"] if do_sample else 1.0),
        repetition_window=gen_cfg["repetition_window"], device=torch.device(device))
    return [(c, len(c) >= frame_cap) for c in out]
