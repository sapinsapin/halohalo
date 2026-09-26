"""Model inputs and outputs that every runtime shares, in numpy only.

The CTC model's frontend and decoder are small enough to own here, which keeps
ONNX Runtime, OpenVINO, ExecuTorch and the browser on identical preprocessing:
a mismatch here would show up as a porting loss that is not one. Whisper's
log-mel frontend differs per runtime by design (whisper.cpp and MLX carry
their own), so it stays with each runtime.
"""

import json
from pathlib import Path

import numpy as np

SR = 16000
CTC_WINDOW_S = 10            # fixed-shape targets (NPUs, Core ML, ExecuTorch)
CTC_STRIDE = 320             # wav2vec2 conv stack: one frame per 20 ms
BLANK, DELIM = 0, "|"


def ctc_config(model_dir: Path) -> dict:
    """Feature-extractor settings and the id -> character map."""
    model_dir = Path(model_dir)
    pre = json.loads((model_dir / "preprocessor_config.json").read_text())
    vocab = json.loads((model_dir / "vocab.json").read_text())
    if vocab and isinstance(next(iter(vocab.values())), dict):   # {lang: {...}} form
        vocab = next(iter(vocab.values()))
    return {"do_normalize": pre.get("do_normalize", True),
            "id2unit": {int(i): u for u, i in vocab.items()}}


def ctc_normalise(audio: np.ndarray, cfg: dict) -> np.ndarray:
    """Wav2Vec2FeatureExtractor's zero-mean, unit-variance step, on the real
    audio only. Padding comes after, so a fixed window never shifts the
    statistics."""
    x = np.asarray(audio, dtype=np.float32)
    if cfg["do_normalize"]:
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    return x.astype(np.float32)


def ctc_windows(audio: np.ndarray, cfg: dict, window_s: int | None = None):
    """-> list of (input (1, N) float32, valid frame count). With no window the
    whole clip is one input (dynamic-shape runtimes); with one, the clip is cut
    into zero-padded windows of exactly window_s seconds."""
    x = ctc_normalise(audio, cfg)
    if not window_s:
        return [(x[None, :], None)]
    n = window_s * SR
    out = []
    for s in range(0, max(len(x), 1), n):
        chunk = x[s:s + n]
        w = np.zeros(n, dtype=np.float32)
        w[:len(chunk)] = chunk
        out.append((w[None, :], max(1, int(np.ceil(len(chunk) / CTC_STRIDE)))))
    return out


def ctc_greedy(frame_ids, id2unit: dict) -> str:
    """Collapse repeats, drop blanks, '|' -> space. Same as finetune_ctc.ctc_decode."""
    out, prev = [], None
    for i in frame_ids:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(id2unit.get(i, ""))
        prev = i
    return "".join(out).replace(DELIM, " ").strip()


def ctc_transcribe(run, audio: np.ndarray, cfg: dict, window_s: int | None = None) -> str:
    """run(inputs (1, N) float32) -> logits (1, T, V). Windows are decoded as
    one frame sequence, so a word cut at a boundary is not split twice."""
    ids = []
    for x, valid in ctc_windows(audio, cfg, window_s):
        logits = np.asarray(run(x))
        frames = logits[0].argmax(-1)
        ids.extend(frames[:valid] if valid else frames)
    return ctc_greedy(ids, cfg["id2unit"])


def write_wavs(audio, out_dir: Path) -> list[Path]:
    """16-bit wavs for runtimes that read files (whisper.cpp)."""
    import wave
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, a in enumerate(audio):
        p = out_dir / f"{i:03d}.wav"
        if not p.exists():
            pcm = (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(pcm)
        paths.append(p)
    return paths
