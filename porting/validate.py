"""Check a ported artefact against the model it came from.

  python -m porting.validate sapinsapin/whisper-small-pld-ceb --runtime torch
  python -m porting.validate sapinsapin/whisper-small-pld-ceb --runtime ort --variant int8

Every runtime transcribes the same evalpack clips (frozen test split, never
seen in training or calibration) and is scored two ways:

  accuracy   CER/WER against the human transcript, normalised as the -norm
             models are scored. Tells a user what the port is worth.
  parity     CER against the PyTorch fp32 transcripts of the same model
             (`--runtime torch`, run first). Isolates what porting lost:
             0 means the port behaves exactly like the original.

plus artefact size and real-time factor on this CPU at a fixed thread count
(PORT_THREADS, default 4, roughly a phone's big cores). The RTF is a relative
measure between variants, not a prediction for an Arm device.

Runtime -> venv: torch, ort -> venv_port_onnx; openvino -> venv_port_ov;
executorch -> venv_port_et; mlx -> venv_port_mlx; whispercpp -> any (calls the
built binary); transformersjs -> any (calls node). Results go to
$FINETUNE_DIR/port/results/<model>/<runtime>-<variant>-<lang>.json.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from porting import frontends
from porting.evalpack import load
from porting.metrics import agreement, score

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
PORT = FINETUNE_DIR / "port"
ART = PORT / "artefacts"
RESULTS = PORT / "results"
THREADS = int(os.environ.get("PORT_THREADS", "4"))
ROOT = Path(__file__).resolve().parent.parent
TOOLS = Path(os.environ.get("PORT_TOOLS", "/mnt/d/halohalo/third_party"))


def family_of(repo):
    return "whisper" if "whisper" in repo else "wav2vec2-ctc"


def whisper_language(model_dir: Path) -> str:
    """The language slot the fine-tune was trained with: "english" for eng,
    "tagalog" for every other Philippine language (finetune_asr.py)."""
    gc = model_dir / "generation_config.json"
    if gc.exists():
        lang = json.loads(gc.read_text()).get("language")
        if lang:
            return lang.strip("<|>")
    return "tagalog"


_T = {}


def loaded():
    """Runners call this once the model is loaded: RTF then counts
    transcription only, not reading weights off the disk."""
    _T["loaded"] = time.perf_counter()


ISO1 = {"tagalog": "tl", "english": "en", "tl": "tl", "en": "en"}


# ---------------------------------------------------------------- runtimes
# Each returns a list of hypotheses, one per clip, in order.

def run_torch(repo, root, variant, audio, fam):
    """The reference: the published checkpoint, fp32, PyTorch on CPU."""
    import torch
    torch.set_num_threads(THREADS)
    tok = os.environ.get("HF_TOKEN")
    if fam == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from porting.hfcompat import config_dir
        proc = WhisperProcessor.from_pretrained(config_dir(repo))
        model = WhisperForConditionalGeneration.from_pretrained(repo, token=tok).eval()
        lang = model.generation_config.language or "tagalog"
        loaded()
        hyps = []
        with torch.inference_mode():
            for a in audio:
                f = proc(a, sampling_rate=16000, return_tensors="pt").input_features
                ids = model.generate(f, language=lang, task="transcribe", num_beams=1,
                                     max_new_tokens=225)
                hyps.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
        return hyps
    from huggingface_hub import snapshot_download
    from transformers import Wav2Vec2ForCTC
    d = Path(snapshot_download(repo, token=tok, allow_patterns=["*.json"]))
    cfg = frontends.ctc_config(d)
    model = Wav2Vec2ForCTC.from_pretrained(repo, token=tok).eval()
    loaded()
    with torch.inference_mode():
        run = lambda x: model(torch.from_numpy(x)).logits.numpy()
        return [frontends.ctc_transcribe(run, a, cfg) for a in audio]


ORT_FILES = {"fp32": "", "int8": "_quantized", "fp16": "_fp16"}
NPU_FILES = {"qdq": "_qdq_int8", "qdq16": "_qdq_a16w8"}   # A8W8, A16W8


SHM = Path('/dev/shm')


def in_ram(files, tag):
    """Copy what a runtime is about to load into RAM (/dev/shm) and return
    the folder. Runtimes memory-map weights, and over WSL's 9P bridge to D:
    every page fault is a round trip: a 1 GB program took 20+ minutes to
    load, an fp32 ONNX Whisper 220 s. One sequential copy takes seconds.
    files: [(source, name in the folder)].

    Without /dev/shm (macOS, or any machine whose disk is local) the folder
    is a temp dir of symlinks: nothing to gain from copying off an SSD."""
    if SHM.is_dir():
        d = SHM / f'port-{tag}'
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for src, name in files:
            if Path(src).is_dir():
                shutil.copytree(src, d / name, ignore=shutil.ignore_patterns('model.safetensors', 'README.md'))
            else:
                shutil.copy(src, d / name)
    else:
        import tempfile
        d = Path(tempfile.mkdtemp(prefix=f'port-{tag}-'))
        for src, name in files:
            (d / name).symlink_to(Path(src).resolve())
    _T.setdefault('ram_dirs', []).append(d)
    return d


MAC = sys.platform == "darwin"


def with_side_files(graph: Path):
    """An ONNX graph plus its external-data file, if it has one."""
    out = [(graph, graph.name)]
    for side in (graph.name + '_data', graph.name + '.data'):
        if (graph.parent / side).exists():
            out.append((graph.parent / side, side))
    return out


def _ort_session(path, providers=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])


def _stage_whisper(root, variant) -> Path:
    """The layout optimum's ORT Whisper loader expects, in RAM (see in_ram)."""
    web = root / 'onnx-web'
    sfx = ORT_FILES.get(variant, '')
    npu = NPU_FILES.get(variant)
    enc = (root / 'npu' / f'encoder_model{npu}.onnx' if npu
           else web / 'onnx' / f'encoder_model{sfx}.onnx')
    dec = web / 'onnx' / f"decoder_model_merged{'' if npu else sfx}.onnx"
    files = [(f, f.name) for f in web.glob('*.json')]
    files += [(enc, 'encoder_model.onnx'), (dec, 'decoder_model_merged.onnx')]
    return in_ram(files, f'{root.name}-{variant}')


def run_ort(repo, root, variant, audio, fam):
    """ONNX Runtime, CPU execution provider: the graphs Arm CPUs, browsers
    (wasm) and the NPU execution providers load. `qdq` runs the NPU graph on
    the CPU, which executes the same int8 arithmetic the NPU would."""
    if fam == "whisper":
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperProcessor
        stage = _stage_whisper(root, variant)
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        model = ORTModelForSpeechSeq2Seq.from_pretrained(stage, use_cache=True,
                                                         session_options=so)
        proc = WhisperProcessor.from_pretrained(stage)
        lang = whisper_language(stage)
        loaded()
        hyps = []
        for a in audio:
            f = proc(a, sampling_rate=16000, return_tensors="pt").input_features
            ids = model.generate(input_features=f, language=lang, task="transcribe",
                                 num_beams=1, max_new_tokens=225)
            hyps.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
        return hyps
    web = root / "onnx-web"
    cfg = frontends.ctc_config(web)
    if variant in NPU_FILES:
        g, window = root / "npu" / f"model{NPU_FILES[variant]}.onnx", frontends.CTC_WINDOW_S
    else:
        g, window = web / "onnx" / f"model{ORT_FILES[variant]}.onnx", None
    sess = _ort_session(in_ram(with_side_files(g), f"{root.name}-{variant}") / g.name)
    name = sess.get_inputs()[0].name
    run = lambda x: sess.run(None, {name: x})[0]
    loaded()
    return [frontends.ctc_transcribe(run, a, cfg, window) for a in audio]


def run_openvino(repo, root, variant, audio, fam):
    """OpenVINO on the CPU plugin, from the optimum-intel export (stateful
    decoder, NNCF int8 weights). The same IR loads on Intel NPUs and iGPUs."""
    d = in_ram([(root / "openvino" / variant, "ir")], f"{root.name}-ov-{variant}") / "ir"
    if fam == "whisper":
        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import WhisperProcessor
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            d, ov_config={"INFERENCE_NUM_THREADS": THREADS, "INFERENCE_PRECISION_HINT": "f32"})
        proc = WhisperProcessor.from_pretrained(d)
        lang = whisper_language(d)
        loaded()
        hyps = []
        for a in audio:
            f = proc(a, sampling_rate=16000, return_tensors="pt").input_features
            ids = model.generate(input_features=f, language=lang, task="transcribe",
                                 num_beams=1, max_new_tokens=225)
            hyps.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
        return hyps
    import openvino as ov
    core = ov.Core()
    compiled = core.compile_model(str(d / "openvino_model.xml"), "CPU",
                                  {"INFERENCE_NUM_THREADS": THREADS,
                                   "INFERENCE_PRECISION_HINT": "f32"})
    cfg = frontends.ctc_config(d)
    run = lambda x: compiled(x)[0]
    loaded()
    return [frontends.ctc_transcribe(run, a, cfg) for a in audio]


def run_executorch(repo, root, variant, audio, fam):
    """ExecuTorch's Python runtime on the XNNPACK-lowered program: the kernels
    the Android/iOS runtime uses."""
    from porting.export_executorch import run_program
    src = root / "executorch" / variant
    # ExecuTorch memory-maps the program; over WSL's 9P bridge every page
    # fault is a round trip, and a 1 GB .pte took 20+ minutes to "load". One
    # sequential copy into RAM (/dev/shm) first.
    stage = in_ram([(src, "pte")], f"{root.name}-et-{variant}") / "pte"
    return run_program(stage, audio, fam, THREADS, on_loaded=loaded)


def run_mlx(repo, root, variant, audio, fam):
    """mlx-whisper. On a Mac: Metal, fp16 activations, the real thing. On
    Linux: MLX's CPU backend in fp32 activations (x86 has no native fp16 and
    MLX's CPU kernels for it are slow), a parity check of weights and graph
    only. Quantised weights stay quantised either way."""
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder
    d = in_ram([(root / "mlx" / variant, "mlx")], f"{root.name}-mlx-{variant}") / "mlx"
    lang = ISO1.get(json.loads((d / "halohalo.json").read_text())["language"], "tl")
    ModelHolder.get_model(str(d), mx.float16 if MAC else mx.float32)   # load without decoding
    loaded()
    hyps = []
    for a in audio:
        r = mlx_whisper.transcribe(np.asarray(a, dtype=np.float32), path_or_hf_repo=str(d),
                                   language=lang, task="transcribe", temperature=0.0,
                                   condition_on_previous_text=False, without_timestamps=True,
                                   fp16=MAC, verbose=None)
        hyps.append(r["text"].strip())
    return hyps


COREML_UNITS = {"all": "ALL", "ane": "CPU_AND_NE", "gpu": "CPU_AND_GPU", "cpu": "CPU_ONLY"}


def run_coreml(repo, root, variant, audio, fam):
    """Core ML, macOS only. The variant picks the compute units: `all` lets
    Core ML choose (Neural Engine where it can), `ane` asks for the Neural
    Engine, `gpu` for the GPU, `cpu` for the CPU alone.

    Whisper: the encoder runs in Core ML — the split WhisperKit and
    whisper.cpp use — and its output feeds transformers' own decoder, so the
    decode loop is exactly the reference's. CTC: the whole model, in the
    fixed 10 s windows it was converted at."""
    import coremltools as ct
    cu = getattr(ct.ComputeUnit, COREML_UNITS[variant])
    if fam == "whisper":
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.modeling_outputs import BaseModelOutput
        from porting.hfcompat import config_dir
        enc = ct.models.MLModel(str(root / "coreml" / "encoder.mlpackage"), compute_units=cu)
        proc = WhisperProcessor.from_pretrained(config_dir(repo))
        model = WhisperForConditionalGeneration.from_pretrained(repo).eval()
        lang = model.generation_config.language or "tagalog"
        loaded()
        hyps = []
        with torch.inference_mode():
            for a in audio:
                f = proc(a, sampling_rate=16000, return_tensors="pt").input_features
                h = enc.predict({"logmel_data": f.numpy().astype(np.float32)})["output"]
                out = BaseModelOutput(last_hidden_state=torch.from_numpy(np.asarray(h, dtype=np.float32)))
                ids = model.generate(input_features=f, encoder_outputs=out, language=lang,
                                     task="transcribe", num_beams=1, max_new_tokens=225)
                hyps.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
        return hyps
    from porting.hfcompat import config_dir
    m = ct.models.MLModel(str(root / "coreml" / "model.mlpackage"), compute_units=cu)
    cfg = frontends.ctc_config(config_dir(repo))
    run = lambda x: m.predict({"input_values": x})["logits"]
    loaded()
    return [frontends.ctc_transcribe(run, a, cfg, frontends.CTC_WINDOW_S) for a in audio]


def run_whispercpp(repo, root, variant, audio, fam):
    """whisper.cpp's CLI, greedy, no temperature fallback — the decoding the
    reference uses. On Arm it takes the NEON/KleidiAI paths; here, AVX2."""
    exe = TOOLS / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    model = root / "ggml" / f"ggml-model-{variant}.bin"
    lang = ISO1.get(json.loads((root / "ggml" / "halohalo.json").read_text())["language"], "tl")
    wavs = frontends.write_wavs(audio, PORT / "evalpack" / "wav" / _lang_of_audio(audio))
    for w in wavs:                                  # stale outputs from another variant
        Path(str(w) + ".txt").unlink(missing_ok=True)
    # one process for all clips: the model loads once, as in an app
    args = [str(exe), "-m", str(model), "-l", lang, "-t", str(THREADS), "-bs", "1", "-bo", "1",
            "-nf", "-nt", "-np", "-otxt"]
    for w in wavs:
        args += ["-f", str(w)]
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    import re
    ms = {k: float(v) for k, v in re.findall(r"(load|total) time =\s*([\d.]+) ms", out.stderr)}
    if "total" in ms:
        _T["run_seconds"] = (ms["total"] - ms.get("load", 0)) / 1000
        _T["load_seconds"] = ms.get("load", 0) / 1000
    return [" ".join(Path(str(w) + ".txt").read_text().split()) for w in wavs]


def run_transformersjs(repo, root, variant, audio, fam):
    """Node: Transformers.js (Whisper) or onnxruntime-node with the page's own
    CTC decoder (wav2vec2) over the onnx-web folder, exactly as a page would
    load it. Node's CPU backend shares its kernels with ORT; WebGPU needs a
    browser (porting/web/index.html)."""
    web_dir = ROOT / "porting" / "web"
    job = PORT / "tmp" / f"tjs-{root.name}-{variant}"
    job.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate([np.asarray(a, dtype=np.float32) for a in audio])
    flat.tofile(job / "audio.f32")
    spec = {"model_dir": str(root / "onnx-web"), "family": fam, "dtype": variant,
            "lengths": [int(len(a)) for a in audio], "audio": str(job / "audio.f32"),
            "language": whisper_language(root / "onnx-web") if fam == "whisper" else None,
            "threads": THREADS}
    (job / "job.json").write_text(json.dumps(spec))
    out = subprocess.run(["node", str(web_dir / "validate.mjs"), str(job / "job.json")],
                         capture_output=True, text=True, cwd=web_dir)
    if out.returncode:
        raise RuntimeError(out.stderr[-3000:])
    r = json.loads(out.stdout.strip().splitlines()[-1])
    _T["run_seconds"], _T["load_seconds"] = r["run_ms"] / 1000, r["load_ms"] / 1000
    return r["hyps"]


RUNTIMES = {"torch": run_torch, "ort": run_ort, "openvino": run_openvino,
            "executorch": run_executorch, "mlx": run_mlx, "whispercpp": run_whispercpp,
            "transformersjs": run_transformersjs, "coreml": run_coreml}

_AUDIO_LANG = {}


def _lang_of_audio(audio):
    return _AUDIO_LANG.get(id(audio), "x")


# ---------------------------------------------------------------- sizes

def artefact_size_mb(root: Path, runtime: str, variant: str) -> float | None:
    """What a device downloads for this runtime/variant."""
    web = root / "onnx-web" / "onnx"
    if runtime == "ort" or runtime == "transformersjs":
        if variant in NPU_FILES:
            files = list((root / "npu").glob(f"*{NPU_FILES[variant]}.onnx*"))
            files += [p for p in web.glob("decoder_model_merged.onnx*")]
        else:
            sfx = ORT_FILES.get(variant, "")
            files = [p for p in web.glob("*") if p.name.split(".onnx")[0].endswith(sfx)
                     and (sfx or not p.name.split(".onnx")[0].endswith(("_quantized", "_fp16", "_q4f16")))]
    elif runtime in ("openvino", "executorch", "mlx"):
        # only what the runtime loads: exporters leave a copy of the HF weights beside it
        keep = {"openvino": (".xml", ".bin"), "executorch": (".pte",), "mlx": (".safetensors",)}[runtime]
        files = [p for p in (root / runtime / variant).rglob("*") if p.is_file() and p.suffix in keep]
    elif runtime == "coreml":
        files = [p for p in (root / "coreml").rglob("*") if p.is_file() and ".mlpackage" in str(p)]
    elif runtime == "whispercpp":
        files = [root / "ggml" / f"ggml-model-{variant}.bin"]
    else:
        return None
    files = [f for f in files if f.exists() and not f.suffix == ".json"]
    return round(sum(f.stat().st_size for f in files) / 2**20, 1) if files else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--runtime", required=True, choices=sorted(RUNTIMES))
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--lang", default=None, help="evalpack language (default: from the repo name)")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    name = args.repo.split("/")[-1]
    fam = family_of(args.repo)
    lang = args.lang or next(l for l in ("bcl", "ceb", "eng", "fil", "hil", "ilo", "pag",
                                         "pam", "tsg", "war") if f"_{l}" in name or f"-{l}" in name)
    root = ART / name
    audio, refs = load(lang)
    audio, refs = audio[:args.n], refs[:args.n]
    _AUDIO_LANG[id(audio)] = lang
    secs = sum(len(a) for a in audio) / 16000

    _T.clear()
    t0 = time.perf_counter()
    try:
        hyps = RUNTIMES[args.runtime](args.repo, root, args.variant, audio, fam)
    finally:
        for d in _T.pop('ram_dirs', []):
            shutil.rmtree(d, ignore_errors=True)
    t1 = time.perf_counter()
    if "run_seconds" in _T:
        wall, load_s = _T["run_seconds"], _T.get("load_seconds", 0.0)
    else:
        tl = _T.get("loaded", t0)
        wall, load_s = t1 - tl, tl - t0

    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("PORT_HOST")          # e.g. "mac": results beside, not over, these
    label = f"{args.runtime}@{host}" if host else args.runtime
    res = {"repo": args.repo, "family": fam, "runtime": label, "variant": args.variant,
           "lang": lang, "clips": len(audio), "audio_seconds": round(secs, 1),
           "threads": THREADS, "load_seconds": round(load_s, 1), "wall_seconds": round(wall, 1), "rtf": round(wall / secs, 3),
           "size_mb": artefact_size_mb(root, args.runtime, args.variant),
           "accuracy": score(refs, hyps), "hyps": hyps}
    ref_file = out / f"torch-fp32-{lang}.json"
    if args.runtime != "torch" and ref_file.exists():
        ref = json.loads(ref_file.read_text())
        res["parity"] = agreement(ref["hyps"][:len(hyps)], hyps)
    (out / f"{label}-{args.variant}-{lang}.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    acc = res["accuracy"]
    par = res.get("parity", {})
    print(f"{name} {label}/{args.variant} [{lang}] CER {acc['cer']:.4f} WER {acc['wer']:.4f}"
          f" | parity CER {par.get('cer_vs_reference', float('nan')):.4f}"
          f" identical {par.get('identical', float('nan'))}"
          f" | RTF {res['rtf']} size {res['size_mb']} MB")


if __name__ == "__main__":
    main()
