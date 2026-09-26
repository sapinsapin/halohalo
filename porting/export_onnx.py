"""ONNX: the hub format. One export feeds four targets — Arm CPUs (ONNX
Runtime), the browser (Transformers.js / onnxruntime-web), Qualcomm and AMD
NPUs (QDQ int8) and AMD GPUs (MIGraphX EP). Intel goes through
porting.export_openvino instead, whose exporter writes a stateful decoder.

Runs in venv_port_onnx.

  python -m porting.export_onnx sapinsapin/whisper-small-pld-ceb
  python -m porting.export_onnx sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm

Writes $FINETUNE_DIR/port/artefacts/<name>/onnx-web/ in the Transformers.js
layout (config, tokenizer and preprocessor at the root, graphs under onnx/):

  onnx/<graph>.onnx              fp32 — ORT CPU/CUDA/MIGraphX, the reference
  onnx/<graph>_quantized.onnx    int8 dynamic — Arm CPUs, wasm in browsers
  onnx/<graph>_fp16.onnx         fp16 — WebGPU, GPUs

plus npu/ (static QDQ int8, fixed shapes: QNN and Ryzen AI).
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"

CTC_WINDOW_S = 10          # NPUs want static shapes; CTC is chunked at 10 s
SR = 16000


def family_of(repo: str) -> str:
    return "whisper" if "whisper" in repo else "wav2vec2-ctc"


def export(repo: str, out: Path) -> list[Path]:
    """fp32 export with optimum-onnx. Whisper gets the merged decoder with KV
    cache, which is what Transformers.js and onnxruntime-web load."""
    from optimum.exporters.onnx import main_export
    task = ("automatic-speech-recognition-with-past" if family_of(repo) == "whisper"
            else "automatic-speech-recognition")
    tmp = out / "_export"
    main_export(repo, output=tmp, task=task, opset=18, do_validation=False,
                token=os.environ.get("HF_TOKEN"))
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    graphs = []
    for f in sorted(tmp.iterdir()):
        if f.suffix == ".onnx" or f.name.endswith(".onnx_data"):
            shutil.move(str(f), out / "onnx" / f.name)
            if f.suffix == ".onnx":
                graphs.append(out / "onnx" / f.name)
        else:
            shutil.move(str(f), out / f.name)       # config, tokenizer, preprocessor
    shutil.rmtree(tmp)
    fetch_side_files(repo, out)
    return graphs


def fetch_side_files(repo: str, out: Path):
    """The exporter re-saves tokenizer and preprocessor through transformers,
    which can drop a file the repo ships (a CTC repo with only vocab.json).
    Copy anything small it left out, so every runtime reads the repo's own."""
    from porting.hfcompat import config_dir, sanitize_tokenizer_config
    src = config_dir(repo)            # sanitised; has preprocessor_config.json
    for f in src.iterdir():
        if not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    sanitize_tokenizer_config(out / "tokenizer_config.json")


def quantize_int8(g: Path) -> Path:
    """Dynamic int8: weights int8 now, activations quantised at run time. No
    calibration, portable to every CPU ORT build, and the default Transformers.js
    loads as `q8`."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = g.with_name(g.stem + "_quantized.onnx")
    # ORT's default op set minus Conv. Convolutions stay fp32 — wav2vec2's
    # positional conv computes its weight at run time (weight norm), which
    # dynamic quantisation cannot take ("Expected ... to be an initializer").
    # A narrower list costs size: without Gather and Transpose, Whisper's tied
    # 51,865-token embedding stays fp32 (int8 decoder 314 MB instead of 148).
    from onnxruntime.quantization.registry import IntegerOpsRegistry
    quantize_dynamic(str(g), str(q), weight_type=QuantType.QInt8,
                     op_types_to_quantize=[o for o in IntegerOpsRegistry if o != "Conv"],
                     use_external_data_format=big(g),
                     # EnableSubgraph: Whisper's merged decoder keeps both
                     # branches (with and without KV cache) inside an If node;
                     # without it the decoder is left fp32 and the "int8" file
                     # is as big as the original.
                     extra_options={"MatMulConstBOnly": True, "EnableSubgraph": True})
    return q


def fresh(data_file: Path) -> None:
    """ONNX appends external tensors to an existing side file instead of
    replacing it, so a file left by an interrupted run grows the new one by
    its whole size (seen: 11.5 GB for a 3.85 GB model). Remove it first."""
    data_file.unlink(missing_ok=True)


def big(g: Path) -> bool:
    """Over protobuf's 2 GB limit: weights live beside the graph."""
    return g.with_name(g.name + "_data").exists() or g.with_suffix(".onnx_data").exists()


def to_fp16(g: Path) -> Path:
    """fp16 weights and activations with fp32 at the graph boundary, so callers
    feed and read float32 either way. What WebGPU and GPUs want."""
    import onnx
    from onnxruntime.transformers.onnx_model import OnnxModel
    block = ["LayerNormalization", "Softmax"]
    if big(g):
        # in-memory shape inference serialises the model, which protobuf
        # refuses past 2 GB; inference from the file path handles external data
        from onnxruntime.transformers.float16 import convert_float_to_float16
        shapes = g.with_name(g.stem + "_shapes.onnx")
        onnx.shape_inference.infer_shapes_path(str(g), str(shapes))
        m = convert_float_to_float16(onnx.load(str(shapes)), keep_io_types=True,
                                     op_block_list=block, disable_shape_infer=True)
        shapes.unlink()
        om = OnnxModel(m)
    else:
        om = OnnxModel(onnx.load(str(g)))
        om.convert_float_to_float16(keep_io_types=True, op_block_list=block)
    # the pass appends its Cast nodes at the end; ONNX Runtime sorts on load,
    # but the ONNX checker (and optimum's merge_decoders) require order
    om.topological_sort()
    m = om.model
    h = g.with_name(g.stem + "_fp16.onnx")
    if big(g):
        fresh(h.with_name(h.name + "_data"))
        onnx.save(m, str(h), save_as_external_data=True, all_tensors_to_one_file=True,
                  location=h.name + "_data")
    else:
        onnx.save(m, str(h))
    return h


def _npu_prepare(g: Path, family: str, fixed: Path, pre: Path, ext: bool):
    """Fix every dynamic dimension to the NPU's shape, then ORT's
    quantisation pre-processing (constant folding, which also turns wav2vec2's
    weight-norm convolution into a plain initializer)."""
    import onnx
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
    m = onnx.load(str(g))
    for inp in m.graph.input:
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_param:
                size = 1 if "batch" in d.dim_param else (
                    CTC_WINDOW_S * SR if family == "wav2vec2-ctc" else 3000)
                make_dim_param_fixed(m.graph, d.dim_param, size)
    for stale in (fixed, pre):
        fresh(stale.with_name(stale.name + "_data"))
    onnx.save(m, str(fixed), save_as_external_data=ext, all_tensors_to_one_file=True,
              location=fixed.name + "_data")
    del m
    quant_pre_process(str(fixed), str(pre), skip_symbolic_shape=True,
                      save_as_external_data=ext, all_tensors_to_one_file=True,
                      external_data_location=pre.name + "_data")


def npu_qdq(g: Path, family: str, out: Path, calib) -> Path | None:
    """Static QDQ int8 at a fixed shape: the form Qualcomm (ORT QNN EP) and AMD
    Ryzen AI (Vitis AI EP) NPUs execute. Activations are calibrated on training
    clips, never test ones. Only the graph an NPU should run: the whole CTC
    model, or Whisper's encoder (the decoder loop stays on the CPU)."""
    import multiprocessing

    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)

    fixed = out / "npu" / (g.stem + "_fixed.onnx")
    fixed.parent.mkdir(parents=True, exist_ok=True)
    ext = big(g)
    pre = fixed.with_name(fixed.stem + "_pre.onnx")
    # In a child process: loading, fixing and optimising a 1B graph leaves
    # ~12 GB resident that Python never hands back, and calibration on top of
    # it was OOM-killed at 22 GB. The child's memory goes when it exits.
    p = multiprocessing.get_context("spawn").Process(target=_npu_prepare, args=(g, family, fixed, pre, ext))
    p.start()
    p.join()
    if p.exitcode:
        raise RuntimeError(f"NPU graph preparation failed (exit {p.exitcode})")
    m = onnx.load(str(pre), load_external_data=False)

    name = m.graph.input[0].name
    feeds = [{name: x} for x in calib]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(feeds)

        def get_next(self):
            return next(self.it, None)

    q = out / "npu" / (g.stem + "_qdq_int8.onnx")
    for side in (q.name + ".data", q.name + "_data"):
        fresh(q.with_name(side))
    # Fold activation ranges in every few clips: the calibrator otherwise holds
    # every clip's outputs until the end (~200 MB a clip for a Whisper encoder
    # at 30 s, measured). ORT 1.30's own CalibMaxIntermediateOutputs drops the
    # batches instead of folding them ("No data is collected"), hence the patch.
    _patch_minmax_fold()
    quantize_static(str(pre), str(q), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, use_external_data_format=ext,
                    extra_options={"ActivationSymmetric": False, "CalibMaxIntermediateOutputs": 4})
    for tmp in (fixed, pre):
        tmp.unlink(missing_ok=True)
        tmp.with_name(tmp.name + "_data").unlink(missing_ok=True)
    return q


def _patch_minmax_fold():
    """MinMaxCalibrater.collect_data that computes (and merges) ranges before
    clearing each batch of outputs, so memory is bounded by the batch."""
    from onnxruntime.quantization import calibrate as C

    def collect_data(self, data_reader):
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            outs = self.infer_session.run(None, inputs)
            self.intermediate_outputs.append(
                [v if o.name not in self.model_original_outputs else None
                 for o, v in zip(self.infer_session.get_outputs(), outs)])
            if self.max_intermediate_outputs and len(self.intermediate_outputs) >= self.max_intermediate_outputs:
                self.compute_data()           # merges into calibrate_tensors_range
                self.clear_collected_data()
        if self.intermediate_outputs:
            self.compute_data()
            self.clear_collected_data()
        if self.calibrate_tensors_range is None:
            raise ValueError("No data is collected.")

    C.MinMaxCalibrater.collect_data = collect_data

    # The calibration session runs the augmented graph (a min and a max for
    # every tensor). With ORT's defaults — memory arena, memory-pattern
    # planning, every core — a 1B model reached 20 GB resident and took the
    # WSL VM down with it. No arena, no pattern, four threads.
    import onnxruntime as ort

    def create_inference_session(self):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.intra_op_num_threads = int(os.environ.get("PORT_THREADS", "4"))
        self.infer_session = ort.InferenceSession(self.augmented_model_path, sess_options=so,
                                                  providers=self.execution_providers)

    C.CalibraterBase.create_inference_session = create_inference_session


def calib_inputs(model_dir: Path, family: str, calib_audio):
    """Model inputs for calibration, at the NPU's fixed shape, built by the
    same frontend validation uses."""
    import numpy as np
    from porting import frontends
    xs = []
    if family == "wav2vec2-ctc":
        cfg = frontends.ctc_config(model_dir)
        for a in calib_audio:
            xs.extend(x for x, _ in frontends.ctc_windows(a, cfg, CTC_WINDOW_S))
    else:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained(model_dir)
        for a in calib_audio:
            xs.append(fe(a, sampling_rate=SR, return_tensors="np").input_features.astype(np.float32))
    return xs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--calib-lang", default=None, help="evalpack language for QDQ calibration")
    ap.add_argument("--skip", nargs="*", default=[], choices=["int8", "fp16", "npu"])
    args = ap.parse_args()

    name = args.repo.split("/")[-1]
    fam = family_of(args.repo)
    root = ART / name
    out = root / "onnx-web"
    log = {"repo": args.repo, "family": fam, "steps": {}}
    t0 = time.perf_counter()

    onnx_dir = out / "onnx"
    if not (onnx_dir / ("encoder_model.onnx" if fam == "whisper" else "model.onnx")).exists():
        export(args.repo, out)
    graphs = {p.stem: p for p in onnx_dir.glob("*.onnx")}
    log["steps"]["export"] = sorted(graphs)
    print(f"{name}: fp32 graphs {sorted(graphs)} ({time.perf_counter() - t0:.0f}s)")

    # What a runtime loads: the encoder and the *merged* decoder (one graph with
    # and without KV cache), or the single CTC graph.
    main_graphs = ([graphs["encoder_model"], graphs["decoder_model_merged"]] if fam == "whisper"
                   else [graphs["model"]])
    for g in main_graphs:
        if "int8" not in args.skip and not g.with_name(g.stem + "_quantized.onnx").exists():
            log["steps"].setdefault("int8", []).append(quantize_int8(g).name)
    if "fp16" not in args.skip:
        for g in main_graphs:
            h = g.with_name(g.stem + "_fp16.onnx")
            if h.exists():
                continue
            if g.stem == "decoder_model_merged":
                # Converting the merged graph breaks it: the fp16 pass renames
                # values inside the If branches and leaves a branch output
                # pointing outside ("Subgraph output (logits) is an outer scope
                # value", measured in onnxruntime-web). Convert the two
                # decoders, then merge them the way the exporter did.
                from optimum.onnx.graph_transformations import merge_decoders
                d16 = to_fp16(graphs["decoder_model"])
                p16 = to_fp16(graphs["decoder_with_past_model"])
                merge_decoders(d16, p16, save_path=h, strict=False)
                d16.unlink()
                p16.unlink()
            else:
                to_fp16(g)
            log["steps"].setdefault("fp16", []).append(h.name)
    print(f"  int8 + fp16 done ({time.perf_counter() - t0:.0f}s)")
    # the split decoders are redundant beside the merged one and double the download
    for stem in ("decoder_model", "decoder_with_past_model"):
        if stem in graphs and graphs[stem].exists() and (
                "fp16" in args.skip or (onnx_dir / "decoder_model_merged_fp16.onnx").exists()):
            graphs[stem].unlink()

    npu_graph = main_graphs[0]
    q = root / "npu" / f"{npu_graph.stem}_qdq_int8.onnx"
    if "npu" not in args.skip and args.calib_lang and not q.exists():
        from porting.evalpack import load
        audio, _ = load(args.calib_lang, split="calib")
        if big(npu_graph):          # >2 GB graphs: half the clips, half the calibration time
            audio = audio[:32]
        q = npu_qdq(npu_graph, fam, root, calib_inputs(out, fam, audio))
        log["steps"]["npu_qdq"] = str(q.relative_to(root))
        print(f"  npu QDQ: {q.name} ({time.perf_counter() - t0:.0f}s)")

    sizes = {str(p.relative_to(root)): round(p.stat().st_size / 2**20, 1)
             for p in root.rglob("*") if p.suffix in (".onnx", ".bin", ".onnx_data")}
    log["sizes_mb"] = sizes
    log["seconds"] = round(time.perf_counter() - t0)
    (root / "build_onnx.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(sizes, indent=1))


if __name__ == "__main__":
    main()
