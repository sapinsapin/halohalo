"""Collect every result file into docs/porting_report.md (and summary.json).

  python -m porting report
"""

import json
import os
from collections import Counter
from pathlib import Path

from porting.registry import FAMILIES, TARGETS, plan

ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
RESULTS = FINETUNE_DIR / "port" / "results"
ART = FINETUNE_DIR / "port" / "artefacts"
OUT = ROOT / "docs" / "porting_report.md"

STANDS_FOR = {   # what a runtime/variant result tells you about which target
    ("torch", "fp32"): "reference (the published checkpoint; also AMD ROCm as-is)",
    ("ort", "fp32"): "Arm CPU ONNX fp32; AMD MIGraphX; GPU EPs",
    ("ort", "int8"): "Arm CPU ONNX int8 (dynamic)",
    ("ort", "fp16"): "WebGPU fp16 graph (numerics on CPU)",
    ("ort", "qdq"): "NPU graph, A8W8 static QDQ (numerics on CPU)",
    ("ort", "qdq16"): "NPU graph, A16W8 static QDQ, Qualcomm's transformer default (numerics on CPU)",
    ("webgpu", "fp16"): "browser on WebGPU (measured in Chrome on the RTX 3070)",
    ("webgpu", "fp32"): "browser on WebGPU, fp32",
    ("webgpu", "q4f16"): "browser on WebGPU, 4-bit weights (measured in Chrome on the RTX 3070)",
    ("wasm", "q8"): "browser on WebAssembly (CPU), int8",
    ("transformersjs", "fp32"): "browser/Node, Transformers.js fp32",
    ("transformersjs", "int8"): "browser wasm, Transformers.js q8",
    ("whispercpp", "f16"): "whisper.cpp f16 (Arm CPU)",
    ("whispercpp", "q8_0"): "whisper.cpp q8_0 (Arm CPU)",
    ("whispercpp", "q5_0"): "whisper.cpp q5_0 (Arm CPU, phones)",
    ("openvino", "fp16"): "Intel NPU/iGPU/CPU, OpenVINO fp16",
    ("openvino", "int8"): "Intel CPU/NPU, OpenVINO int8 weights",
    ("executorch", "xnnpack"): "Android/iOS CPU, ExecuTorch XNNPACK fp32 (CTC: fixed 10 s windows, zero-padded)",
    ("executorch", "xnnpack-int8"): "Android/iOS CPU, ExecuTorch XNNPACK int8",
    ("mlx", "fp16"): "Apple silicon, MLX fp16",
    ("mlx", "q8"): "Apple silicon, MLX 8-bit",
    ("mlx", "q4"): "Apple silicon, MLX 4-bit",
}


TARGET_OF = {   # which target a runtime/variant result speaks for
    ("ort", "fp32"): ["arm-cpu-onnx", "amd-rocm"], ("ort", "int8"): ["arm-cpu-onnx"],
    ("ort", "qdq"): ["npu-qnn", "npu-ryzenai"], ("ort", "qdq16"): ["npu-qnn", "npu-ryzenai"],
    ("whispercpp", "f16"): ["arm-cpu-ggml"], ("whispercpp", "q8_0"): ["arm-cpu-ggml"],
    ("whispercpp", "q5_0"): ["arm-cpu-ggml"],
    ("openvino", "fp16"): ["npu-openvino"], ("openvino", "int8"): ["npu-openvino"],
    ("executorch", "xnnpack"): ["arm-mobile-executorch"], ("executorch", "xnnpack-int8"): ["arm-mobile-executorch"],
    ("mlx", "fp16"): ["mac-mlx"], ("mlx", "q8"): ["mac-mlx"], ("mlx", "q4"): ["mac-mlx"],
    ("webgpu", "fp16"): ["web-webgpu"], ("webgpu", "q4f16"): ["web-webgpu"],
    ("transformersjs", "int8"): ["web-webgpu (wasm fallback)"],
    ("transformersjs", "fp32"): ["web-webgpu (wasm fallback)"],
}
BUDGET_CER = 0.5     # percentage points of CER a port may lose against PyTorch and still be chosen


def ref_cer(ref, r):
    """PyTorch's CER on the same clips as r: short parity checks (5 or 20
    clips on slow CPU stand-ins) must not be compared with a 100-clip score."""
    n = r.get("clips", ref["clips"])
    if n >= ref["clips"] or not ref.get("hyps"):
        return ref["accuracy"]["cer"]
    refs_file = FINETUNE_DIR / "port" / "evalpack" / "wav" / r.get("lang", "ceb") / "refs.json"
    if not refs_file.exists():
        return ref["accuracy"]["cer"]
    from porting.metrics import score
    return score(json.loads(refs_file.read_text())[:n], ref["hyps"][:n])["cer"]


def recommend(rows):
    """Per target: the smallest artefact within BUDGET_CER of PyTorch, ties to
    the faster. That is the 'optimal' port: as small as the target allows
    without costing accuracy."""
    ref = next((r for r in rows if r["runtime"] == "torch"), None)
    if not ref:
        return []
    by_target = {}
    for r in rows:
        for t in TARGET_OF.get((r["runtime"], r["variant"]), []):
            by_target.setdefault(t, []).append(r)
    out = ["| target | choose | size MB | Δ CER vs PyTorch | RTF | rejected (over budget) |",
           "|---|---|---:|---:|---:|---|"]
    for t, rs in sorted(by_target.items()):
        d = lambda r: (r["accuracy"]["cer"] - ref_cer(ref, r)) * 100
        ok = [r for r in rs if d(r) <= BUDGET_CER]
        bad = [f"{r['runtime']}/{r['variant']} ({d(r):+.1f})" for r in rs if d(r) > BUDGET_CER]
        if not ok:
            out.append(f"| {t} | none within budget | — | — | — | {', '.join(bad)} |")
            continue
        best = min(ok, key=lambda r: ((r.get("size_mb") or 1e9), r["rtf"]))
        out.append(f"| {t} | {best['runtime']}/{best['variant']} | {best.get('size_mb') or '—'} | "
                   f"{d(best):+.2f} | {best['rtf']:.3f} | {', '.join(bad) or '—'} |")
    return out


def f(x, pct=True, nd=1):
    if x is None:
        return "—"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def asr_table(rows):
    ref = next((r for r in rows if r["runtime"] == "torch"), None)
    if ref:           # results filed before the reference existed (browser runs) get parity now
        from porting.metrics import agreement
        for r in rows:
            if r is not ref and "parity" not in r and r.get("hyps"):
                r["parity"] = agreement(ref["hyps"][:len(r["hyps"])], r["hyps"])
    lines = ["| runtime / variant | stands for | clips | size MB | load s | RTF | CER % | WER % | Δ CER vs PyTorch (same clips) | CER vs PyTorch output % | identical to PyTorch % |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = list(STANDS_FOR)
    rows = sorted(rows, key=lambda r: order.index((r["runtime"], r["variant"]))
                  if (r["runtime"], r["variant"]) in order else 99)
    for r in rows:
        acc, par = r["accuracy"], r.get("parity", {})
        d = (acc["cer"] - ref_cer(ref, r)) * 100 if ref and r is not ref else None
        lines.append(f"| {r['runtime']} / {r['variant']} | {STANDS_FOR.get((r['runtime'], r['variant']), '')} "
                     f"| {r.get('clips', '—')} | {r.get('size_mb') or '—'} | {r.get('load_seconds', '—')} | {r['rtf']:.3f} | {f(acc['cer'])} | {f(acc['wer'])} "
                     f"| {'—' if d is None else f'{d:+.2f}'} | {f(par.get('cer_vs_reference'), nd=2)} "
                     f"| {f(par.get('identical'), nd=0)} |")
    return lines


def write() -> Path:
    rows = plan()
    lines = ["# Porting report", "",
             "Generated by `python -m porting report` from `$FINETUNE_DIR/port/results/`. "
             "How the pipeline works and what each target needs: [porting_pipeline.md](porting_pipeline.md).", "",
             "## What applies where, and when", "",
             "P1 = the workstation (RTX 3070, 8 GB) can convert and check it. "
             "P2 = waits for the RTX PRO 6000 (the reason is in `python -m porting plan -v`). "
             "— = does not apply (reason in `porting/registry.py`).", ""]
    fams = list(FAMILIES)
    lines.append("| target | " + " | ".join(fams) + " |")
    lines.append("|---|" + "---|" * len(fams))
    for t in TARGETS:
        cells = []
        for fam in fams:
            rs = [r for r in rows if r["family"] == fam and r["target"] == t.id]
            if not rs:
                cells.append("—")
                continue
            c = Counter(r["phase"] for r in rs)
            cells.append(" ".join(f"P{p}×{n}" for p, n in sorted(c.items())))
        lines.append(f"| {t.id} | " + " | ".join(cells) + " |")
    c = Counter(r["phase"] for r in rows)
    lines += ["", f"{len(rows)} model–target ports: {c.get(1, 0)} in Phase 1, {c.get(2, 0)} in Phase 2.", ""]

    summary = {}
    if RESULTS.exists():
        for d in sorted(RESULTS.iterdir()):
            files = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
            asr = [x for x in files if "accuracy" in x]
            snac = [x for x in files if "logmel_db_port_vs_torch" in x]
            tts = [x for x in files if "judge" in x]
            other = [x for x in files if x not in asr and x not in snac and x not in tts]
            if tts:
                lines += [f"## {d.name} (Orpheus TTS, end to end)", "",
                          "Text → the ported LLM → SNAC → audio, re-transcribed by the independent "
                          f"judge ({tts[0]['judge']}). Sampling is stochastic, so compare CER with the "
                          "PyTorch row on the same sentences, not token by token.", "",
                          "| runtime / variant | sentences | judge CER % | tokens/s | audio s | empty outputs |",
                          "|---|---:|---:|---:|---:|---:|"]
                for x in sorted(tts, key=lambda x: (x["runtime"] != "torch", x["runtime"], x["variant"])):
                    lines.append(f"| {x['runtime']} / {x['variant']} | {x['sentences']} | {100 * x['cer']:.1f} | "
                                 f"{x['tokens_per_second']} | {x['audio_seconds']} | {x['empty_outputs']} |")
                lines.append("")
            if asr:
                by_lang = {}
                for x in asr:
                    by_lang.setdefault(x["lang"], []).append(x)
                for lang, rs in by_lang.items():
                    n = rs[0]["clips"]
                    lines += [f"## {d.name} [{lang}]", "",
                              f"{n} frozen-test clips ({rs[0]['audio_seconds']} s), CPU, "
                              f"{rs[0]['threads']} threads. RTF = seconds of compute per second of audio "
                              "on this Ryzen 7 3700X, loading excluded: compare variants with it, not "
                              "devices. Load times mostly measure the workstation's hard disk.", "",
                              "A negative Δ CER means fewer errors than PyTorch. For Whisper that comes "
                              "from decoding, not from quantisation improving the model: whisper.cpp and "
                              "the int8 graphs break some repetition loops (a clip that PyTorch runs to the "
                              "225-token cap) where PyTorch does not. The two parity columns say how far "
                              "the transcripts actually differ.", ""]
                    lines += asr_table(rs) + [""]
                    rec = recommend(rs)
                    if rec:
                        lines += [f"**What to ship, per target** (smallest artefact within {BUDGET_CER} CER "
                                  "points of PyTorch):", ""] + rec + [""]
                    summary[f"{d.name}/{lang}"] = [{k: x.get(k) for k in ("runtime", "variant", "size_mb", "rtf")}
                                                   | {"cer": x["accuracy"]["cer"], "wer": x["accuracy"]["wer"],
                                                      "parity_cer": x.get("parity", {}).get("cer_vs_reference")}
                                                   for x in rs]
            if snac:
                lines += [f"## {d.name}", "",
                          "Log-mel distance of the port's audio from PyTorch's, next to the distance "
                          "between two PyTorch runs (SNAC adds noise inside its decoder).", "",
                          "| variant | port vs PyTorch dB | PyTorch vs PyTorch dB | verdict | RTF |",
                          "|---|---:|---:|---|---:|"]
                for x in snac:
                    lines.append(f"| {x['variant']} | {x['logmel_db_port_vs_torch']} | "
                                 f"{x['logmel_db_torch_vs_torch']} | {x['parity']} | {x['rtf']} |")
                lines.append("")
                summary[d.name] = snac
            for x in other:
                if "identical_top_label" in x:          # the language identifier
                    lines += [f"## {d.name} (fastText language ID)", "",
                              f"The same model.ftz in the browser (WebAssembly) and in Python, over "
                              f"{x['sentences']} held-out sentences in ten languages.", "",
                              "| build | top-1 accuracy % | same label as Python % | max probability difference | model MB |",
                              "|---|---:|---:|---:|---:|",
                              f"| Python (C++) | {100 * x['accuracy_python']:.1f} | — | — | {x['model_mb']} |",
                              f"| browser ({x['variant']}) | {100 * x['accuracy_wasm']:.1f} | "
                              f"{100 * x['identical_top_label']:.1f} | {x['max_prob_diff']:.2g} | {x['model_mb']} |", ""]
                else:
                    lines += [f"## {d.name}", "", "```json",
                              json.dumps(x, indent=1, ensure_ascii=False)[:2000], "```", ""]

    built_only = []
    if ART.exists():
        for d in sorted(ART.iterdir()):
            for sub, why in (("coreml", "Core ML: predicts only on macOS"),
                             ("executorch/coreml", "ExecuTorch Core ML: runs only on Apple devices")):
                p = d / sub
                if p.exists() and any(p.iterdir()):
                    mb = sum(q.stat().st_size for q in p.rglob("*") if q.is_file()) / 2**20
                    built_only.append(f"| {d.name} | {sub}/ | {mb:.0f} | {why} |")
    if built_only:
        lines += ["## Built, not checkable on this machine", "",
                  "| model | artefact | MB | why |", "|---|---|---:|---|"] + built_only + [""]
    OUT.write_text("\n".join(lines) + "\n")
    (RESULTS / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    return OUT
