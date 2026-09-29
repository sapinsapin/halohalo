"""The porting pipeline: plan, build, validate, report.

  python -m porting plan  [--phase 1]                 # what applies where, and when
  python -m porting build --phase 1 [--models demo]   # run the exporters
  python -m porting validate --phase 1 [--models demo] [--n 100]
  python -m porting report                            # docs/porting_report.md
  python -m porting all --phase 1                     # build + validate + report

Runs from any Python (stdlib only): every step is a subprocess in its
toolchain's venv (porting/setup_venvs.sh), with CUDA hidden, so a porting run
never competes with a training queue for the GPU. Steps are resumable: a
build whose artefact exists and a validation whose result file exists are
skipped.

Model sets: `demo` = one model per family (the fully validated set);
`phase` = every model the phase admits; or a comma list of repo names.
Siblings of a demo model (other languages, same architecture) get a short
smoke validation (--smoke clips) instead of the full evalpack: the port is the
same code path, only the weights differ.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from porting.registry import MODELS, TARGET_BY_ID, plan

ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
PORT = FINETUNE_DIR / "port"
ART, RESULTS, LOGS = PORT / "artefacts", PORT / "results", PORT / "logs"

VENV = {"onnx": "venv_port_onnx", "openvino": "venv_port_ov", "executorch": "venv_port_et",
        "mlx": "venv_port_mlx", "coreml": "venv_port_coreml",
        "main": "venv"}           # the training venv: has fasttext, the TTS judge, SNAC

DEMO = ["sapinsapin/whisper-small-pld-ceb",
        "sapinsapin/omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm",
        "hubertsiuzdak/snac_24khz"]

LANGS = ("bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war")


def lang_of(repo):
    name = repo.split("/")[-1]
    return next((l for l in LANGS if f"_{l}" in name or f"-{l}" in name), None)


def onnx_args(repo):
    """NPU calibration only where the workstation can hold it (registry.NPU_CALIB_MAX_M);
    larger models get theirs in Phase 2 (a big-memory host sets NPU_CALIB_MAX_M
    higher; the registry's phase split stays the workstation's)."""
    from porting.registry import NPU_CALIB_MAX_M
    limit = float(os.environ.get("NPU_CALIB_MAX_M", NPU_CALIB_MAX_M))
    m = next((m for m in MODELS if m.repo == repo), None)
    if m and m.params_m > limit:
        return ["--skip", "npu"]
    return ["--calib-lang", lang_of(repo) or "ceb"]


# ------------------------------------------------------------------ recipes
# A build: (toolchain venv, module, marker path relative to the artefact dir).
# The marker's existence means "built"; args are always [repo] + extra.
BUILDS = {
    "onnx": ("onnx", "porting.export_onnx", "build_onnx.json", lambda r: onnx_args(r)),
    "openvino": ("openvino", "porting.export_openvino", "openvino/build.json", lambda r: []),
    "ggml": ("onnx", "porting.export_ggml", "ggml/build.json", lambda r: []),
    "mlx": ("mlx", "porting.export_mlx", "mlx/q4/config.json", lambda r: []),
    "executorch": ("executorch", "porting.export_executorch", "executorch/build.json", lambda r: []),
    "executorch-coreml": ("executorch", "porting.export_executorch", "executorch/build_coreml.json",
                          lambda r: ["--backend", "coreml"]),
    "coreml": ("coreml", "porting.export_coreml", "coreml/build.json", lambda r: []),
    "snac": ("onnx", "porting.export_snac", "build_snac.json", lambda r: []),
    "snac-executorch": ("executorch", "porting.export_snac", "executorch/build.json",
                        lambda r: ["--target", "executorch"]),
    "snac-coreml": ("coreml", "porting.export_snac", "coreml/build.json", lambda r: ["--target", "coreml"]),
    "lid-web": ("main", "porting.validate_lid", "web/build.json", lambda r: ["--build-only"]),
}

# A validation: (toolchain venv, runtime, variants). None = cannot run here.
V = lambda venv, rt, *vs: (venv, rt, list(vs))
RECIPES = {   # (family, target) -> (builds, validations, what a device loads)
    ("whisper", "arm-cpu-onnx"): (["onnx"], [V("onnx", "ort", "fp32", "int8")], "onnx-web/onnx/*_quantized.onnx"),
    ("whisper", "arm-cpu-ggml"): (["ggml"], [V("onnx", "whispercpp", "f16", "q8_0", "q5_0")], "ggml/ggml-model-q5_0.bin"),
    ("whisper", "arm-mobile-executorch"): (["executorch"], [V("executorch", "executorch", "xnnpack")], "executorch/xnnpack/*.pte"),
    ("whisper", "npu-qnn"): (["onnx"], [V("onnx", "ort", "qdq", "qdq16")], "npu/encoder_model_qdq_a16w8.onnx + decoder on CPU"),
    ("whisper", "npu-ryzenai"): (["onnx"], [V("onnx", "ort", "qdq", "qdq16")], "npu/encoder_model_qdq_a16w8.onnx + decoder on CPU"),
    ("whisper", "npu-ane"): (["coreml"], [], "coreml/encoder.mlpackage (+ whisper.cpp or WhisperKit decoder)"),
    ("whisper", "npu-openvino"): (["openvino"], [V("openvino", "openvino", "fp16", "int8")], "openvino/int8/"),
    ("whisper", "mac-mlx"): (["mlx"], [V("mlx", "mlx", "fp16", "q8", "q4")], "mlx/q4/ or mlx/fp16/"),
    ("whisper", "mac-executorch"): ([], [], "macOS only: python -m porting.export_executorch <repo> --backend coreml, on a Mac"),
    ("whisper", "web-webgpu"): (["onnx"], [V("onnx", "transformersjs", "fp32", "int8"), V("onnx", "ort", "fp16")],
                                "onnx-web/ (fp16 on WebGPU, q8 on wasm)"),
    ("whisper", "amd-rocm"): (["onnx"], [], "the checkpoint as-is (PyTorch ROCm) or onnx-web/onnx/*.onnx (MIGraphX EP)"),

    ("wav2vec2-ctc", "arm-cpu-onnx"): (["onnx"], [V("onnx", "ort", "fp32", "int8")], "onnx-web/onnx/model_quantized.onnx"),
    ("wav2vec2-ctc", "arm-mobile-executorch"): (["executorch"], [V("executorch", "executorch", "xnnpack", "xnnpack-int8")],
                                                "executorch/xnnpack-int8/model.pte"),
    ("wav2vec2-ctc", "npu-qnn"): (["onnx"], [V("onnx", "ort", "qdq", "qdq16")], "npu/model_qdq_a16w8.onnx (10 s window)"),
    ("wav2vec2-ctc", "npu-ryzenai"): (["onnx"], [V("onnx", "ort", "qdq", "qdq16")], "npu/model_qdq_a16w8.onnx (10 s window)"),
    ("wav2vec2-ctc", "npu-ane"): (["coreml"], [], "coreml/model.mlpackage (10 s window)"),
    ("wav2vec2-ctc", "npu-openvino"): (["openvino"], [V("openvino", "openvino", "fp16", "int8")], "openvino/int8/"),
    ("wav2vec2-ctc", "mac-executorch"): ([], [], "macOS only: python -m porting.export_executorch <repo> --backend coreml, on a Mac"),
    ("wav2vec2-ctc", "web-webgpu"): (["onnx"], [V("onnx", "transformersjs", "fp32", "int8"), V("onnx", "ort", "fp16")],
                                     "onnx-web/onnx/model_fp16.onnx (WebGPU) or model_quantized.onnx (wasm)"),
    ("wav2vec2-ctc", "amd-rocm"): (["onnx"], [], "the checkpoint as-is or onnx-web/onnx/model.onnx"),

    ("snac", "arm-cpu-onnx"): (["snac"], [V("onnx", "snac", "fp32", "fp16")], "snac/decoder.onnx"),
    ("snac", "web-webgpu"): (["snac"], [V("onnx", "snac", "fp16")], "snac/decoder_fp16.onnx"),
    ("snac", "arm-mobile-executorch"): (["snac-executorch"], [V("executorch", "snac", "executorch")], "executorch/decoder.pte"),
    ("snac", "npu-ane"): (["snac-coreml"], [], "coreml/decoder.mlpackage"),

    ("fasttext", "web-webgpu"): (["lid-web"], [V("main", "lid", "wasm")], "web/model.ftz + fasttext.wasm"),
}


def select(models_arg: str, phase: int | None, targets: str | None = None):
    rows = [r for r in plan() if phase is None or r["phase"] == phase]
    if targets:
        keep = {t.strip() for t in targets.split(",")}
        rows = [r for r in rows if r["target"] in keep]
    if models_arg == "demo":
        rows = [r for r in rows if r["model"] in DEMO or r["family"] == "fasttext"]
    elif models_arg != "phase":
        want = {m.strip() for m in models_arg.split(",")}
        rows = [r for r in rows if r["model"] in want or r["model"].split("/")[-1] in want]
    return rows


def mem_available_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("inf")


def est_gb(repo: str, kind: str) -> float:
    """A job's expected peak RAM, so a start leaves PORT_MIN_FREE_GB free
    after the job has grown: conversion ~3x the fp32 weights (as
    registry.phase_of assumes), static int8 calibration ~7x (the 963M CTC
    graph peaked at 25 GB), a validation ~1.5x."""
    m = next((m for m in MODELS if m.repo == repo), None)
    fp32 = (m.params_m if m else 0) * 1e6 * 4 / 2**30
    return fp32 * {"build": 3, "calib": 7, "validate": 1.5}[kind]


def wait_for_memory(need_gb: float = 0.0):
    """Start a job only while PORT_MIN_FREE_GB (default 16) would remain free
    once it reaches its expected peak: parallelism fills spare memory, never
    the headroom an OOM kill needs (in WSL one takes every session down; on
    the GB10 the GPU shares the same 128 GB)."""
    need = float(os.environ.get("PORT_MIN_FREE_GB", "16")) + need_gb
    try:
        total = int(open("/proc/meminfo").readline().split()[1]) / 2**20
        need = min(need, total - 4)       # a job bigger than the machine waits for an idle one, not forever
    except (OSError, ValueError, IndexError):
        pass
    while mem_available_gb() < need:
        time.sleep(20)


_START = threading.Lock()


def run(venv: str, module: str, args: list[str], log: Path, env_extra: dict | None = None,
        need_gb: float = 0.0) -> bool:
    py = ROOT / VENV[venv] / "bin" / "python3"
    if not py.exists():
        print(f"    skip: {VENV[venv]} not installed (bash porting/setup_venvs.sh {venv})")
        return False
    import tempfile
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT),
               HF_XET_CHUNK_CACHE_SIZE_BYTES="0", TMPDIR=os.environ.get("TMPDIR") or tempfile.gettempdir())
    env.update(env_extra or {})
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log, "a") as f:
        f.write(f"\n=== {time.strftime('%F %T')} {module} {' '.join(args)}\n")
        f.flush()
        # PORT_CPUS pins validations to fixed cores (e.g. "0-3") so their RTF
        # is comparable while builds run on the others
        pin = (["taskset", "-c", os.environ["PORT_CPUS"]]
               if os.environ.get("PORT_CPUS") and "validate" in module else [])
        # one start at a time, held PORT_STAGGER_S (with --jobs > 1) so the
        # next gate reads memory after this job has begun to grow
        with _START:
            wait_for_memory(need_gb)
            p = subprocess.Popen([*pin, str(py), "-u", "-m", module, *args], cwd=ROOT, env=env,
                                 stdout=f, stderr=subprocess.STDOUT)
            time.sleep(float(os.environ.get("PORT_STAGGER_S", "0")))
        rc = p.wait()
    print(f"    {'ok ' if rc == 0 else 'FAIL'} {module} {' '.join(args)} ({time.perf_counter() - t0:.0f}s)"
          + ("" if rc == 0 else f"  -> {log}"))
    return rc == 0


def cmd_plan(args):
    rows = select(args.models, args.phase, args.targets)
    by = {}
    for r in rows:
        by.setdefault((r["phase"], r["family"]), []).append(r)
    for (ph, fam), rs in sorted(by.items()):
        print(f"phase {ph} · {fam}: {len({r['model'] for r in rs})} models x "
              f"{len({r['target'] for r in rs})} targets = {len(rs)} ports")
        if args.verbose:
            for r in rs:
                print(f"    {r['model'].split('/')[-1]:<48} {r['target']:<22} {r['how']}")
        for why in sorted({r["why"] for r in rs}):
            print(f"    why phase {ph}: {why}")


def run_groups(groups: dict, jobs: int):
    """groups: {model: [task thunks]}. A model's tasks run in order (they
    share files); different models run side by side, up to `jobs` at once
    and only while memory allows (wait_for_memory, inside run())."""
    if jobs <= 1:
        for tasks in groups.values():
            for t in tasks:
                t()
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for f in [pool.submit(lambda ts=ts: [t() for t in ts]) for ts in groups.values()]:
            f.result()


def cmd_build(args):
    done, groups = set(), {}
    for r in select(args.models, args.phase, args.targets):
        builds, _, _ = RECIPES.get((r["family"], r["target"]), ([], [], ""))
        for b in builds:
            if (r["model"], b) in done:
                continue
            done.add((r["model"], b))
            venv, module, marker, extra = BUILDS[b]
            name = r["model"].split("/")[-1]
            if (ART / name / marker).exists():
                continue
            a = [r["model"], *extra(r["model"])]
            need = est_gb(r["model"], "calib" if "--calib-lang" in a else "build")
            groups.setdefault(r["model"], []).append(
                lambda venv=venv, module=module, a=a, name=name, b=b, need=need: (
                    print(f"  build {name} · {b}"),
                    run(venv, module, a, LOGS / f"build_{name}_{b}.log", need_gb=need)))
    run_groups(groups, args.jobs)


def cmd_validate(args):
    done, groups = set(), {}
    host = os.environ.get("PORT_HOST")        # results from another machine are tagged <runtime>@host
    for r in select(args.models, args.phase, args.targets):
        _, vals, _ = RECIPES.get((r["family"], r["target"]), ([], [], ""))
        name = r["model"].split("/")[-1]
        n = args.n if r["model"] in DEMO else args.smoke
        lang = lang_of(r["model"])
        if r["family"] in ("whisper", "wav2vec2-ctc"):
            ref = RESULTS / name / f"torch-fp32-{lang}.json"
            if (r["model"], "torch") not in done and not ref.exists():
                done.add((r["model"], "torch"))
                # the reference is the same on every machine: never host-tagged
                groups.setdefault(r["model"], []).append(
                    lambda r=r, n=n, name=name: (
                        print(f"  validate {name} · torch/fp32 (reference)"),
                        run("onnx", "porting.validate", [r["model"], "--runtime", "torch", "--n", str(n)],
                            LOGS / f"validate_{name}.log", {"PORT_HOST": ""},
                            need_gb=est_gb(r["model"], "validate"))))
        for venv, rt, variants in vals:
            for v in variants:
                if (r["model"], rt, v) in done:
                    continue
                done.add((r["model"], rt, v))
                if rt in ("snac", "lid"):
                    module, a = f"porting.validate_{rt}", [r["model"], "--variant", v]
                    out = RESULTS / name / (f"{rt}-{v}@{host}.json" if host else f"{rt}-{v}.json")
                else:
                    module = "porting.validate"
                    # MLX's x86 CPU kernels are a slow stand-in for Metal, and
                    # ExecuTorch's Python runtime runs a 1B model single-threaded
                    # (~40 min for 100 clips): a 20-clip parity check for those
                    big = next((m.params_m for m in MODELS if m.repo == r["model"]), 0) > 500
                    slow = rt == "mlx" or (rt == "executorch" and big)
                    cap = 5 if rt == "mlx" else 20        # MLX on x86: ~2 min a clip, one core
                    a = [r["model"], "--runtime", rt, "--variant", v, "--n", str(min(n, cap) if slow else n)]
                    lab = f"{rt}@{host}" if host else rt
                    out = RESULTS / name / f"{lab}-{v}-{lang}.json"
                if out.exists():
                    continue
                groups.setdefault(r["model"], []).append(
                    lambda venv=venv, module=module, a=a, name=name, rt=rt, v=v, r=r: (
                        print(f"  validate {name} · {rt}/{v}"),
                        run(venv, module, a, LOGS / f"validate_{name}.log",
                            need_gb=est_gb(r["model"], "validate"))))
    run_groups(groups, args.jobs)


def cmd_report(args):
    from porting.report import write
    print(write())


def main():
    ap = argparse.ArgumentParser(prog="python -m porting", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "build", "validate", "report", "all"):
        p = sub.add_parser(name)
        p.add_argument("--phase", type=int, default=None if name in ("plan", "report") else 1)
        p.add_argument("--models", default="phase" if name == "plan" else "demo")
        p.add_argument("--n", type=int, default=100, help="clips for demo models")
        p.add_argument("--smoke", type=int, default=20, help="clips for sibling models")
        p.add_argument("-v", "--verbose", action="store_true")
        p.add_argument("--targets", default=None, help="comma list of target ids, e.g. arm-cpu-onnx,arm-cpu-ggml")
        p.add_argument("--jobs", type=int, default=int(os.environ.get("PORT_JOBS", "1")),
                       help="models to build/validate side by side (memory permitting)")
    args = ap.parse_args()
    if args.cmd == "plan":
        cmd_plan(args)
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "report":
        cmd_report(args)
    else:
        cmd_build(args)
        cmd_validate(args)
        cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
