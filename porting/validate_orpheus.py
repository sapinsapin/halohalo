"""End-to-end check of a ported Orpheus: text -> LLM runtime -> SNAC -> audio
-> the independent MMS-1b-all judge. Phase 2; the llama.cpp path also runs on
the workstation CPU as a dry run.

  python -m porting.validate_orpheus sapinsapin/orpheus-3b-0.1-pretrained-char-pld-ceb \
      --runtime llamacpp --variant q4_k_m --n 10

Runs in the main venv (it has the judge, SNAC and the Orpheus frontend). The
runtimes are reached the way an app reaches them:

  torch       merged checkpoint, bf16 (the reference; GPU)
  llamacpp    llama-server over HTTP with prompt token ids (GGUF)
  onnx        onnxruntime-genai Generator (int4)
  openvino    optimum-intel OVModelForCausalLM (int4)
  mlx         mlx_lm (Apple; Linux CPU wheel as a proxy)

Sentences, speakers and the frontend are tts_eval's (the manifest behind the
published TTS table), so a port's judge CER is directly comparable with the
published PyTorch number. Sampling is stochastic (temperature 0.6, top-p 0.9,
repetition penalty 1.1, each runtime's own sampler), so parity is a CER
difference against the torch run on the same sentences, not token identity.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"
RESULTS = FINETUNE_DIR / "port" / "results"
TOOLS = Path(os.environ.get("PORT_TOOLS", "/mnt/d/halohalo/third_party"))
SAMPLING = {"temperature": 0.6, "top_p": 0.9, "repetition_penalty": 1.1}
MAX_NEW = 1400


def prompts(repo, n):
    import finetune_orpheus as fo
    import tts_eval
    from transformers import AutoTokenizer
    units, lang = tts_eval.adapter_spec(repo.split("/")[-1].replace("orpheus-3b-0.1-pretrained-", "orpheus_").replace("-pld-", "_pld_"))
    rows = [r for r in tts_eval.manifest() if r["lang"] == lang][:n]
    tok = AutoTokenizer.from_pretrained(repo, token=os.environ.get("HF_TOKEN"))
    out = []
    for r in rows:
        ids = tok(f"{r['speaker_id']}: {fo.frontend_text(r['text'], units)}", add_special_tokens=False).input_ids
        out.append((r, [fo.SOH] + ids + [fo.EOT, fo.EOH, fo.SOAI, fo.SOS]))
    return lang, out


def cut(tokens):
    import finetune_orpheus as fo
    return tokens[:tokens.index(fo.EOS_SPEECH)] if fo.EOS_SPEECH in tokens else tokens


# ------------------------------------------------------------------ runtimes
def gen_torch(root, variant, ids_list):
    import torch
    from transformers import AutoModelForCausalLM
    import finetune_orpheus as fo
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(root / "merged", dtype=torch.bfloat16, device_map={"": dev}).eval()
    outs = []
    for ids in ids_list:
        x = torch.tensor([ids], device=dev)
        with torch.inference_mode():
            g = m.generate(x, max_new_tokens=MAX_NEW, do_sample=True, eos_token_id=fo.EOS_SPEECH,
                           pad_token_id=128263, **SAMPLING)
        outs.append(g[0, len(ids):].tolist())
    return outs


def gen_llamacpp(root, variant, ids_list):
    import urllib.request
    exe = TOOLS / "llama.cpp" / "build" / "bin" / "llama-server"
    port = 8089
    ngl = "99" if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" else "0"
    srv = subprocess.Popen([str(exe), "-m", str(root / "gguf" / f"model-{variant}.gguf"), "--port", str(port),
                            "-c", "4096", "-ngl", ngl, "-t", os.environ.get("PORT_THREADS", "8")],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        outs = []
        for ids in ids_list:
            body = {"prompt": ids, "n_predict": MAX_NEW, "temperature": SAMPLING["temperature"],
                    "top_p": SAMPLING["top_p"], "repeat_penalty": SAMPLING["repetition_penalty"],
                    "top_k": 0, "min_p": 0, "return_tokens": True, "cache_prompt": False,
                    "stop": ["<custom_token_2>"]}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
            outs.append(r.get("tokens") or [])
        return outs
    finally:
        srv.terminate()
        srv.wait()


def gen_onnx(root, variant, ids_list):
    import onnxruntime_genai as og
    model = og.Model(str(root / "onnx-genai" / variant))
    outs = []
    for ids in ids_list:
        p = og.GeneratorParams(model)
        p.set_search_options(do_sample=True, max_length=len(ids) + MAX_NEW, temperature=SAMPLING["temperature"],
                             top_p=SAMPLING["top_p"], repetition_penalty=SAMPLING["repetition_penalty"])
        g = og.Generator(model, p)
        g.append_tokens(ids)
        new = []
        while not g.is_done() and len(new) < MAX_NEW:
            g.generate_next_token()
            t = int(g.get_next_tokens()[0])
            new.append(t)
            if t == 128258:
                break
        outs.append(new)
    return outs


def gen_openvino(root, variant, ids_list):
    import torch
    from optimum.intel import OVModelForCausalLM
    m = OVModelForCausalLM.from_pretrained(root / "openvino" / variant)
    outs = []
    for ids in ids_list:
        g = m.generate(torch.tensor([ids]), max_new_tokens=MAX_NEW, do_sample=True,
                       eos_token_id=128258, pad_token_id=128263, **SAMPLING)
        outs.append(g[0, len(ids):].tolist())
    return outs


def gen_mlx(root, variant, ids_list):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    model, tok = load(str(root / "mlx" / variant))
    outs = []
    for ids in ids_list:
        text_tokens = []
        from mlx_lm import stream_generate
        for resp in stream_generate(model, tok, ids, max_tokens=MAX_NEW,
                                    sampler=make_sampler(temp=SAMPLING["temperature"], top_p=SAMPLING["top_p"]),
                                    logits_processors=make_logits_processors(
                                        repetition_penalty=SAMPLING["repetition_penalty"])):
            text_tokens.append(resp.token)
            if resp.token == 128258:
                break
        outs.append(text_tokens)
    return outs


RUNTIMES = {"torch": gen_torch, "llamacpp": gen_llamacpp, "onnx": gen_onnx,
            "openvino": gen_openvino, "mlx": gen_mlx}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--runtime", required=True, choices=sorted(RUNTIMES))
    ap.add_argument("--variant", default="bf16")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    os.environ.setdefault("TTS_JUDGE", "mms-1b-all")   # read when tts_eval is imported
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    import finetune_orpheus as fo
    import tts_eval
    from porting.metrics import normalise_text
    import jiwer

    name = args.repo.split("/")[-1]
    root = ART / name
    lang, rows = prompts(args.repo, args.n)
    t0 = time.perf_counter()
    gens = RUNTIMES[args.runtime](root, args.variant, [ids for _, ids in rows])
    wall = time.perf_counter() - t0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    snac = fo.load_snac(dev)
    out_dir = root / "tts" / f"{args.runtime}-{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs, secs = [], 0.0
    for (r, _), g in zip(rows, gens):
        wav = fo.decode_tokens(snac, cut(g), dev)
        sf.write(out_dir / f"{r['i']:02d}.wav", wav, fo.SNAC_SR)
        wavs.append(wav)
        secs += len(wav) / fo.SNAC_SR
    del snac

    transcribe = tts_eval.make_transcriber(lang, dev)          # takes 16 kHz audio
    hyps = [transcribe(resample_poly(w, 2, 3).astype(np.float32)) if len(w) else "" for w in wavs]
    refs = [normalise_text(r["text"]) for r, _ in rows]
    hyps_n = [normalise_text(h) for h in hyps]
    host = os.environ.get("PORT_HOST")
    label = f"{args.runtime}@{host}" if host else args.runtime
    res = {"repo": args.repo, "runtime": label, "variant": args.variant, "lang": lang,
           "sentences": len(rows), "judge": "facebook/mms-1b-all",
           "cer": jiwer.cer(refs, [h or " " for h in hyps_n]),
           "audio_seconds": round(secs, 1), "wall_seconds": round(wall, 1),
           "tokens_per_second": round(sum(len(g) for g in gens) / max(wall, 1e-9), 1),
           "empty_outputs": sum(1 for w in wavs if len(w) == 0), "hyps": hyps}
    RESULTS.joinpath(name).mkdir(parents=True, exist_ok=True)
    (RESULTS / name / f"orpheus-{label}-{args.variant}.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    print(f"{name} {args.runtime}/{args.variant}: judge CER {100 * res['cer']:.1f}% over {len(rows)} sentences, "
          f"{res['tokens_per_second']} tok/s, audio {res['audio_seconds']} s")


if __name__ == "__main__":
    main()
