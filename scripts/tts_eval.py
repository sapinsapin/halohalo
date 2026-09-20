"""
TTS eval harness for the PLD languages — one table for every model we can run.

Plan item S1 in docs/pld_models_plan.md. Every model synthesizes the same
frozen sentences per language; each output is re-transcribed by our own
whisper-small-pld-{lang} and scored on CER, plus ECAPA speaker similarity to
the human reference recording and a duration ratio (catches truncation and
babble). The judge's CER on the human recordings is reported as the floor
no model can beat.

Runs on CPU by default so it can sit beside a training job on the GPU.

  python scripts/tts_eval.py --stage sentences                 # freeze 50 sentences/lang from the Hub test split
  python scripts/tts_eval.py --stage synth --model speecht5 --model mms
  python scripts/tts_eval.py --stage score --model speecht5 --model mms
  python scripts/tts_eval.py --stage table

Models: speecht5 (sapinsapin/speecht5_tts-pld-{lang}), mms (facebook/mms-tts-*),
orpheus:<adapter dir or repo> (LoRA on the Orpheus base; needs --device cuda).
Outputs land in $FINETUNE_DIR/tts_eval/.
"""

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

SR = 16000
ORG = "sapinsapin"
WORK = Path(os.environ.get("FINETUNE_DIR", ROOT / "finetune_runs")) / "tts_eval"
LANGS = ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]
MMS_CODE = {"fil": "tgl", "tsg": None}          # no MMS-TTS for Tausug; tgl stands in for fil
WHISPER_LANG = {"eng": "english"}               # everything else trains under <|tl|>
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and diacritics. PLD prompts mark stress
    inconsistently (únsa / unsa) and the ASR judge never emits accents, so
    scoring with accents would charge every model for the prompt's typography."""
    import unicodedata
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", PUNCT.sub(" ", text)).strip()


def manifest() -> list[dict]:
    path = WORK / "manifest.json"
    if not path.exists():
        sys.exit("no manifest — run --stage sentences first")
    return json.loads(path.read_text())


def read_wav(path: Path) -> np.ndarray:
    import librosa
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


# --------------------------------------------------------------------------- sentences

def stage_sentences(per_lang: int):
    """Freeze the eval sentences: read speech, 4-15 words, 1-12 s, distinct
    text, from the Hub test split. PLD prompts repeat across speakers, so
    these sentences are not unseen text — they are unseen *recordings*; the
    plan says so and the table must too."""
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(f"{ORG}/pld", split="test", streaming=True,
                      token=os.environ.get("HF_TOKEN"))
    ds = ds.cast_column("audio", Audio(decode=False))   # decode only what we keep
    picked = {l: [] for l in LANGS}
    seen = {l: set() for l in LANGS}
    for row in ds:
        l = row["language"]
        if l not in picked or len(picked[l]) >= per_lang:
            if all(len(v) >= per_lang for v in picked.values()):
                break
            continue
        if row["speech_type"] != "read" or row["text_is_prompt"]:
            continue
        if not (4 <= row["num_words"] <= 15 and 1.0 <= row["duration"] <= 12.0):
            continue
        text = row["sentence"].strip()
        if len(normalize(text)) < 3 or re.search(r"\d", text) or normalize(text) in seen[l]:
            continue
        seen[l].add(normalize(text))
        i = len(picked[l])
        ref_dir = WORK / "ref" / l
        ref_dir.mkdir(parents=True, exist_ok=True)
        wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        ref = ref_dir / f"{i:02d}.wav"
        sf.write(ref, wav, sr)
        picked[l].append({"lang": l, "i": i, "text": text,
                          "speaker_id": row["speaker_id"], "ref": str(ref),
                          "duration": row["duration"]})
        print(f"  {l} {i:02d} {row['speaker_id']} :: {text}", flush=True)
    rows = [r for l in LANGS for r in picked[l]]
    (WORK / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print({l: len(v) for l, v in picked.items()})


# --------------------------------------------------------------------------- synth

def synth_speecht5(rows, device):
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
    from finetune_tts import build_speaker_embedder

    embedder = build_speaker_embedder()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["lang"], []).append(r)
    for l, lrows in by_lang.items():
        repo = f"{ORG}/speecht5_tts-pld-{l}"
        proc = SpeechT5Processor.from_pretrained(repo)
        model = SpeechT5ForTextToSpeech.from_pretrained(repo).to(device).eval()
        for r in lrows:
            out = WORK / "out" / "speecht5" / l / f"{r['i']:02d}.wav"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            ref = torch.tensor(read_wav(Path(r["ref"]))).unsqueeze(0)
            with torch.no_grad():
                emb = torch.nn.functional.normalize(embedder.encode_batch(ref), dim=2)
                emb = emb.squeeze(1).to(device)            # (1, 512), the clip's own voice
                ids = proc(text=r["text"].replace("’", "'"), return_tensors="pt")["input_ids"].to(device)
                speech = model.generate_speech(ids, emb, vocoder=vocoder)
            _write(out, speech.cpu().numpy(), SR)
        print(f"  speecht5 {l}: done", flush=True)


def synth_mms(rows, device):
    import torch
    from transformers import AutoTokenizer, VitsModel

    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["lang"], []).append(r)
    for l, lrows in by_lang.items():
        code = MMS_CODE.get(l, l)
        if code is None:
            print(f"  mms {l}: no checkpoint, skipped", flush=True)
            continue
        repo = f"facebook/mms-tts-{code}"
        tok = AutoTokenizer.from_pretrained(repo)
        model = VitsModel.from_pretrained(repo).to(device).eval()
        for r in lrows:
            out = WORK / "out" / "mms" / l / f"{r['i']:02d}.wav"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            inputs = tok(r["text"].lower(), return_tensors="pt").to(device)
            with torch.no_grad():
                wav = model(**inputs).waveform[0].cpu().numpy()
            _write(out, wav, model.config.sampling_rate)
        print(f"  mms {l}: done ({repo})", flush=True)


def adapter_name(adapter: str) -> str:
    """Name a run after its run dir, not its leaf.

    Every arm's adapter lives at <run>/final, so naming by the leaf makes all
    three arms of an ablation "orpheus_final" and they overwrite each other's
    results.
    """
    q = Path(adapter)
    name = q.parent.name if q.name == "final" else q.name
    return name if name.startswith("orpheus") else "orpheus_" + name


def adapter_spec(adapter: str) -> tuple[str, str]:
    """(units, language) from a run dir named orpheus_<units>_<dataset>_<lang>.

    Both matter at synthesis time. The units decide how the text is spelled out
    before it reaches the tokenizer, and an arm evaluated on a frontend it was
    not trained on is measuring the wrong thing. The language decides which
    sentences to synthesize at all: these adapters are per-language, so running
    them over all ten languages is ten times the work for nine languages of
    noise.
    """
    q = Path(adapter)
    name = q.parent.name if q.name == "final" else q.name
    parts = name.split("_")           # orpheus, <units>, <dataset>, <lang>
    units = parts[1] if len(parts) > 1 else "bpe"
    lang = parts[-1]
    return units, lang


def synth_orpheus(rows, adapter, device, batch_size=8):
    """LoRA adapter on the Orpheus base.

    bf16, not 4-bit: the adapters were trained against a bf16 base, and on a
    96 GB card quantising for inference only mismatches the adapter and slows
    generation down (measured at minutes per clip in 4-bit).

    Generation is batched. Prompts are left-padded so every sequence in a batch
    ends at the same position, which is what lets one `generate` call serve the
    whole batch: with right padding the model would continue from the padding
    instead of from the prompt.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import finetune_orpheus as fo

    units, lang = adapter_spec(adapter)
    rows = [r for r in rows if r["lang"] == lang]
    name = adapter_name(adapter)
    print(f"  {name}: {len(rows)} sentences, units={units}, lang={lang}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(fo.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        fo.BASE_MODEL, device_map={"": 0}, dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, adapter).eval()
    snac = fo.load_snac(device)
    pad_id = tok.pad_token_id or 128263

    todo = []
    for r in rows:
        out = WORK / "out" / name / r["lang"] / f"{r['i']:02d}.wav"
        if not out.exists():
            todo.append((r, out))

    for k in range(0, len(todo), batch_size):
        chunk = todo[k:k + batch_size]
        prompts = []
        for r, _ in chunk:
            # the same frontend the arm was trained with, or the arm is being
            # asked to read text in a notation it never saw
            text = fo.frontend_text(r["text"], units)
            ids = tok(f"{r['speaker_id']}: {text}",
                      add_special_tokens=False).input_ids
            prompts.append([fo.SOH] + ids + [fo.EOT, fo.EOH, fo.SOAI, fo.SOS])

        width = max(len(q) for q in prompts)
        input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(prompts), width), dtype=torch.long)
        for i, q in enumerate(prompts):       # left pad
            input_ids[i, width - len(q):] = torch.tensor(q, dtype=torch.long)
            attn[i, width - len(q):] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)

        with torch.inference_mode():
            gen = model.generate(input_ids, attention_mask=attn,
                                 max_new_tokens=1400, do_sample=True,
                                 temperature=0.6, top_p=0.9,
                                 repetition_penalty=1.1,
                                 eos_token_id=fo.EOS_SPEECH,
                                 pad_token_id=pad_id)

        for i, (_, out) in enumerate(chunk):
            out.parent.mkdir(parents=True, exist_ok=True)
            wav = fo.decode_tokens(snac, gen[i, width:].tolist(), device)
            _write(out, wav, fo.SNAC_SR)
        print(f"    {min(k + batch_size, len(todo))}/{len(todo)}", flush=True)

    print(f"  {name}: done", flush=True)
    return name


def _write(path: Path, wav: np.ndarray, sr: int):
    import soundfile as sf
    sf.write(path, np.asarray(wav, dtype=np.float32), sr)


def stage_synth(models, device):
    rows = manifest()
    for m in models:
        if m == "speecht5":
            synth_speecht5(rows, device)
        elif m == "mms":
            synth_mms(rows, device)
        elif m.startswith("orpheus:"):
            synth_orpheus(rows, m.split(":", 1)[1], device)
        else:
            sys.exit(f"unknown model {m!r}")


# --------------------------------------------------------------------------- score

def stage_score(models, device):
    import jiwer
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    rows = manifest()
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path(os.environ.get("HF_HOME", "~/.cache")).expanduser() / "speechbrain-ecapa"),
        run_opts={"device": device})
    results_path = WORK / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    names = ["reference"] + [m if not m.startswith("orpheus:") else
                             adapter_name(m.split(":", 1)[1]) for m in models]

    for l in LANGS:
        lrows = [r for r in rows if r["lang"] == l and len(normalize(r["text"])) >= 3]
        if not lrows:
            continue
        judge_id = judge_model(l)
        proc = WhisperProcessor.from_pretrained(judge_id)
        judge = WhisperForConditionalGeneration.from_pretrained(judge_id).to(device).eval()
        wl = WHISPER_LANG.get(l, "tagalog")

        def transcribe(wav):
            feats = proc(wav, sampling_rate=SR, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                ids = judge.generate(feats, language=wl, task="transcribe", max_new_tokens=200)
            return proc.batch_decode(ids, skip_special_tokens=True)[0]

        def embed(wav):
            with torch.no_grad():
                e = ecapa.encode_batch(torch.tensor(wav).unsqueeze(0).to(device))
            return torch.nn.functional.normalize(e.squeeze(), dim=0).cpu()

        for name in names:
            refs, hyps, sims, ratios, n = [], [], [], [], 0
            for r in lrows:
                path = Path(r["ref"]) if name == "reference" else WORK / "out" / name / l / f"{r['i']:02d}.wav"
                if not path.exists():
                    continue
                wav = read_wav(path)
                if len(wav) < SR // 10:
                    continue
                refs.append(normalize(r["text"]))
                hyps.append(normalize(transcribe(wav)))
                if name != "reference":
                    sims.append(float(torch.dot(embed(wav), embed(read_wav(Path(r["ref"]))))))
                    ratios.append(len(wav) / SR / r["duration"])
                n += 1
            if not n:
                continue
            results.setdefault(name, {})[l] = {
                "n": n,
                "cer": jiwer.cer(refs, hyps),
                "wer": jiwer.wer(refs, hyps),
                "spk_sim": float(np.mean(sims)) if sims else None,
                "dur_ratio": float(np.mean(ratios)) if ratios else None,
                "hyps": hyps,          # kept so the table can be re-normalised without re-transcribing
            }
            print(f"  {l} {name:24} n={n:2d} CER {results[name][l]['cer']*100:5.1f}%"
                  + (f"  spk {np.mean(sims):.3f}  dur×{np.mean(ratios):.2f}" if sims else ""), flush=True)
        results_path.write_text(json.dumps(results, indent=1))


def log_wandb(results: dict):
    """Version the scored table in W&B (project halohalo-tts) when a key is set;
    the JSON on disk stays the source of truth."""
    if not os.environ.get("WANDB_API_KEY"):
        return
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "halohalo-tts"),
                     name="tts_eval", job_type="eval", reinit=True)
    table = wandb.Table(columns=["model", "lang", "n", "cer", "wer", "spk_sim", "dur_ratio"])
    for name, per_lang in results.items():
        for l, m in per_lang.items():
            table.add_data(name, l, m["n"], m["cer"], m["wer"], m["spk_sim"], m["dur_ratio"])
    run.log({"tts_eval": table})
    run.finish()


def stage_table():
    results = json.loads((WORK / "results.json").read_text())
    log_wandb(results)
    names = ["reference"] + sorted(n for n in results if n != "reference")
    for metric, fmt in (("cer", lambda v: f"{v*100:.1f}"), ("spk_sim", lambda v: f"{v:.2f}")):
        print(f"\n**{metric}** (round-trip with the per-language judge printed above; reference = judge floor)\n")
        print("| model | " + " | ".join(LANGS) + " |")
        print("|---|" + "---|" * len(LANGS))
        for name in names:
            cells = []
            for l in LANGS:
                v = results.get(name, {}).get(l, {}).get(metric)
                cells.append(fmt(v) if v is not None else "—")
            print(f"| {name} | " + " | ".join(cells) + " |")


JUDGE = os.environ.get("TTS_JUDGE", "whisper-large-v3")
JUDGE_PINNED = "TTS_JUDGE" in os.environ
_JUDGE_CACHE: dict[str, str] = {}


def judge_model(lang: str) -> str:
    """The ASR model that re-transcribes synthesized speech.

    Prefers the bake-off winner, whisper-large-v3-pld-<lang>, because a weak
    judge charges TTS for its own transcription errors. But that model exists
    only for the languages the bake-off covered — ceb and pam — so every other
    language falls back to the whisper-small fleet, and the fallback is printed
    rather than silent: a table whose rows were judged by different models has
    to say so.

    Still our own models on our own corpus. docs/tts_sota_plan.md P5 keeps an
    independent judge (Omnilingual ASR) as the requirement before publication.
    TTS_JUDGE pins one judge for every language.
    """
    from huggingface_hub import repo_exists

    if lang in _JUDGE_CACHE:
        return _JUDGE_CACHE[lang]

    want = f"{ORG}/{JUDGE}-pld-{lang}"
    if JUDGE_PINNED or repo_exists(want, token=os.environ.get("HF_TOKEN")):
        _JUDGE_CACHE[lang] = want
    else:
        fallback = f"{ORG}/whisper-small-pld-{lang}"
        print(f"  judge: {want} does not exist, using {fallback} for {lang}")
        _JUDGE_CACHE[lang] = fallback
    return _JUDGE_CACHE[lang]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stage", choices=["sentences", "synth", "score", "table"], required=True)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--per-lang", type=int, default=50)
    ap.add_argument("--device", default="cpu", help="cpu (default, GPU is usually training) or cuda")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.stage == "sentences":
        stage_sentences(args.per_lang)
    elif args.stage == "synth":
        stage_synth(args.model or ["speecht5", "mms"], args.device)
    elif args.stage == "score":
        stage_score(args.model or ["speecht5", "mms"], args.device)
    else:
        stage_table()


if __name__ == "__main__":
    main()
