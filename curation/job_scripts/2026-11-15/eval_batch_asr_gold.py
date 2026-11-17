"""Score curation.asr_batch on the AWS T4 runtime's gold gate sets (50 PLD clips per language). No network requests.

Gate sets (nyo-serve data/asr_pld/fil.npz sha 689c44b9..., data/asr_pld_hil_all/hil.npz sha 30e4a0ad..., the files the
T4 registry gates on); text normalised with nyo-serve modules/asr_fw/textnorm.normalise_text (copied below verbatim).
Decodes each set greedy-only and with the batch temperature fallback, under the shared GPU lock.

Usage (dwnd-01): HF_HUB_OFFLINE=1 PYTHONPATH=. .venv-batchasr/bin/python eval_batch_asr_gold.py FIL_NPZ HIL_NPZ OUT.json
"""
import fcntl, hashlib, json, re, sys, time, unicodedata
import jiwer
import numpy as np
from curation import asr_batch as ab

MODELS = {'fil': ('sapinsapin/whisper-large-v3-pld-fil-norm', 'ee5b4f0cd1af616b38f742ae05b644d6783b52fe', '689c44b991f9'),
          'hil': ('sapinsapin/whisper-large-v3-pld-hil-norm', '7a695b0f737b3e7cec482c29f0d12c2126cb4f4c', '30e4a0adbb30')}


def normalise_text(text):  # nyo-serve modules/asr_fw/textnorm.py
    t = unicodedata.normalize("NFD", (text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = unicodedata.normalize("NFC", t).replace("’", "'").replace("‘", "'")
    return " ".join(re.sub(r"[^\w\s']", " ", t).split())


report = {}
for lang, path in (('fil', sys.argv[1]), ('hil', sys.argv[2])):
    sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    repo, rev, want = MODELS[lang]
    if not sha.startswith(want):
        raise SystemExit(f'{path} is not the gate set')
    z = np.load(path, allow_pickle=True)
    offs = np.concatenate([[0], np.cumsum(z['lens'])])
    audios = [z['audio'][offs[i]:offs[i + 1]].astype(np.float32) for i in range(len(z['lens']))]
    refs = [str(r) for r in z['refs_norm']]
    for label, cfg in (('greedy', ab.make_config(repo, rev, lang, compression_ratio_threshold=None, temperatures=())),
                       ('batch_fallback', ab.make_config(repo, rev, lang))):
        with open(ab.GPU_LOCK, 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            model = ab.WhisperTranscriber(cfg)
            t = time.monotonic()
            out = []
            for b in range(0, len(audios), 16):
                out += model(audios[b:b + 16])
            wall = time.monotonic() - t
            del model
        hyps = [normalise_text(t[0] if isinstance(t, tuple) else t) for t in out]
        report[f'{lang}_{label}'] = {
            'n': len(refs), 'audio_s': round(sum(map(len, audios)) / 16000, 2), 'wall_s': round(wall, 2),
            'wer_pct': round(100 * jiwer.wer(refs, hyps), 3), 'cer_pct': round(100 * jiwer.cer(refs, hyps), 3),
            'fallbacks': sum(bool(isinstance(t, tuple) and t[1].get('temperature')) for t in out),
            'unresolved': sum(bool(isinstance(t, tuple) and t[1].get('repetition_unresolved')) for t in out),
            'npz_sha256': sha, 'hyps': hyps}
        print(lang, label, {k: v for k, v in report[f'{lang}_{label}'].items() if k != 'hyps'}, flush=True)
open(sys.argv[3], 'w').write(json.dumps(report, indent=2, ensure_ascii=False))
