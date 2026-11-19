"""Transcribe the disjoint PLD gold set with our per-language Whisper fine-tunes (curation.asr_batch, batch fallback config).

Writes one hypotheses file per model under OUT_DIR/ours/<lang>.jsonl ({id, hypothesis}); scoring is done by
score_gold.py so every system shares one normaliser. Holds the shared GPU lock only while a model is loaded.

Usage (dwnd-01, cwd /home/iotim/chika-curation): PYTHONPATH=. .venv-batchasr/bin/python eval_gold_ours.py GOLD_DIR
"""
import fcntl, json, sys, time
from pathlib import Path
import soundfile as sf
from curation import asr_batch as ab

MODELS = {'fil': ('sapinsapin/whisper-large-v3-pld-fil-norm', 'ee5b4f0cd1af616b38f742ae05b644d6783b52fe'),
          'hil': ('sapinsapin/whisper-large-v3-pld-hil-norm', '7a695b0f737b3e7cec482c29f0d12c2126cb4f4c'),
          'ceb': ('sapinsapin/whisper-large-v3-pld-ceb-norm', '109c7f42e40bfa071d53270115734dc2d88fb521'),
          'bcl': ('sapinsapin/whisper-large-v3-pld-bcl-norm', 'a35847d1db75e5c72409be4cc7f8b03775b45fb7'),
          'ilo': ('sapinsapin/whisper-large-v3-pld-ilo-norm', '00f3c4812849d7abfc0bb3519741b466457368ab')}
gold = Path(sys.argv[1])
(gold / 'ours').mkdir(exist_ok=True)
for lang, (repo, rev) in MODELS.items():
    refs = [json.loads(l) for l in (gold / lang / 'refs.jsonl').read_text().splitlines()]
    audios = [sf.read(gold / r['file'], dtype='float32')[0] for r in refs]
    cfg = ab.make_config(repo, rev, 'bik' if lang == 'bcl' else lang)
    with open(ab.GPU_LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = ab.WhisperTranscriber(cfg)
        t = time.monotonic()
        out = []
        for b in range(0, len(audios), 16):
            out += model(audios[b:b + 16])
        wall = time.monotonic() - t
        del model
    rows = [{'id': r['id'], 'hypothesis': o[0] if isinstance(o, tuple) else o} for r, o in zip(refs, out)]
    (gold / 'ours' / f'{lang}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    print(lang, repo, f'{wall:.1f}s', flush=True)
