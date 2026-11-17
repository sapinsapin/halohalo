"""Benchmark curation.asr_batch against transcripts the streaming route already produced. No network requests.

Inputs are the frozen input.jsonl files of finished streaming batches; references are those batches' asr/manifest.jsonl.
Runs under the shared GPU lock (asr_batch takes it per chunk). Writes OUTPUT/bench.json.

Usage (dwnd-01, cwd /home/iotim/chika-curation):
  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv-batchasr/bin/python bench_batch_asr.py OUTPUT
"""
import json, sys, time
from pathlib import Path
import jiwer
import torch
from curation import asr_batch as ab
from curation.schema import load_manifest

O = Path('/home/iotim/chika-curation/curation/outputs')
MODELS = {'fil': ('sapinsapin/whisper-large-v3-pld-fil-norm', 'ee5b4f0cd1af616b38f742ae05b644d6783b52fe'),
          'hil': ('sapinsapin/whisper-large-v3-pld-hil-norm', '7a695b0f737b3e7cec482c29f0d12c2126cb4f4c')}
root = Path(sys.argv[1]).resolve()
if root.exists():
    raise SystemExit('output exists')
root.mkdir()


def streaming_batches(queue, names=None):
    out = []
    for b in sorted(p for p in (O / queue).iterdir() if p.is_dir() and p.name.startswith('batch-')):
        w = json.loads((b / 'worker.json').read_text())
        if w.get('status') == 'complete' and (names is None or b.name in names):
            out.append((b, w))
    return out


def build(lang, batches):
    rows, refs, wall, audio = [], {}, 0.0, 0.0
    for b, w in batches:
        inp = load_manifest(b / 'input.jsonl')
        done = {r['id']: r['sentence'] for r in load_manifest(b / 'asr/manifest.jsonl')}
        if set(done) != {r['id'] for r in inp}:
            raise RuntimeError(f'{b} is not fully transcribed')
        rows += inp
        refs.update(done)
        if w.get('started_at') and w.get('finished_at'):
            wall += float(w['finished_at']) - float(w['started_at'])
            audio += sum(float(r['duration']) for r in inp)
    path = root / f'{lang}-input.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    return path, rows, refs, {'wall_seconds': round(wall, 1), 'audio_seconds': round(audio, 1),
                              'realtime_factor': round(audio / wall, 2) if wall else None}


def agreement(rows, refs, out_dir):
    hyp = {r['id']: r['sentence'] for r in load_manifest(out_dir / 'manifest.jsonl')}
    ids = [r['id'] for r in rows if r['id'] in hyp]
    both = [i for i in ids if refs[i].strip() and hyp[i].strip()]
    norm = lambda s: ' '.join(s.lower().split())
    return {'rows': len(ids), 'exact_match': sum(norm(refs[i]) == norm(hyp[i]) for i in ids) / len(ids),
            'wer_vs_streaming': jiwer.wer([norm(refs[i]) for i in both], [norm(hyp[i]) for i in both]),
            'cer_vs_streaming': jiwer.cer([norm(refs[i]) for i in both], [norm(hyp[i]) for i in both]),
            'empty_streaming_only': sum(not refs[i].strip() and bool(hyp[i].strip()) for i in ids),
            'empty_batch_only': sum(bool(refs[i].strip()) and not hyp[i].strip() for i in ids),
            'examples': [{'streaming': refs[i], 'batch': hyp[i]} for i in ids if norm(refs[i]) != norm(hyp[i])][:12]}


report = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'runs': []}
fil_names = {f'batch-{n:03d}' for n in range(91, 101)}
for lang, queue, names in (('fil', 'remote-fil-asr-queue-v6', fil_names), ('hil', 'remote-hil-asr-queue-v3', None)):
    path, rows, refs, streaming = build(lang, streaming_batches(queue, names))
    config = ab.make_config(*MODELS[lang], lang)
    torch.cuda.reset_peak_memory_stats()
    t = time.monotonic()
    model = ab.WhisperTranscriber(config)
    load_seconds = time.monotonic() - t
    sizes = (1, 8, 16, 32) if lang == 'fil' else (16,)
    for bs in sizes:
        sub = path if bs == 16 else root / f'{lang}-input-256.jsonl'
        if bs != 16 and not sub.exists():
            sub.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows[:256]))
        out = root / f'{lang}-bs{bs}'
        ab.run(sub, out, config, batch_size=bs, chunk_rows=256, transcriber=model)
        w = json.loads((out / 'worker.json').read_text())
        run = {'language': lang, 'batch_size': bs, 'rows': w['rows_done'], 'audio_seconds': w['audio_seconds'],
               'compute_seconds': w['compute_seconds'],
               'realtime_factor': round(w['audio_seconds'] / w['compute_seconds'], 1),
               'audit': ab.audit(sub, out), 'agreement': agreement(rows, refs, out)}
        if bs == 16:
            run['streaming_same_rows'] = streaming
        report['runs'].append(run)
        report[f'{lang}_model_load_seconds'] = round(load_seconds, 1)
        report[f'{lang}_peak_gpu_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        (root / 'bench.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    del model
    torch.cuda.empty_cache()
print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2))
for r in report['runs']:
    print(r['language'], r['batch_size'], r['rows'], r['realtime_factor'], round(r['agreement']['wer_vs_streaming'], 4),
          round(r['agreement']['exact_match'], 3), r.get('streaming_same_rows'))
