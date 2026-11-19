"""Audit and export the completed prefix of a stopped offline batch ASR run (curation.asr_batch), then export.

curation.asr_batch.audit only accepts a finished run. This applies the same per-row checks to every completed chunk
file of a run that was stopped on purpose: chunk IDs must equal the frozen plan's chunks in order, the input manifest
must still match the plan hash, and each row must keep its config/cache identity, re-hashed audio and non-ASR fields.
Unfinished chunks are left out and listed in audit.json; they are not transcribed here. No ASR or network requests.

Usage (dwnd-01, cwd /home/iotim/chika-curation, PYTHONPATH=.):
  python merge_export_partial_batch.py OUTPUT_DIR --input INPUT.jsonl --asr ASR_DIR [--published MANIFEST ...]
"""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path
from curation.asr_batch import cache_key, local_path, sha
from curation.schema import digest, load_manifest

p = argparse.ArgumentParser()
p.add_argument('output')
p.add_argument('--input', required=True)
p.add_argument('--asr', required=True)
p.add_argument('--published', action='append', default=[])
p.add_argument('--shard-size', type=int, default=1000)
p.add_argument('--python', default='/home/iotim/chika-curation/.venv-compute/bin/python')
a = p.parse_args()
root = Path(a.output).resolve()
if root.exists():
    raise SystemExit('output exists; outputs are immutable')
root.mkdir()
s = {'pid': os.getpid(), 'status': 'running', 'stage': 'auditing', 'started_at': time.time()}


def save():
    t = root / 'worker.tmp'
    t.write_text(json.dumps(s))
    t.replace(root / 'worker.json')


save()
try:
    asr = Path(a.asr)
    worker = json.loads((asr / 'worker.json').read_text())
    if worker.get('pid') and Path(f"/proc/{worker['pid']}").exists():
        raise RuntimeError('batch worker process is still alive')
    plan = json.loads((asr / 'plan.json').read_text())
    if digest(a.input) != plan['input_sha256']:
        raise RuntimeError('input manifest differs from the frozen plan')
    config = plan['config']
    if plan['config_sha256'] != sha(config):
        raise RuntimeError('plan config hash mismatch')
    source = {r['id']: r for r in load_manifest(a.input)}
    asr_fields = {'sentence', 'num_words', 'asr_status', 'review_status', 'llm_status', 'provenance', 'audio_review'}
    rows, done, missing = [], [], []
    for n, chunk in enumerate(plan['chunks']):
        path = asr / 'chunks' / f'chunk-{n:05d}.jsonl'
        if not path.exists():
            missing.append(n)
            continue
        part = load_manifest(path)
        if [r['id'] for r in part] != chunk:
            raise RuntimeError(f'{path.name} does not match the frozen plan')
        done.append(n)
        rows.extend(part)
    for r in rows:
        src = source[r['id']]
        pv = r['provenance']
        if pv['asr_config'] != config or pv['asr_config_sha256'] != sha(config) or pv['asr_cache_key'] != cache_key(src, config):
            raise RuntimeError(f"config or cache identity mismatch for {r['id']}")
        if pv['asr_audio_sha256'] != src['audio_sha256'] or digest(local_path(r['audio_path'])) != src['audio_sha256']:
            raise RuntimeError(f"audio mismatch for {r['id']}")
        if {k: v for k, v in r.items() if k not in asr_fields} != {k: v for k, v in src.items() if k not in asr_fields}:
            raise RuntimeError(f"non-ASR fields changed for {r['id']}")
        if {k: v for k, v in pv.items() if not k.startswith('asr_')} != {k: v for k, v in (src.get('provenance') or {}).items() if not k.startswith('asr_')}:
            raise RuntimeError(f"provenance changed for {r['id']}")
        if r['num_words'] != len(r['sentence'].split()) or r['asr_status'] != ('transcribed' if r['sentence'] else 'empty'):
            raise RuntimeError(f"inconsistent ASR fields for {r['id']}")
        if r['review_status'] != 'draft' or config.get('engine') != 'local-batch':
            raise RuntimeError('unexpected row state')
    ids, hashes = [r['id'] for r in rows], [r['audio_sha256'] for r in rows]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise RuntimeError('duplicate id or audio within output')
    for m in a.published:
        pub = load_manifest(m)
        if {r['id'] for r in pub} & set(ids) or {r['audio_sha256'] for r in pub} & set(hashes):
            raise RuntimeError(f'overlap with published manifest {m}')
    manifest = root / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    nonempty = [r for r in rows if r['sentence'].strip()]
    info = {'rows': len(rows), 'nonempty': len(nonempty), 'languages': sorted({r['language'] for r in rows}),
            'padded_seconds': round(sum(float(r['duration']) for r in rows), 3),
            'nonempty_hours': sum(float(r['duration']) for r in nonempty) / 3600,
            'manifest_sha256': digest(manifest), 'input': a.input, 'input_sha256': plan['input_sha256'],
            'asr': a.asr, 'config_sha256': plan['config_sha256'], 'config': config,
            'scope': 'completed-chunk prefix of a stopped batch run',
            'chunks_total': len(plan['chunks']), 'chunks_included': len(done), 'chunks_not_transcribed': missing,
            'rows_not_transcribed': sum(len(plan['chunks'][n]) for n in missing),
            'not_attempted_over_30s': len(plan['not_attempted_over_30s']),
            'temperature_fallbacks': sum(bool(r['provenance'].get('asr_decode', {}).get('temperature')) for r in rows),
            'repetition_unresolved': sum(bool(r['provenance'].get('asr_decode', {}).get('repetition_unresolved')) for r in rows),
            'published_checked': a.published, 'worker_at_merge': worker, 'network_requests': 0}
    (root / 'audit.json').write_text(json.dumps(info, indent=2))
    if shutil.disk_usage(root).free < 100 * 1024 ** 3:
        raise RuntimeError('100 GiB floor')
    s['stage'] = 'export'
    save()
    subprocess.run(['nice', '-n', '10', a.python, '-m', 'curation.cli', 'export', '--manifest', str(manifest),
                    '--output', str(root / 'export'), '--profile', 'curation', '--shard-size', str(a.shard_size)], check=True)
    if digest(manifest) != info['manifest_sha256']:
        raise RuntimeError('manifest changed during export')
    s.update(status='complete', stage='complete', rows=len(rows), finished_at=time.time())
    save()
except BaseException as e:
    s.update(status='stopped_for_inspection', error=f'{type(e).__name__}: {e}'[:400], finished_at=time.time())
    save()
    raise
