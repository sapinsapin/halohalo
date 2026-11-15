"""Audit a finished offline batch ASR output (curation.asr_batch) on dwnd-01, check overlap, then export.

Runs the batch audit (frozen plan, exact coverage, config/cache identity, unchanged non-ASR fields, audio re-hashed at
the original paths on this host), refuses any ID or audio overlap with the given published manifests or within the
output, and exports PLD-style Parquet shards. Works for outputs computed on reth5et too: copy that host's ASR output
directory back first (rows keep the original dwnd-01 audio_path). Makes no ASR or network requests.

Usage (dwnd-01, cwd /home/iotim/chika-curation, PYTHONPATH=.):
  python merge_export_batch.py OUTPUT_DIR --input INPUT.jsonl --asr ASR_DIR [--published MANIFEST ...]
"""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path
from curation.asr_batch import audit
from curation.schema import digest, load_manifest

p = argparse.ArgumentParser()
p.add_argument('output')
p.add_argument('--input', required=True)
p.add_argument('--asr', required=True)
p.add_argument('--published', action='append', default=[])
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
    report = audit(a.input, a.asr)
    rows = load_manifest(Path(a.asr) / 'manifest.jsonl')
    ids, hashes = [r['id'] for r in rows], [r['audio_sha256'] for r in rows]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise RuntimeError('duplicate id or audio within output')
    for m in a.published:
        pub = load_manifest(m)
        if {r['id'] for r in pub} & set(ids) or {r['audio_sha256'] for r in pub} & set(hashes):
            raise RuntimeError(f'overlap with published manifest {m}')
    if any(r['provenance']['asr_config']['engine'] != 'local-batch' or r['review_status'] != 'draft' for r in rows):
        raise RuntimeError('unexpected row state')
    manifest = root / 'manifest.jsonl'
    shutil.copyfile(Path(a.asr) / 'manifest.jsonl', manifest)
    if digest(manifest) != report['manifest_sha256']:
        raise RuntimeError('manifest changed after audit')
    nonempty = [r for r in rows if r['sentence'].strip()]
    summary = json.loads((Path(a.asr) / 'summary.json').read_text())
    info = {'rows': len(rows), 'nonempty': len(nonempty), 'languages': sorted({r['language'] for r in rows}),
            'padded_seconds': round(sum(float(r['duration']) for r in rows), 3),
            'nonempty_hours': sum(float(r['duration']) for r in nonempty) / 3600,
            'manifest_sha256': report['manifest_sha256'], 'input': a.input, 'asr': a.asr,
            'published_checked': a.published, 'batch_audit': report,
            'temperature_fallbacks': summary['temperature_fallbacks'],
            'repetition_unresolved': len(summary['repetition_unresolved']), 'config': summary['config']}
    (root / 'audit.json').write_text(json.dumps(info, indent=2))
    if shutil.disk_usage(root).free < 100 * 1024 ** 3:
        raise RuntimeError('100 GiB floor')
    s['stage'] = 'export'
    save()
    subprocess.run(['nice', '-n', '10', a.python, '-m', 'curation.cli', 'export', '--manifest', str(manifest),
                    '--output', str(root / 'export'), '--profile', 'curation', '--shard-size', '100'], check=True)
    if digest(manifest) != info['manifest_sha256']:
        raise RuntimeError('manifest changed during export')
    s.update(status='complete', stage='complete', rows=len(rows), finished_at=time.time())
    save()
except BaseException as e:
    s.update(status='stopped_for_inspection', error=f'{type(e).__name__}: {e}'[:400], finished_at=time.time())
    save()
    raise
