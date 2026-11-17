"""Merge completed Filipino ASR queues v4+v5 (4,000 rows) into one immutable manifest, audit, then export.

Runs on dwnd-01 from /home/iotim/chika-curation. Makes no ASR or network requests.
"""
import json, os, shutil, subprocess, time
from pathlib import Path
from curation.asr_queue import confirmed_complete
from curation.audit_asr import audit
from curation.schema import digest

base = Path('/home/iotim/chika-curation')
out = base / 'curation/outputs'
root = Path(__file__).resolve().parent
python = str(base / '.venv-compute/bin/python')
queues = ['remote-fil-asr-queue-v4', 'remote-fil-asr-queue-v5']
published = out / 'remote-fil-asr-2000-v3/manifest.jsonl'
s = {'pid': os.getpid(), 'status': 'running', 'stage': 'auditing_queues', 'started_at': time.time()}


def save():
    p = root / 'worker.tmp'
    p.write_text(json.dumps(s))
    p.replace(root / 'worker.json')


save()
try:
    reports, rows = [], []
    for q in queues:
        state = json.loads((out / q / 'state/worker.json').read_text())
        if state['status'] != 'complete':
            raise RuntimeError(f'{q} not complete')
        plan = json.loads((out / q / 'plan.json').read_text())
        for item in plan['batches']:
            b = Path(item['root'])
            confirmed_complete(b, item['rows'])
            r = audit(b / 'input.jsonl', b / 'asr/manifest.jsonl', b / 'asr')
            r['queue'] = q
            reports.append(r)
            rows += [json.loads(l) for l in (b / 'asr/manifest.jsonl').read_text().splitlines() if l.strip()]
    ids = [r['id'] for r in rows]
    hashes = [r['audio_sha256'] for r in rows]
    if len(rows) != 4000 or len(set(ids)) != 4000:
        raise RuntimeError(f'expected 4000 unique ids, got {len(rows)}/{len(set(ids))}')
    if len(set(hashes)) != len(hashes):
        raise RuntimeError('duplicate audio within v4+v5')
    pub = [json.loads(l) for l in published.read_text().splitlines() if l.strip()]
    if {r['id'] for r in pub} & set(ids) or {r['audio_sha256'] for r in pub} & set(hashes):
        raise RuntimeError('overlap with published fil_dwnd_asr_2000_v1')
    for r in rows:
        if r['schema_version'] != 'sapinsapin.speech.v1' or r['split'] != 'unassigned' or r['review_status'] != 'draft':
            raise RuntimeError('unexpected row state')
        if digest(r['audio_path']) != r['audio_sha256']:
            raise RuntimeError('audio hash mismatch')
    manifest = root / 'manifest.jsonl'
    tmp = root / 'manifest.tmp'
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    tmp.replace(manifest)
    nonempty = [r for r in rows if r['sentence'].strip()]
    a = {'rows': len(rows), 'nonempty': len(nonempty),
         'padded_seconds': round(sum(float(r['duration']) for r in rows), 3),
         'nonempty_hours': sum(float(r['duration']) for r in nonempty) / 3600,
         'manifest_sha256': digest(manifest), 'queues': queues,
         'excluded_overlap_with': str(published), 'offline_audits': reports}
    (root / 'audit.json').write_text(json.dumps(a, indent=2))
    if shutil.disk_usage(root).free < 100 * 1024 ** 3:
        raise RuntimeError('100 GiB floor')
    s['stage'] = 'export'
    save()
    subprocess.run(['nice', '-n', '10', python, '-m', 'curation.cli', 'export', '--manifest', str(manifest),
                    '--output', str(root / 'export'), '--profile', 'curation', '--shard-size', '100'], cwd=base, check=True)
    if digest(manifest) != a['manifest_sha256']:
        raise RuntimeError('manifest changed during export')
    s.update(status='complete', stage='complete', finished_at=time.time())
    save()
except BaseException as e:
    s.update(status='stopped_for_inspection', error=f'{type(e).__name__}: {e}', finished_at=time.time())
    save()
    raise
