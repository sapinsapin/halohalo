"""Merge the completed rows of one or more remote ASR queue roots into an immutable manifest, audit, then export.

Handles both batch states the queues leave behind:
* complete batches: curation.asr_queue.confirmed_complete, then the full offline audit;
* quarantined batches (curation.asr_quarantine_queue status quarantined_finalize_timeout, or an asr_queue batch
  stopped_after_error whose single failure classifies as a plain finalize timeout): the completed rows must
  be the leading prefix of the frozen input in session order, the next input row must be the quarantined row, and
  the offline audit runs against a derived subset input holding exactly those leading rows.
Batches in any other state are skipped and listed. Excludes nothing silently: any ID or audio hash overlap with
the given published manifests, or within the merge, stops the job. Makes no ASR or network requests.

Usage (on dwnd-01, cwd /home/iotim/chika-curation, PYTHONPATH=.):
  python merge_export_queue.py OUTPUT_DIR --queue QROOT [--queue QROOT ...] [--published MANIFEST ...]
"""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path
from curation.asr_queue import confirmed_complete
from curation.asr_quarantine_queue import classify_failure
from curation.audit_asr import audit
from curation.schema import digest, load_manifest

p = argparse.ArgumentParser()
p.add_argument('output')
p.add_argument('--queue', action='append', required=True)
p.add_argument('--published', action='append', default=[])
p.add_argument('--python', default='/home/iotim/chika-curation/.venv-compute/bin/python')
a = p.parse_args()
root = Path(a.output).resolve()
if root.exists():
    raise SystemExit('output exists; outputs are immutable')
root.mkdir()
(root / 'subsets').mkdir()
s = {'pid': os.getpid(), 'status': 'running', 'stage': 'auditing', 'started_at': time.time()}


def save():
    t = root / 'worker.tmp'
    t.write_text(json.dumps(s))
    t.replace(root / 'worker.json')


save()
try:
    reports, rows, skipped = [], [], []
    for q in map(Path, a.queue):
        for b in sorted(d for d in q.iterdir() if d.is_dir() and d.name.startswith('batch-')):
            w = json.loads((b / 'worker.json').read_text())
            inputs = load_manifest(b / 'input.jsonl')
            if w.get('status') == 'complete':
                confirmed_complete(b, len(inputs))
                r = audit(b / 'input.jsonl', b / 'asr/manifest.jsonl', b / 'asr')
            elif w.get('status') == 'quarantined_finalize_timeout' or (
                    w.get('status') == 'stopped_after_error' and classify_failure(b) is not None):
                done = load_manifest(b / 'asr/manifest.jsonl') if (b / 'asr/manifest.jsonl').exists() else []
                summary = json.loads((b / 'asr/summary.json').read_text())
                failed_id = summary['failed'][0]['id']
                order = [r['id'] for r in inputs]
                if [r['id'] for r in done] != order[:len(done)] or order[len(done)] != failed_id:
                    raise RuntimeError(f'{b}: completed rows are not the input prefix before the quarantined row')
                if not done:
                    skipped.append({'batch': str(b), 'reason': 'quarantined on first row'})
                    continue
                sub = root / 'subsets' / f'{q.name}-{b.name}.jsonl'
                sub.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in inputs[:len(done)]))
                r = audit(sub, b / 'asr/manifest.jsonl', b / 'asr')
                r['partial_of'] = len(inputs)
            else:
                skipped.append({'batch': str(b), 'reason': f"status {w.get('status')}"})
                continue
            r['batch'] = str(b)
            reports.append(r)
            rows += load_manifest(b / 'asr/manifest.jsonl')
    ids = [r['id'] for r in rows]
    hashes = [r['audio_sha256'] for r in rows]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise RuntimeError('duplicate id or audio within merge')
    for m in a.published:
        pub = load_manifest(m)
        if {r['id'] for r in pub} & set(ids) or {r['audio_sha256'] for r in pub} & set(hashes):
            raise RuntimeError(f'overlap with published manifest {m}')
    for r in rows:
        if r['schema_version'] != 'sapinsapin.speech.v1' or r['split'] != 'unassigned' or r['review_status'] != 'draft':
            raise RuntimeError('unexpected row state')
        if digest(r['audio_path']) != r['audio_sha256']:
            raise RuntimeError('audio hash mismatch')
    manifest = root / 'manifest.jsonl'
    t = root / 'manifest.tmp'
    t.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    t.replace(manifest)
    nonempty = [r for r in rows if r['sentence'].strip()]
    info = {'rows': len(rows), 'nonempty': len(nonempty), 'languages': sorted({r['language'] for r in rows}),
            'padded_seconds': round(sum(float(r['duration']) for r in rows), 3),
            'nonempty_hours': sum(float(r['duration']) for r in nonempty) / 3600,
            'manifest_sha256': digest(manifest), 'queues': a.queue, 'published_checked': a.published,
            'skipped_batches': skipped, 'offline_audits': reports}
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
