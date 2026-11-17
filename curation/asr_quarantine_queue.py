"""Run small frozen ASR batches, each in a fresh session, quarantining server finalize timeouts instead of stopping.

For routes that intermittently return a non-fatal ``finalize_timeout`` (observed on Hiligaynon: the server's
3,000 ms finalize deadline). Every batch is one ``curation.asr_session`` run (one persistent session, zero retries,
stop on first error), exactly as ``curation.asr_queue`` launches it. The difference is only what happens after a
batch stops early:

* If the batch's single failure is a saved attempt whose wire events contain an ``error`` event with code
  ``finalize_timeout`` and no completed result, the row is appended to ``quarantine.jsonl`` and the queue moves on to
  the next batch after a pause. The quarantined request is **never replayed**: it may have been processed or charged.
  The batch's later rows stay unattempted and are listed in ``unattempted.jsonl`` for a future frozen selection.
* Any other failure (transport closure, protocol mismatch, storage floor, unknown error) stops the whole queue for
  inspection, as ``asr_queue`` would.

The queue also stops after ``--max-quarantined`` timeouts in total or ``--max-consecutive`` batches in a row that
quarantined their first row (the route is then degraded, not intermittently slow).

Plan format is ``asr_queue``'s version 1 (batches of {root, rows, input_sha256}, worker.json status=queued) but with
no 20-batch limit, so 25-row batches can cover a 2,000-row reservation.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .asr_queue import confirmed_complete
from .review import atomic
from .schema import digest, load_manifest

QUARANTINE_CODES = {'finalize_timeout'}


def prepare_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_bytes())
    if plan.get('version') != 1 or not isinstance(plan.get('batches'), list) or not plan['batches']:
        raise ValueError('version 1 plan with at least one batch required')
    batches, roots, ids = [], set(), set()
    for item in plan['batches']:
        root = (path.parent / item['root']).resolve()
        if root in roots:
            raise ValueError('duplicate batch root')
        roots.add(root)
        if type(item.get('rows')) is not int or not 1 <= item['rows'] <= 1000:
            raise ValueError('batch rows must be 1..1000')
        if digest(root / 'input.jsonl') != item.get('input_sha256'):
            raise ValueError('input manifest changed')
        rows = load_manifest(root / 'input.jsonl')
        batch_ids = {r['id'] for r in rows}
        if len(rows) != item['rows'] or len(batch_ids) != len(rows) or ids & batch_ids:
            raise ValueError('row count or duplicate identifiers')
        ids |= batch_ids
        batches.append({**item, 'root': root})
    return batches, digest(path)


def classify_failure(root):
    """Return a quarantine record if the batch stopped on exactly one finalize timeout, else None."""
    summary = json.loads((root / 'asr/summary.json').read_text())
    failed = summary.get('failed') or []
    if not summary.get('stopped_early') or len(failed) != 1:
        return None
    failed_attempts = []
    for f in sorted((root / 'asr').glob('*.attempt-*.json')):
        attempt = json.loads(f.read_text())
        if attempt.get('status') == 'failed':
            failed_attempts.append((f, attempt))
    if len(failed_attempts) != 1:
        return None
    f, attempt = failed_attempts[0]
    events = [e.get('event') for e in attempt.get('events') or [] if isinstance(e, dict)]
    events = [e for e in events if isinstance(e, dict)]
    codes = {e.get('code') for e in events if e.get('type') == 'error'}
    if not codes or not codes <= QUARANTINE_CODES:
        return None
    if any(e.get('type') in ('result', 'finalize_end') for e in events):
        return None  # a result arrived: this is not a plain timeout, inspect it instead
    row_id = failed[0]['id']
    inputs = {r['id']: r for r in load_manifest(root / 'input.jsonl')}
    if row_id not in inputs:
        return None
    return {'batch': str(root), 'id': row_id, 'audio_sha256': inputs[row_id]['audio_sha256'],
            'duration': inputs[row_id].get('duration'), 'attempt': f.name, 'codes': sorted(codes),
            'replay': 'never: request may have been processed or charged'}


def unattempted(root, quarantined_id):
    done = {r['id'] for r in load_manifest(root / 'asr/manifest.jsonl')} if (root / 'asr/manifest.jsonl').exists() else set()
    return [r for r in load_manifest(root / 'input.jsonl') if r['id'] not in done and r['id'] != quarantined_id]


def run(plan_path, state_root, key_file, language, model, min_free_gib=45, max_quarantined=20, max_consecutive=3,
        pause_seconds=20, launch=subprocess.Popen, sleep=time.sleep):
    if not math.isfinite(min_free_gib) or min_free_gib < 0 or max_quarantined < 1 or max_consecutive < 1:
        raise ValueError('invalid limits')
    state_root = Path(state_root).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batches, plan_sha256 = prepare_plan(plan_path)
        if state_root in {b['root'] for b in batches}:
            raise ValueError('queue state must be separate from batch roots')
        state = dict(pid=os.getpid(), status='running', started_at=time.time(), plan_sha256=plan_sha256,
                     completed_batches=0, quarantined=0, skipped_finished=0)
        atomic(state_root / 'worker.json', state)
        consecutive = 0
        try:
            for item in batches:
                root = item['root']
                worker = json.loads((root / 'worker.json').read_text())
                if worker.get('status') in ('complete', 'quarantined_finalize_timeout'):
                    state['skipped_finished'] += 1
                    continue
                if worker.get('status') != 'queued' or (root / 'asr').exists():
                    raise ValueError(f'batch {root.name} already started; inspect before recovery')
                if digest(plan_path) != plan_sha256 or digest(root / 'input.jsonl') != item['input_sha256']:
                    raise ValueError('frozen plan or input changed')
                if shutil.disk_usage(root).free < min_free_gib * 2**30:
                    raise ValueError('storage floor')
                command = [sys.executable, '-m', 'curation.asr_session', '--manifest', str(root / 'input.jsonl'),
                           '--output', str(root / 'asr'), '--key-file', str(Path(key_file).resolve()),
                           '--language', language, '--model', model, '--max-items', str(item['rows']),
                           '--timeout', '60', '--pace', '0', '--min-free-gib', str(min_free_gib)]
                worker = dict(status='launching', started_at=time.time(), rows=item['rows'], input_sha256=item['input_sha256'])
                atomic(root / 'worker.json', worker)
                with (root / 'worker.log').open('ab') as log:
                    process = launch(command, stdout=log, stderr=log)
                    worker.update(pid=process.pid, status='running')
                    atomic(root / 'worker.json', worker)
                    state.update(batch=str(root))
                    atomic(state_root / 'worker.json', state)
                    code = process.wait()
                worker.update(returncode=code, finished_at=time.time())
                if code == 0:
                    worker['status'] = 'complete'
                    atomic(root / 'worker.json', worker)
                    confirmed_complete(root, item['rows'])
                    state['completed_batches'] += 1
                    consecutive = 0
                    atomic(state_root / 'worker.json', state)
                    continue
                record = classify_failure(root)
                if record is None:
                    worker['status'] = 'stopped_after_error'
                    atomic(root / 'worker.json', worker)
                    raise RuntimeError(f'batch {root.name} stopped on a non-quarantinable failure')
                worker['status'] = 'quarantined_finalize_timeout'
                atomic(root / 'worker.json', worker)
                with (state_root / 'quarantine.jsonl').open('a') as q:
                    q.write(json.dumps(record) + '\n')
                left = unattempted(root, record['id'])
                with (state_root / 'unattempted.jsonl').open('a') as u:
                    u.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in left)
                state['quarantined'] += 1
                first_row = not (root / 'asr/manifest.jsonl').exists() or not load_manifest(root / 'asr/manifest.jsonl')
                consecutive = consecutive + 1 if first_row else 0
                atomic(state_root / 'worker.json', state)
                if state['quarantined'] >= max_quarantined:
                    raise RuntimeError('quarantine budget reached')
                if consecutive >= max_consecutive:
                    raise RuntimeError('route degraded: consecutive batches failed on their first row')
                sleep(pause_seconds)
            state.update(status='complete', finished_at=time.time())
        except BaseException as exc:
            state.update(status='stopped_after_error', error=f'{type(exc).__name__}: {exc}'[:300], finished_at=time.time())
            raise
        finally:
            atomic(state_root / 'worker.json', state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plan', required=True)
    p.add_argument('--state-dir', required=True)
    p.add_argument('--key-file', required=True)
    p.add_argument('--language', choices=['fil', 'hil'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--min-free-gib', type=float, default=45)
    p.add_argument('--max-quarantined', type=int, default=20)
    p.add_argument('--max-consecutive', type=int, default=3)
    p.add_argument('--pause-seconds', type=float, default=20)
    a = p.parse_args()
    run(a.plan, a.state_dir, a.key_file, a.language, a.model, a.min_free_gib, a.max_quarantined, a.max_consecutive,
        a.pause_seconds)


if __name__ == '__main__':
    main()
