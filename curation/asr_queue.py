"""Run a frozen, bounded ASR plan sequentially; never retry an uncertain stage.

Plans use version=1, batches=[{root, rows, input_sha256}], where each root
contains input.jsonl and a worker.json with status=queued. Optional dependency
contains root and rows. Relative roots resolve against the plan's directory.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .review import atomic
from .schema import digest, load_manifest


def prepare_plan(path):
    path = Path(path).resolve()
    blob = path.read_bytes()
    plan = json.loads(blob)
    if plan.get('version') != 1 or not isinstance(plan.get('batches'), list) or not 1 <= len(plan['batches']) <= 20:
        raise ValueError('version 1 plan requires 1..20 batches')
    batches, roots, identifiers = [], set(), set()
    for item in plan['batches']:
        root = (path.parent / item['root']).resolve()
        if root in roots:
            raise ValueError('duplicate batch root')
        roots.add(root)
        if type(item.get('rows')) is not int or not 1 <= item['rows'] <= 10000:
            raise ValueError('batch rows must be 1..10000')
        manifest = root/'input.jsonl'
        if digest(manifest) != item.get('input_sha256'):
            raise ValueError('input manifest changed')
        rows = load_manifest(manifest)
        ids = {r['id'] for r in rows}
        if len(rows) != item['rows'] or len(ids) != len(rows) or identifiers & ids:
            raise ValueError('row count or duplicate batch identifiers')
        identifiers.update(ids)
        worker = json.loads((root/'worker.json').read_text())
        if worker.get('status') != 'queued' or (root/'asr').exists():
            raise ValueError('batch already started; inspect and reconcile before recovery')
        batches.append({**item, 'root':root})
    dependency = plan.get('dependency')
    if dependency is not None:
        dependency = {**dependency, 'root':(path.parent / dependency['root']).resolve()}
        if type(dependency.get('rows')) is not int or dependency['rows'] < 1 or dependency['root'] in roots:
            raise ValueError('invalid dependency')
    return batches, dependency, hashlib.sha256(blob).hexdigest()


def confirmed_complete(root, expected):
    worker = json.loads((root/'worker.json').read_text())
    summary = json.loads((root/'asr/summary.json').read_text())
    if worker.get('status') != 'complete' or worker.get('returncode') != 0:
        raise ValueError('worker did not complete successfully')
    if (summary.get('selected') != expected or summary.get('completed') != expected
            or summary.get('failed') != [] or summary.get('stopped_early') is not False):
        raise ValueError('incomplete ASR summary')
    rows = load_manifest(root/'asr/manifest.jsonl')
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise ValueError('incomplete or duplicate output rows')
    if any(r.get('asr_status') not in {'transcribed', 'empty'} for r in rows):
        raise ValueError('output contains unprocessed rows')
    source = root/'input.jsonl'
    if source.exists():
        inputs = {r['id']:r['audio_sha256'] for r in load_manifest(source)}
        if {r['id']:r['audio_sha256'] for r in rows} != inputs:
            raise ValueError('output does not match frozen input identities')


def dependency_state(item):
    root = item['root']
    worker = json.loads((root/'worker.json').read_text())
    if worker.get('status') == 'complete':
        confirmed_complete(root, item['rows'])
        return 'complete'
    if worker.get('status') != 'running' or type(worker.get('pid')) is not int:
        raise ValueError('dependency stopped; inspect before recovery')
    process = subprocess.run(['ps', '-p', str(worker['pid']), '-o', 'command='], capture_output=True, text=True, timeout=10)
    if process.returncode or root.name not in process.stdout:
        raise ValueError('dependency process absent or changed')
    return 'running'


def run(plan_path, state_root, key_file, language, model, min_free_gib=45, poll_seconds=30):
    if min_free_gib < 0 or poll_seconds < 30:
        raise ValueError('invalid disk floor or polling interval')
    state_root = Path(state_root).resolve(); state_root.mkdir(parents=True, exist_ok=True)
    with (state_root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batches, dependency, plan_sha256 = prepare_plan(plan_path)
        if state_root in {item['root'] for item in batches} or (dependency and state_root == dependency['root']):
            raise ValueError('queue state must be separate from batch and dependency roots')
        state = dict(pid=os.getpid(), status='waiting_for_dependency', started_at=time.time(),
                     plan_sha256=plan_sha256, completed_batches=0)
        atomic(state_root/'worker.json', state)
        try:
            while dependency and dependency_state(dependency) != 'complete':
                time.sleep(poll_seconds)
            for item in batches:
                root = item['root']
                with (root/'queue-launch.lock').open('a') as batch_lock:
                    fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if digest(plan_path) != state['plan_sha256'] or digest(root/'input.jsonl') != item['input_sha256']:
                        raise ValueError('frozen plan or input changed')
                    if shutil.disk_usage(root).free < min_free_gib*2**30:
                        raise ValueError('storage floor')
                    if json.loads((root/'worker.json').read_text()).get('status') != 'queued' or (root/'asr').exists():
                        raise ValueError('batch already started; inspect before recovery')
                    command = [sys.executable, '-m', 'curation.asr_session', '--manifest', str(root/'input.jsonl'),
                               '--output', str(root/'asr'), '--key-file', str(Path(key_file).resolve()),
                               '--language', language, '--model', model, '--max-items', str(item['rows']),
                               '--timeout', '60', '--pace', '0']
                    # Record intent BEFORE spawning: a crash must never leave a paid
                    # child behind while the batch still appears safe to launch.
                    worker = dict(status='launching', started_at=time.time(), rows=item['rows'], input_sha256=item['input_sha256'])
                    atomic(root/'worker.json', worker)
                    with (root/'worker.log').open('ab') as log:
                        process = subprocess.Popen(command, stdout=log, stderr=log)
                        worker.update(pid=process.pid, status='running'); atomic(root/'worker.json', worker)
                        state.update(status='running', batch=str(root)); atomic(state_root/'worker.json', state)
                        code = process.wait()
                    worker.update(status='complete' if code == 0 else 'stopped_after_error', returncode=code, finished_at=time.time())
                    atomic(root/'worker.json', worker)
                    if digest(plan_path) != state['plan_sha256'] or digest(root/'input.jsonl') != item['input_sha256']:
                        raise ValueError('frozen plan or input changed during stage')
                    confirmed_complete(root, item['rows'])
                    state['completed_batches'] += 1; atomic(state_root/'worker.json', state)
            state.update(status='complete', finished_at=time.time())
        except BaseException as exc:
            state.update(status='stopped_after_error', error_type=type(exc).__name__, finished_at=time.time())
            raise
        finally:
            atomic(state_root/'worker.json', state)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--state-dir', required=True)
    parser.add_argument('--key-file', required=True)
    parser.add_argument('--language', choices=['fil', 'hil'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--min-free-gib', type=float, default=45)
    args = parser.parse_args()
    run(args.plan, args.state_dir, args.key_file, args.language, args.model, args.min_free_gib)


if __name__ == '__main__':
    main()
