"""Read-only status for an acquisition host; emits counts, never transcript text.

Run on the host: python -m curation.remote_status --root curation/outputs
This observes process identities and ledgers; it does not rehash audio or restart jobs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess


def read(path, errors):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append({'path': str(path), 'error_kind': type(exc).__name__})
        return {}


def observe(root):
    errors = []
    report = {'root': str(root.resolve())}
    # Recovery launches supersede historical launch PIDs; worker records are newest.
    state = {}
    for name in ('launch.json', 'recovery-launch.json', 'worker.json'):
        if (root / name).exists():
            state.update(read(root / name, errors))
    report['worker'] = {k: state[k] for k in
                        ('pid', 'status', 'stage', 'verified_windows', 'returncode', 'started_at', 'finished_at') if k in state}
    pid = state.get('pid')
    if isinstance(pid, int) and pid > 0:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                                capture_output=True, text=True, timeout=10)
        report['process_present'] = result.returncode == 0
        report['process_matches_root'] = result.returncode == 0 and str(root.resolve()) in result.stdout
        # Avoid recording command strings, which can contain credentials in other jobs.
    statuses = Counter()
    hours = 0.0
    for path in root.glob('*.json'):
        if path.name.endswith('.info.json'):
            continue
        row = read(path, errors)
        if isinstance(row, dict) and row.get('id') and row.get('source_id') and row.get('status'):
            statuses[row['status']] += 1
            if row['status'] == 'downloaded':
                hours += float(row.get('duration') or 0) / 3600
    if statuses:
        report['acquisition'] = {'statuses': dict(statuses), 'raw_hours': hours}
    db_path = root / 'state.sqlite'
    if db_path.exists():
        try:
            with sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
                groups = db.execute('SELECT status,count(*),sum(duration)/3600 FROM items GROUP BY status').fetchall()
                report['rss'] = [{'status': s, 'items': n, 'raw_hours': h or 0} for s, n, h in groups]
        except sqlite3.Error as exc:
            errors.append({'path': str(db_path), 'error_kind': type(exc).__name__})
    summaries = {}
    fields = ('windows', 'window_hours', 'input_windows', 'input_hours', 'segments',
              'candidate_hours', 'selected', 'completed', 'failed', 'manifest_rows',
              'stopped_early', 'stop_reason', 'transcribed_hours', 'verified_hours', 'empty')
    for name in ('summary.json', 'prepared/summary.json', 'vad/summary.json', 'responses/summary.json'):
        path = root / name
        if path.exists():
            value = read(path, errors)
            if isinstance(value, dict):
                summaries[name] = {k: len(value[k]) if isinstance(value[k], list) else value[k]
                                   for k in fields if k in value}
    if summaries:
        report['summaries'] = summaries
    report['observation_errors'] = errors
    return report


def snapshot(root):
    root = Path(root).resolve()
    candidates = set()
    for pattern in ('*/launch.json', '*/worker.json', '*/*/launch.json', '*/*/worker.json'):
        candidates.update(p.parent for p in root.glob(pattern))
    # ASR response cache may have no wrapper of its own.
    candidates.update(p.parent for p in root.glob('remote-*/summary.json'))
    return {'host_scope': str(root), 'updated_at': datetime.now(timezone.utc).isoformat(),
            'free_gib': shutil.disk_usage(root).free / 2**30,
            'jobs': [observe(p) for p in sorted(candidates)],
            'limitations': ['Ledger observations, not fresh media or backup verification.',
                           'Raw, prepared, VAD and ASR hours overlap: never sum stages.',
                           'Failed item states can be historical while recovery is running.',
                           'Missing process alone is not proof of successful completion.']}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.root), indent=2))


if __name__ == '__main__':
    main()
