"""Bounded RSS storage supervision; observation is the default.

python -m curation.supervise --once
python -m curation.supervise --apply

Only the two explicitly registered SQLite RSS roots are supported. Archive-source
manifests (including non-RSS Hiligaynon archives) need their own checkpoint and
worker adapters and are deliberately not restarted or evicted here. No workers
are killed. A global flock serializes supervisors; an inherited per-root launch
flock closes the check/spawn/crash window until a launched worker exits.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time

from . import hub, evict

BASE = Path(__file__).resolve().parent.parent
GIB = 1024 ** 3


@dataclass(frozen=True)
class Profile:
    root: Path
    namespace: str
    max_hours: float
    max_source_hours: float


PROFILES = (
    Profile(BASE/'curation/outputs/rss', 'hiligaynon-rss', 0, 0),
    Profile(BASE/'curation/outputs/taglish/shared', 'taglish-rss', 300, 50),
)


@contextmanager
def lock(path):
    """Close without explicit LOCK_UN: inherited descriptors retain the lock."""
    if path.is_symlink():
        raise ValueError('lock must not be a symlink')
    handle = path.open('a')
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield handle
    finally:
        handle.close()


def inventory(profile):
    root = profile.root
    if root.is_symlink() or (root/'state.sqlite').is_symlink():
        raise ValueError('root and ledger must not be symlinks')
    with sqlite3.connect((root/'state.sqlite').as_uri()+'?mode=ro', uri=True) as db:
        hours = db.execute("SELECT coalesce(sum(duration),0)/3600 FROM items WHERE status IN ('downloaded','remote')").fetchone()[0]
        eligible = db.execute("""SELECT count(*) FROM items i WHERE status IN ('pending','failed','downloading')
            AND attempts < 3 AND (?=0 OR (SELECT coalesce(sum(duration),0) FROM items d
            WHERE d.source_id=i.source_id AND d.status IN ('downloaded','remote')) < ?)""",
            (profile.max_source_hours, profile.max_source_hours*3600)).fetchone()[0]
        local = db.execute("SELECT count(*) FROM items WHERE status='downloaded'").fetchone()[0]
        attempts = db.execute('SELECT coalesce(sum(attempts),0) FROM items').fetchone()[0]
        return dict(hours=hours, eligible=eligible if not profile.max_hours or hours < profile.max_hours else 0, local=local, attempts=attempts)


def processes(root):
    """Diagnostic only; OS locks, never PID matching, decide worker ownership."""
    result = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True, timeout=10)
    found = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or 'curation.collect' not in parts[1]:
            continue
        try:
            args = shlex.split(parts[1])
            target = args[args.index('--root')+1]
            if Path(target).resolve() == root.resolve():
                found.append(dict(pid=int(parts[0]), command=parts[1]))
        except (ValueError, IndexError):
            continue
    return found


def save(path, state):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as out:
        json.dump(state, out, indent=2)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


class Supervisor:
    def __init__(self, state_dir, *, apply=False, profiles=PROFILES,
                 threshold=40, target=45, floor=25, cooldown=300,
                 max_files=20, max_evict_bytes=10*GIB):
        if not 0 < floor < threshold <= target or cooldown < 30 or max_files < 1 or max_evict_bytes < 1:
            raise ValueError('invalid storage limits, cooldown, or batch bounds')
        self.state_dir = Path(state_dir)
        self.apply, self.profiles = apply, profiles
        self.threshold, self.target, self.floor = threshold, target, floor
        self.cooldown, self.max_files, self.max_evict_bytes = cooldown, max_files, max_evict_bytes

    def tick(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with lock(self.state_dir/'supervisor.lock') as global_lock:
            if global_lock is None:
                return {'busy': True}
            path = self.state_dir/'heartbeat.json'
            state = json.loads(path.read_text()) if path.exists() else {'roots': {}}
            state.update(heartbeat=time.time(), pid=os.getpid(), apply=self.apply)
            for profile in self.profiles:
                entry = state['roots'].setdefault(str(profile.root), {})
                try:
                    self.step(profile, entry, lambda: save(path, state))
                except Exception as exc:
                    entry['failures'] = entry.get('failures', 0)+1
                    entry['error'] = {'at': time.time(), 'type': type(exc).__name__, 'message': str(exc)}
                    entry['next_action'] = time.time()+min(3600, self.cooldown*2**min(entry['failures']-1, 5))
                state['heartbeat'] = time.time()
                save(path, state)
            return state

    def step(self, profile, entry, persist):
        root = profile.root
        entry.update(observed_at=time.time(), profile=dict(namespace=profile.namespace,
                     max_hours=profile.max_hours, max_source_hours=profile.max_source_hours))
        if not (root/'state.sqlite').is_file():
            entry['decision'] = 'missing SQLite root; no adapter/no action'
            return
        entry['workers'] = processes(root)
        entry['inventory'] = inventory(profile)
        entry['free_gib'] = shutil.disk_usage(root).free/GIB
        with lock(root/'download.lock') as download_lock:
            entry['downloader_active'] = download_lock is None
        if not self.apply:
            entry['decision'] = 'observe only'
            return
        if time.time() < entry.get('next_action', 0):
            entry['decision'] = 'cooldown/backoff'
            return
        # Persist throttle before a potentially long operation; a crash cannot
        # cause immediate repeated uploads or worker launches after restart.
        entry['next_action'] = time.time()+self.cooldown
        persist()
        if entry['free_gib'] < self.threshold and entry['inventory']['local']:
            entry['decision'] = 'checkpointing'
            persist()
            hub.backup(root, profile.namespace, max_files=self.max_files)
            # backup returns only after remote snapshot hash verification. evict
            # independently validates immutable checkpoint + per-file LFS proof.
            entry['checkpoint'] = json.loads((root/'hub-checkpoint.json').read_text())
            persist()
            try:
                with lock(root/'download.lock') as idle_lock:
                    if idle_lock is None:
                        raise BlockingIOError('downloader active')
                entry['eviction'] = evict.evict(root, max_files=self.max_files,
                    max_bytes=self.max_evict_bytes, target_free_gib=self.target, apply=True)
            except BlockingIOError:
                entry['decision'] = 'checkpoint saved; active downloader prevents eviction'
            persist()
        # Launch guard is inherited by the child and stays locked for its entire
        # life, including the interval before collect acquires download.lock.
        with lock(root/'supervisor-launch.lock') as launch_lock:
            if launch_lock is None:
                entry['decision'] = 'supervised worker owns launch guard'
                return
            with lock(root/'download.lock') as download_lock:
                if download_lock is None:
                    entry['decision'] = 'existing downloader active'
                    return
                current = inventory(profile)
                eligible = current['eligible']
                free = shutil.disk_usage(root).free/GIB
                if not eligible or free < self.threshold:
                    entry['decision'] = 'no eligible work' if not eligible else 'storage pause'
                    return
                progress = [current['hours'], current['attempts']]
                if entry.get('launch_progress') != progress:
                    entry['launches_without_progress'] = 0
                if entry.get('launches_without_progress', 0) >= 5:
                    entry['decision'] = 'resume retry limit; manual investigation required'
                    return
                entry['launch_progress'] = progress
                entry['launches_without_progress'] = entry.get('launches_without_progress', 0)+1
                command = [sys.executable, '-m', 'curation.collect', 'download', '--root', str(root),
                    '--max-hours', str(profile.max_hours), '--max-source-hours', str(profile.max_source_hours),
                    '--max-items', '10000', '--max-attempts', '3', '--min-free-gib', str(self.floor)]
                entry['launch_intent'] = dict(at=time.time(), command=command)
                persist()
            # The separate launch guard still spans check+spawn; collect itself
            # obtains download.lock, so a competing manual start fails closed.
            with (root/'supervised-worker.log').open('ab') as out:
                worker = subprocess.Popen(command, cwd=BASE, stdout=out, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(launch_lock.fileno(),))
            entry['launched_worker'] = dict(pid=worker.pid, command=command, at=time.time())
            entry['decision'] = 'worker launched'
            entry['failures'] = 0
            persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error('interval must be at least 30 seconds')
    supervisor = Supervisor(BASE/'curation/outputs/supervisor', apply=args.apply)
    while True:
        print(json.dumps(supervisor.tick()), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
