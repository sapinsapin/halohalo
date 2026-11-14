"""Restore an RSS checkpoint into a NEW empty root without touching the old archive.

python -m curation.restore --receipt old-root/hub-checkpoint.json --root new-root --max-files 5
python -m curation.restore --repo sapinsapin/chika --revision COMMIT_SHA \
  --checkpoint checkpoints/namespace/SHA256.sqlite --root new-root --max-files 0

Unselected backed-up objects are status=remote, not downloaded. Missing backups
become pending with original metadata retained in restore_inventory. This command
never evicts originals and never accepts a mutable Hub branch as a revision.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import tempfile

from .collect import file_hash, now

HASH = re.compile(r'^[0-9a-f]{64}$')
REVISION = re.compile(r'^[0-9a-f]{40}(?:[0-9a-f]{24})?$')


def safe_repo(repo):
    if not isinstance(repo, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo):
        raise ValueError('expected Hub repository owner/name')
    return repo


def safe_remote_path(path):
    if not isinstance(path, str) or not path or '\\' in path:
        raise ValueError('invalid Hub path')
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or str(p) != path:
        raise ValueError('unsafe Hub path')
    return path


def receipt_value(receipt=None, repo=None, revision=None, checkpoint=None, checkpoint_sha256=None):
    if receipt is not None:
        if any(x is not None for x in (repo, revision, checkpoint, checkpoint_sha256)):
            raise ValueError('provide receipt OR explicit checkpoint coordinates')
        data = json.loads(Path(receipt).read_text())
    else:
        data = dict(repo=repo, revision=revision, checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha256)
    safe_repo(data.get('repo'))
    if not isinstance(data.get('revision'), str) or not REVISION.fullmatch(data['revision']):
        raise ValueError('checkpoint revision must be an immutable full commit SHA')
    path = safe_remote_path(data.get('checkpoint'))
    parts = PurePosixPath(path).parts
    if len(parts) != 3 or parts[0] != 'checkpoints' or not re.fullmatch(r'[a-z0-9_-]+', parts[1]):
        raise ValueError('expected checkpoints/namespace/SHA256.sqlite')
    stem = PurePosixPath(path).stem
    if not HASH.fullmatch(stem) or not path.endswith('.sqlite'):
        raise ValueError('checkpoint filename must be content-addressed by SHA256')
    expected = data.get('checkpoint_sha256') or stem
    if expected != stem:
        raise ValueError('checkpoint hash conflicts with content-addressed path')
    return {k: data[k] for k in ('repo', 'revision', 'checkpoint')} | {'checkpoint_sha256': expected}


def remote_metadata(api, backup):
    safe_repo(backup['repo'])
    safe_remote_path(backup['path'])
    if not REVISION.fullmatch(backup['revision'] or ''):
        raise ValueError('media revision must be an immutable full commit SHA')
    objects = api.get_paths_info(backup['repo'], [backup['path']], repo_type='dataset', revision=backup['revision'])
    if len(objects) != 1:
        raise ValueError('backed-up media is missing from immutable revision')
    obj = objects[0]
    lfs = getattr(obj, 'lfs', None)
    sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
    if sha != backup['sha256'] or obj.size != backup['bytes']:
        raise ValueError('remote media hash/size mismatch')
    return obj.size


def restore(root, *, receipt=None, repo=None, revision=None, checkpoint=None,
            checkpoint_sha256=None, max_files=5, min_free_gib=25, max_bytes=2 * 1024**3, api=None, downloader=None):
    if max_files < 0 or min_free_gib < 0 or max_bytes < 0:
        raise ValueError('max-files, min-free-gib and max-bytes must be nonnegative')
    coords = receipt_value(receipt, repo, revision, checkpoint, checkpoint_sha256)
    # Do not resolve a symlink into an existing data directory.
    root = Path(root).absolute()
    if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise ValueError('restore root must be new or empty and cannot be a symlink')
    root.parent.mkdir(parents=True, exist_ok=True)
    if api is None or downloader is None:
        from huggingface_hub import HfApi, hf_hub_download
        api = api or HfApi()
        downloader = downloader or hf_hub_download
    if shutil.disk_usage(root.parent).free < min_free_gib * 1024**3:
        raise OSError('storage floor reached before restore')
    checkpoint_file = Path(downloader(repo_id=coords['repo'], repo_type='dataset',
                                     filename=coords['checkpoint'], revision=coords['revision']))
    if file_hash(checkpoint_file) != coords['checkpoint_sha256']:
        raise ValueError('checkpoint SHA256 mismatch')
    staging = Path(tempfile.mkdtemp(prefix='.chika-restore-', dir=root.parent))
    db = None
    try:
        source = sqlite3.connect(checkpoint_file.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            check = source.execute('PRAGMA integrity_check').fetchall()
            if check != [('ok',)]:
                raise ValueError('checkpoint SQLite integrity check failed')
            tables = {r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'items', 'sources', 'events'} <= tables:
                raise ValueError('checkpoint missing RSS acquisition tables')
            # Use SQLite backup so a downloaded cache file is never mutated.
            db = sqlite3.connect(staging/'state.sqlite')
            source.backup(db)
        finally:
            source.close()
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA trusted_schema=OFF')
        # Imported unlink intents belong to the old filesystem. Preserve them as
        # provenance, never execute them against this recovered archive.
        if 'eviction_journal' in tables:
            db.execute('CREATE TABLE IF NOT EXISTS restore_eviction_history(restored_at TEXT,payload TEXT)')
            for entry in list(db.execute('SELECT * FROM eviction_journal')):
                db.execute('INSERT INTO restore_eviction_history VALUES(?,?)', (now(), json.dumps(dict(entry))))
            db.execute('DROP TABLE eviction_journal')
        db.execute('''CREATE TABLE IF NOT EXISTS restore_inventory(
            item_id TEXT PRIMARY KEY, original_status TEXT, original_path TEXT,
            original_sha256 TEXT, original_duration REAL, original_bytes INTEGER,
            recovery_state TEXT, restored_at TEXT)''')
        rows = list(db.execute('SELECT * FROM items ORDER BY id'))
        backups = {}
        if 'hub_backups' in tables:
            for r in db.execute('SELECT * FROM hub_backups'):
                if not HASH.fullmatch(r['sha256'] or ''):
                    raise ValueError('invalid media hash in checkpoint backup ledger')
                if not isinstance(r['bytes'], int) or r['bytes'] <= 0:
                    raise ValueError('invalid byte length in backup ledger')
                backups[r['sha256']] = dict(r)
        selected = []
        for row in rows:
            original = dict(row)
            sha = row['sha256']
            state = row['status']
            if state == 'downloaded' and not HASH.fullmatch(sha or ''):
                raise ValueError('downloaded checkpoint row lacks a valid audio hash')
            backup = backups.get(sha)
            if backup is not None and state in {'downloaded', 'remote'}:
                if row['bytes'] != backup['bytes'] or not row['duration'] or row['duration'] <= 0:
                    raise ValueError('media metadata conflicts with backup ledger')
                next_state = 'remote'
                db.execute("UPDATE items SET status='remote',path=NULL,updated_at=? WHERE id=?", (now(), row['id']))
                if sha not in selected and len(selected) < max_files:
                    selected.append(sha)
            else:
                next_state = 'pending' if state in {'downloaded','downloading','remote'} else state
                # Missing local files must not retain a claim to local duration/bytes.
                db.execute('''UPDATE items SET status=?,path=NULL,sha256=NULL,duration=NULL,bytes=NULL,
                              etag=NULL,last_modified=NULL,attempts=?,updated_at=? WHERE id=?''',
                           (next_state, 0 if next_state == 'pending' else row['attempts'], now(), row['id']))
            db.execute('INSERT OR REPLACE INTO restore_inventory VALUES(?,?,?,?,?,?,?,?)',
                       (row['id'], original['status'], original['path'], sha, original['duration'],
                        original['bytes'], next_state, now()))
        # Feed snapshots aren't currently included in hub.py's backup. Keep provenance
        # but do not present paths on another machine as locally available snapshots.
        for row in list(db.execute('SELECT id,metadata FROM sources')):
            metadata = json.loads(row['metadata'])
            if metadata.get('feed_snapshot'):
                metadata['original_feed_snapshot'] = metadata.pop('feed_snapshot')
                metadata['feed_snapshot_available'] = False
            db.execute('UPDATE sources SET metadata=? WHERE id=?', (json.dumps(metadata), row['id']))
        downloaded = []
        if max_bytes and sum(backups[sha]['bytes'] for sha in selected) > max_bytes:
            raise ValueError('selected files exceed max-bytes; lower max-files or raise byte budget')
        (staging/'audio').mkdir()
        for sha in selected:
            backup = backups[sha]
            size = remote_metadata(api, backup)
            # Account for both cache download and the independent restored copy.
            if shutil.disk_usage(staging).free < min_free_gib * 1024**3 + 2 * size:
                raise OSError('storage floor would be crossed by restored media')
            cached = Path(downloader(repo_id=backup['repo'], repo_type='dataset',
                                     filename=backup['path'], revision=backup['revision']))
            if cached.stat().st_size != size or file_hash(cached) != sha:
                raise ValueError('downloaded media hash/size mismatch')
            target = staging/'audio'/f'{sha}.media'
            shutil.copyfile(cached, target)
            if target.stat().st_size != size or file_hash(target) != sha:
                raise ValueError('restored media hash/size mismatch')
            final_path = root/'audio'/target.name
            db.execute("UPDATE items SET status='downloaded',path=?,updated_at=? WHERE sha256=? AND status='remote'",
                       (str(final_path), now(), sha))
            db.execute("UPDATE restore_inventory SET recovery_state='downloaded' WHERE original_sha256=? AND recovery_state='remote'", (sha,))
            downloaded.append({'sha256':sha,'bytes':size,'path':str(final_path)})
        counts = {r[0]:r[1] for r in db.execute('SELECT status,count(*) FROM items GROUP BY status')}
        summary = {'restored_at':now(), 'checkpoint':coords, 'root':str(root),
                   'max_files':max_files, 'max_bytes':max_bytes, 'local_unique_files':len(downloaded), 'status_counts':counts,
                   'files':downloaded, 'notes':'Remote rows are not locally downloaded. Pending rows lack backups and may be reacquired from their source URLs. No original files were changed.'}
        db.execute('INSERT INTO events VALUES(?,?,?,?)', (now(),None,'checkpoint_restored',json.dumps(summary)))
        db.commit()
        db.close()
        db = None
        (staging/'restore-receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
        (staging/'hub-checkpoint.json').write_text(json.dumps(coords,indent=2)+'\n')
        # Atomic publish; a failed restore leaves no misleading completed ledger.
        if root.exists():
            root.rmdir()  # succeeds only if still empty; never deletes user data
        staging.rename(root)
        return summary
    finally:
        if db is not None:
            db.close()
        if staging.exists():
            shutil.rmtree(staging)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--receipt')
    p.add_argument('--repo')
    p.add_argument('--revision')
    p.add_argument('--checkpoint')
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--max-files',type=int,default=5)
    p.add_argument('--min-free-gib',type=float,default=25)
    p.add_argument('--max-bytes',type=int,default=2 * 1024**3,help='maximum selected media bytes; 0 means unlimited')
    a = p.parse_args()
    print(json.dumps(restore(a.root,receipt=a.receipt,repo=a.repo,revision=a.revision,
                            checkpoint=a.checkpoint,checkpoint_sha256=a.checkpoint_sha256,
                            max_files=a.max_files,min_free_gib=a.min_free_gib,max_bytes=a.max_bytes),indent=2))


if __name__ == '__main__':
    main()
