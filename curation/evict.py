"""Evict checkpointed RSS originals only after re-verifying their private Hub copies.

Dry run (default): python -m curation.evict --root ARCHIVE --max-files 5
Apply: python -m curation.evict --root ARCHIVE --max-files 5 --apply

The downloader's OS lock must be free. This command never stops a downloader.
An immutable checkpoint must prove every selected item's backup. The local ledger
is committed to status=remote BEFORE unlink; eviction_journal records the original
path so an interrupted unlink can be safely reconciled on the next invocation.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat

from .collect import file_hash, now
from .restore import HASH, receipt_value, remote_metadata


JOURNAL_SQL = '''CREATE TABLE IF NOT EXISTS eviction_journal(
    original_path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
    item_ids TEXT NOT NULL, checkpoint TEXT NOT NULL, state TEXT NOT NULL,
    prepared_at TEXT NOT NULL, completed_at TEXT)'''


def local_path(root, value):
    """Only regular .media paths under this root's audio/, with no inner symlinks."""
    if not isinstance(value, str):
        raise ValueError('missing local media path')
    path = Path(value)
    if not path.is_absolute() or path.suffix != '.media' or '..' in path.parts:
        raise ValueError('eviction requires an absolute .media path')
    resolved_root = root.resolve()
    try:
        path.resolve().relative_to(resolved_root/'audio')
    except ValueError:
        raise ValueError('media path is outside archive audio directory')
    for component in (path, *path.parents):
        if component.resolve() == resolved_root:
            break
        if component.is_symlink():
            raise ValueError('media path contains a symlink')
    return path


def fingerprint(path, expected_sha, expected_size):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ValueError('local media type/size mismatch')
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
        after = os.fstat(handle.fileno())
        mark = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if mark != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or h.hexdigest() != expected_sha:
            raise ValueError('local media hash changed or mismatched')
        return mark


def unlink_verified(path, sha, size, original_fingerprint):
    # Verify again after the durable remote transition; a crash leaves a journal,
    # and any replacement or edit makes this operation fail closed.
    mark = fingerprint(path, sha, size)
    if mark != original_fingerprint:
        raise ValueError('local media was replaced after planning')
    current = path.lstat()
    if stat.S_ISLNK(current.st_mode) or mark != (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns):
        raise ValueError('local media changed before unlink')
    path.unlink()


def open_checkpoint(coords, api, downloader):
    if not api.dataset_info(coords['repo']).private:
        raise ValueError('eviction requires a private backup repository')
    cached = Path(downloader(repo_id=coords['repo'],repo_type='dataset',filename=coords['checkpoint'],revision=coords['revision']))
    if file_hash(cached) != coords['checkpoint_sha256']:
        raise ValueError('checkpoint SHA256 mismatch')
    db = sqlite3.connect(cached.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    db.row_factory = sqlite3.Row
    try:
        if [tuple(r) for r in db.execute('PRAGMA integrity_check')] != [('ok',)]:
            raise ValueError('checkpoint integrity check failed')
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'items','hub_backups'} <= tables:
            raise ValueError('checkpoint lacks item/backup proof tables')
        return db
    except BaseException:
        db.close()
        raise


def prove(rows, local_db, checkpoint_db, coords, api):
    backup = None
    for row in rows:
        if not HASH.fullmatch(row['sha256'] or ''):
            raise ValueError('invalid local ledger audio hash')
        saved = checkpoint_db.execute('SELECT * FROM items WHERE id=?',(row['id'],)).fetchone()
        if saved is None or saved['status'] not in {'downloaded','remote'}:
            raise ValueError('item is not captured as acquired in immutable checkpoint')
        for key in ('source_id','url','guid','sha256','bytes','duration'):
            if saved[key] != row[key]:
                raise ValueError(f'checkpoint item metadata mismatch: {key}')
        local = local_db.execute('SELECT * FROM hub_backups WHERE sha256=?',(row['sha256'],)).fetchone()
        saved_backup = checkpoint_db.execute('SELECT * FROM hub_backups WHERE sha256=?',(row['sha256'],)).fetchone()
        if local is None or saved_backup is None:
            raise ValueError('checkpoint does not prove this item backup')
        for key in ('repo','path','revision','bytes','sha256'):
            if local[key] != saved_backup[key]:
                raise ValueError(f'checkpoint backup metadata mismatch: {key}')
        if not local['verified_at'] or local['repo'] != coords['repo'] or local['bytes'] != row['bytes']:
            raise ValueError('backup is unverified or belongs to another repository')
        if backup is not None and dict(local) != backup:
            raise ValueError('one local file is claimed by conflicting backup records')
        backup = dict(local)
    if backup is None:
        raise ValueError('no item proof for local file')
    remote_metadata(api, backup)


def evict(root, *, receipt=None, max_files=1, max_bytes=1024**3,
          target_free_gib=None, apply=False, api=None, downloader=None):
    if max_files < 0 or max_bytes < 0 or (target_free_gib is not None and target_free_gib < 0):
        raise ValueError('limits must be nonnegative')
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir() or not (root/'state.sqlite').is_file():
        raise ValueError('existing non-symlink archive root and state.sqlite required')
    if (root/'download.lock').is_symlink() or (root/'state.sqlite').is_symlink():
        raise ValueError('archive lock and ledger must not be symlinks')
    with (root/'download.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise BlockingIOError('download.lock is owned; no eviction performed')
        if api is None or downloader is None:
            from huggingface_hub import HfApi, hf_hub_download
            api = api or HfApi()
            downloader = downloader or hf_hub_download
        coords=receipt_value(receipt=receipt or root/'hub-checkpoint.json')
        db=sqlite3.connect((root/'state.sqlite').resolve().as_uri()+('?mode=rw' if apply else '?mode=ro'),uri=True)
        db.row_factory=sqlite3.Row
        checkpoints={}
        try:
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'items','hub_backups','events'} <= tables:
                raise ValueError('archive lacks item/backup proof tables')
            candidates=[]
            prepared_paths=set()
            if 'eviction_journal' in tables:
                for j in db.execute("SELECT * FROM eviction_journal WHERE state='prepared' ORDER BY prepared_at,original_path"):
                    path=local_path(root,j['original_path'])
                    ids=json.loads(j['item_ids'])
                    rows=[db.execute('SELECT * FROM items WHERE id=?',(ident,)).fetchone() for ident in ids]
                    if not rows or any(r is None or r['status']!='remote' or r['path'] is not None or r['sha256']!=j['sha256'] or r['bytes']!=j['bytes'] for r in rows):
                        raise ValueError('prepared eviction journal conflicts with current ledger')
                    jc=receipt_value(**json.loads(j['checkpoint']))
                    candidates.append((path,rows,jc,True))
                    prepared_paths.add(str(path))
            groups={}
            for row in db.execute("SELECT i.* FROM items i JOIN hub_backups b ON b.sha256=i.sha256 WHERE i.status='downloaded' ORDER BY i.id"):
                path=local_path(root,row['path'])
                if str(path) not in prepared_paths:
                    groups.setdefault(str(path),[]).append(row)
            for path,rows in groups.items():
                # A shared restored file may serve several IDs. All references must
                # transition together and all must have checkpoint proof.
                refs=list(db.execute('SELECT * FROM items WHERE path=?',(path,)))
                if len(refs)!=len(rows) or any(r['status']!='downloaded' for r in refs):
                    raise ValueError('local file has an uncheckpointed or active ledger reference')
                candidates.append((Path(path),rows,coords,False))
            free=shutil.disk_usage(root).free
            target=None if target_free_gib is None else target_free_gib*1024**3
            plan=[]
            planned_bytes=0
            for path,rows,proof_coords,prepared in candidates:
                if len(plan)>=max_files or (target is not None and free+planned_bytes>=target):
                    break
                size=rows[0]['bytes']
                sha=rows[0]['sha256']
                if not isinstance(size,int) or size<=0 or any(r['bytes']!=size or r['sha256']!=sha for r in rows):
                    raise ValueError('conflicting local file metadata')
                if max_bytes and planned_bytes+size>max_bytes:
                    continue
                key=json.dumps(proof_coords,sort_keys=True)
                if key not in checkpoints:
                    checkpoints[key]=open_checkpoint(proof_coords,api,downloader)
                prove(rows,db,checkpoints[key],proof_coords,api)
                exists=path.exists()
                if not exists and not prepared:
                    raise ValueError('downloaded media is missing without a prepared eviction journal')
                mark=fingerprint(path,sha,size) if exists else None
                plan.append(dict(path=str(path),sha256=sha,bytes=size,item_ids=[r['id'] for r in rows],
                                 checkpoint=proof_coords,prepared=prepared,already_absent=not exists,fingerprint=mark))
                planned_bytes+=size if exists else 0
            completed=[]
            if apply:
                db.execute(JOURNAL_SQL)
                db.commit()
                for entry in plan:
                    if target is not None and shutil.disk_usage(root).free>=target:
                        break
                    path=local_path(root,entry['path'])
                    if not entry['already_absent']:
                        mark=fingerprint(path,entry['sha256'],entry['bytes'])
                        if mark!=entry['fingerprint']:
                            raise ValueError('local file changed since eviction planning')
                    db.execute('BEGIN IMMEDIATE')
                    try:
                        for ident in entry['item_ids']:
                            row=db.execute('SELECT * FROM items WHERE id=?',(ident,)).fetchone()
                            expected='remote' if entry['prepared'] else 'downloaded'
                            if row is None or row['status']!=expected or row['sha256']!=entry['sha256'] or row['bytes']!=entry['bytes'] or row['path']!=(None if entry['prepared'] else str(path)):
                                raise ValueError('ledger changed after eviction planning')
                        db.execute('''INSERT INTO eviction_journal VALUES(?,?,?,?,?,?,?,?)
                            ON CONFLICT(original_path) DO UPDATE SET sha256=excluded.sha256,bytes=excluded.bytes,
                            item_ids=excluded.item_ids,checkpoint=excluded.checkpoint,state=excluded.state,
                            prepared_at=excluded.prepared_at,completed_at=NULL''',
                            (str(path),entry['sha256'],entry['bytes'],json.dumps(entry['item_ids']),
                             json.dumps(entry['checkpoint']),'prepared',now(),None))
                        for ident in entry['item_ids']:
                            db.execute("UPDATE items SET status='remote',path=NULL,updated_at=? WHERE id=?",(now(),ident))
                        db.commit()
                    except BaseException:
                        db.rollback()
                        raise
                    if not entry['already_absent']:
                        unlink_verified(path,entry['sha256'],entry['bytes'],entry['fingerprint'])
                    db.execute("UPDATE eviction_journal SET state='unlinked',completed_at=? WHERE original_path=?",(now(),str(path)))
                    db.execute('INSERT INTO events VALUES(?,?,?,?)',(now(),None,'media_evicted',json.dumps({k:v for k,v in entry.items() if k!='fingerprint'})))
                    db.commit()
                    completed.append(entry)
            return dict(dry_run=not apply,root=str(root),planned_files=len(plan),
                        planned_bytes=planned_bytes,evicted_files=len(completed),
                        unlinked_bytes=sum(e['bytes'] for e in completed if not e['already_absent']),
                        free_bytes_before=free,free_bytes_after=shutil.disk_usage(root).free,
                        files=[{k:v for k,v in e.items() if k!='fingerprint'} for e in (completed if apply else plan)],
                        note='Only checkpoint-proven originals are eligible. Remote status retains metadata; the eviction journal supports interrupted unlink recovery. No downloader is stopped.')
        finally:
            for checkpoint_db in checkpoints.values():checkpoint_db.close()
            db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--receipt')
    p.add_argument('--max-files',type=int,default=1)
    p.add_argument('--max-bytes',type=int,default=1024**3,help='maximum planned media bytes; 0 means unlimited')
    p.add_argument('--target-free-gib',type=float)
    p.add_argument('--apply',action='store_true',help='actually mark remote and unlink; default is dry run')
    a=p.parse_args()
    print(json.dumps(evict(a.root,receipt=a.receipt,max_files=a.max_files,max_bytes=a.max_bytes,
                           target_free_gib=a.target_free_gib,apply=a.apply),indent=2))


if __name__=='__main__':main()
