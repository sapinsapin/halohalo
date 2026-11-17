"""Content-addressed private Hub backup. Never deletes local files automatically.

Back up completed RSS downloads and a consistent SQLite snapshot:
python -m curation.hub --root curation/outputs/rss --namespace hiligaynon-rss
Only files whose remote LFS SHA256 and length match are checkpointed as backed up.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from .collect import connect, file_hash, now


def backup(root, namespace, repo='sapinsapin/chika', max_files=20):
    from huggingface_hub import HfApi, hf_hub_download
    if not namespace or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in namespace):
        raise ValueError('namespace must use lowercase letters, digits, hyphen or underscore')
    root = Path(root)
    db = connect(root)
    api = HfApi()
    api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
    info = api.dataset_info(repo)
    if not info.private:
        raise ValueError('raw acquisition backup requires a private repository')
    db.execute('''CREATE TABLE IF NOT EXISTS hub_backups(
        sha256 TEXT PRIMARY KEY, repo TEXT, path TEXT, revision TEXT, verified_at TEXT, bytes INTEGER)''')
    rows = db.execute("SELECT * FROM items WHERE status='downloaded' AND sha256 NOT IN (SELECT sha256 FROM hub_backups WHERE repo=?) LIMIT ?", (repo,max_files)).fetchall()
    for row in rows:
        local = Path(row['path'])
        if file_hash(local) != row['sha256']:
            raise ValueError(f'local hash mismatch: {row["id"]}')
        remote = f'raw/{namespace}/{row["sha256"]}.media'
        result = api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type='dataset',
                                 commit_message=f'Archive {namespace} recording {row["id"]}')
        objects = api.get_paths_info(repo, [remote], repo_type='dataset', revision=result.oid)
        obj = objects[0]
        lfs = obj.lfs
        remote_sha = getattr(lfs, 'sha256', None) if lfs is not None else None
        if isinstance(lfs, dict):
            remote_sha = lfs.get('sha256')
        if remote_sha != row['sha256'] or obj.size != local.stat().st_size:
            raise ValueError('remote hash/size not proven; retain local file')
        db.execute('INSERT OR REPLACE INTO hub_backups VALUES(?,?,?,?,?,?)',
                   (row['sha256'],repo,remote,result.oid,now(),obj.size))
        db.commit()
        print(json.dumps({'id':row['id'],'remote':remote,'verified_revision':result.oid}),flush=True)
    snapshot = root / 'hub-state.sqlite'
    dest = sqlite3.connect(snapshot)
    db.backup(dest)
    dest.close()
    # A content-addressed checkpoint is immutable and survives later failed runs.
    sha = file_hash(snapshot)
    result = api.upload_file(path_or_fileobj=snapshot,path_in_repo=f'checkpoints/{namespace}/{sha}.sqlite',
                            repo_id=repo,repo_type='dataset',commit_message=f'Checkpoint {namespace} acquisition ledger')
    fetched = hf_hub_download(repo_id=repo, repo_type='dataset', filename=f'checkpoints/{namespace}/{sha}.sqlite', revision=result.oid)
    if file_hash(fetched) != sha:
        raise ValueError('remote checkpoint hash mismatch')
    receipt = dict(repo=repo,checkpoint=f'checkpoints/{namespace}/{sha}.sqlite',revision=result.oid,
                   checkpoint_sha256=sha,uploaded_at=now())
    (root/'hub-checkpoint.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--namespace',required=True)
    p.add_argument('--repo',default='sapinsapin/chika')
    p.add_argument('--max-files',type=int,default=20)
    a=p.parse_args()
    backup(a.root,a.namespace,a.repo,a.max_files)


if __name__=='__main__': main()
