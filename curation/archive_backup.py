"""Bounded private archive backup; never deletes local media.

Default CLI is a verified local plan. --upload explicitly enables network writes.
Objects and recovery checkpoints are content addressed; receipts pin HF commits.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from .acquire_archives import FFPROBE
from .schema import digest, load_manifest


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode()


def safe_url(value):
    u = urlsplit(value)
    if u.scheme not in ('https', 'http') or not u.hostname:
        raise ValueError('public HTTP source URL required')
    # Retain only identity queries; signed URLs, passwords and tracking are omitted.
    return urlunsplit((u.scheme, u.hostname + (f':{u.port}' if u.port else ''), u.path,
                      urlencode([(k, v) for k, v in parse_qsl(u.query) if k in ('v', 'id')]), ''))


SOURCE_KEYS = ('id', 'source_id', 'title', 'language', 'language_status', 'source_kind',
               'duration', 'bytes', 'audio_sha256', 'sample_rate', 'uploader', 'license',
               'language_reported', 'updated_at', 'started_at', 'ended_at', 'requested_seconds')


def prepare(manifest, max_files=4, max_bytes=512 * 1024**2):
    if not 1 <= max_files <= 100 or max_bytes <= 0:
        raise ValueError('positive bounded limits required (at most 100 files)')
    records, total = [], 0
    for row in load_manifest(manifest):
        p = row['provenance']
        if p.get('import_kind') not in ('archive', 'radio') or p.get('acquisition_status') not in ('downloaded', 'acquired'):
            raise ValueError('canonical completed full-recording import required')
        path = Path(row['audio_path'])
        if path.is_symlink() or not path.is_file() or any(x in path.name for x in ('.part', '.tmp')):
            raise ValueError('unsafe audio path')
        before = path.stat()
        if total + before.st_size > max_bytes:
            continue
        if before.st_size <= 0 or before.st_size != row['audio_bytes'] or digest(path) != row['audio_sha256']:
            raise ValueError('local audio hash/size mismatch')
        probe = json.loads(subprocess.run([FFPROBE, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)],
                                         check=True, capture_output=True, timeout=60).stdout)
        duration = float(probe['format']['duration'])
        if not math.isfinite(duration) or duration <= 0 or abs(duration - row['duration']) > max(.1, duration * .001):
            raise ValueError('duration mismatch')
        if not any(s.get('codec_type') == 'audio' for s in probe.get('streams', [])):
            raise ValueError('no audio stream')
        if path.stat() != before:
            raise ValueError('audio changed during validation')
        raw = Path(p['acquisition_metadata_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != p['acquisition_metadata_sha256']:
            raise ValueError('acquisition metadata checksum mismatch')
        source = json.loads(raw)
        metadata = {k: row[k] for k in ('schema_version', 'id', 'source_id', 'recording_id', 'audio_sha256', 'audio_bytes', 'duration', 'language', 'languages', 'speech_type', 'review_status')}
        metadata.update(training_ready=False, rights={k: row['rights'][k] for k in ('training', 'commercial', 'redistribution')},
                        provenance={k: p[k] for k in ('import_kind', 'acquisition_status', 'acquisition_metadata_sha256')},
                        source={k: source[k] for k in SOURCE_KEYS if k in source and isinstance(source[k], (str, int, float, bool, type(None)))})
        metadata['provenance']['source_url'] = safe_url(p['source_url'])
        for k in ('page_url', 'channel_url', 'source_page', 'source_url'):
            if source.get(k):
                metadata['source'][k] = safe_url(source[k])
        blob = encoded(metadata)
        records.append(dict(audio=path, stat=before, metadata=blob, sha256=row['audio_sha256'], bytes=before.st_size))
        total += before.st_size
        if len(records) == max_files:
            break
    if not records:
        raise ValueError('no recordings within bounds')
    return records


def durable(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        name = f.name
        f.write(blob); f.flush(); os.fsync(f.fileno())
    try:
        # Immutable local checkpoints; never replace conflicting receipts.
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != blob:
                raise ValueError('conflicting local checkpoint')
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        os.unlink(name)


def verify(api, repo, revision, entry, download):
    objects = api.get_paths_info(repo, [entry['path']], repo_type='dataset', revision=revision)
    if not objects:
        return False
    obj = objects[0]
    if obj.size != entry['bytes']:
        raise ValueError('remote size conflict')
    lfs = getattr(obj, 'lfs', None)
    sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
    if sha:
        if sha != entry['sha256']:
            raise ValueError('remote hash conflict')
    elif digest(download(repo_id=repo, repo_type='dataset', filename=entry['path'], revision=revision)) != entry['sha256']:
        raise ValueError('remote hash conflict')
    return True


def backup(manifest, state_dir, repo='sapinsapin/chika', max_files=4, max_bytes=512 * 1024**2,
           api=None, download=None, operation=None):
    if api is None or download is None or operation is None:
        from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
        api, download, operation = api or HfApi(), download or hf_hub_download, operation or CommitOperationAdd
    records = prepare(manifest, max_files, max_bytes)
    state = Path(state_dir); state.mkdir(parents=True, exist_ok=True)
    with (state / 'backup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts = []
        for record in records:
            sha = record['sha256']; metadata = record['metadata']; meta_sha = hashlib.sha256(metadata).hexdigest()
            entries = [dict(path=f'raw/archive/sha256/{sha}/audio', sha256=sha, bytes=record['bytes']),
                       dict(path=f'raw/archive/metadata/{meta_sha}.json', sha256=meta_sha, bytes=len(metadata))]
            durable(state / 'metadata' / f'{meta_sha}.json', metadata)
            durable(state / 'plans' / f'{meta_sha}.json', encoded(dict(repo=repo, entries=entries)))
            revisions = []
            for entry, content in zip(entries, (record['audio'], metadata)):
                for attempt in range(3):
                    info = api.dataset_info(repo)
                    if not info.private:
                        raise ValueError('existing private dataset required')
                    revision = info.sha
                    if verify(api, repo, revision, entry, download):
                        break
                    if record['audio'].stat() != record['stat'] or digest(record['audio']) != sha:
                        raise ValueError('audio changed before upload')
                    try:
                        commit = api.create_commit(repo_id=repo, repo_type='dataset', parent_commit=revision,
                                                   operations=[operation(path_in_repo=entry['path'], path_or_fileobj=content)],
                                                   commit_message='Back up verified full recording object')
                    except Exception as exc:
                        # A competing commit can land during a long object upload.
                        # Only explicit precondition conflicts are safe to rebase:
                        # re-read privacy/head and reverify immutable content first.
                        if getattr(getattr(exc, 'response', None), 'status_code', None) != 412 or attempt == 2:
                            raise
                        conflict = dict(path=entry['path'], parent_commit=revision, attempt=attempt+1, status_code=412)
                        durable(state / 'conflicts' / f'{hashlib.sha256(encoded(conflict)).hexdigest()}.json', encoded(conflict))
                        continue
                    revision = commit.oid
                    if not verify(api, repo, revision, entry, download):
                        raise ValueError('remote object absent after upload')
                    break
                progress = dict(repo=repo, revision=revision, **entry)
                durable(state / 'progress' / f'{hashlib.sha256(encoded(progress)).hexdigest()}.json', encoded(progress))
                revisions.append(revision)
            # Verify both objects together at one immutable commit before issuing receipt.
            for entry in entries:
                if not verify(api, repo, revisions[-1], entry, download):
                    raise ValueError('remote checkpoint incomplete')
            receipt = dict(version=1, repo=repo, revision=revisions[-1], entries=entries, local_deletion_authorized=False)
            blob = encoded(receipt)
            receipt_path = state / 'receipts' / f'{hashlib.sha256(blob).hexdigest()}.json'
            durable(receipt_path, blob)
            receipts.append(str(receipt_path))
        return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('--state-dir', default='curation/outputs/archive-backup')
    parser.add_argument('--repo', default='sapinsapin/chika'); parser.add_argument('--max-files', type=int, default=4)
    parser.add_argument('--max-bytes', type=int, default=512 * 1024**2); parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    if args.upload:
        print(json.dumps(backup(args.manifest, args.state_dir, args.repo, args.max_files, args.max_bytes)))
    else:
        records = prepare(args.manifest, args.max_files, args.max_bytes)
        print(json.dumps(dict(files=len(records), bytes=sum(r['bytes'] for r in records), upload=False)))


if __name__ == '__main__':
    main()
