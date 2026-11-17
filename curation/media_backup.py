"""Bounded, verified private backup of finalized radio audio and provenance.

python -m curation.media_backup --root curation/outputs/radio
Local recordings are never deleted. Rights remain unknown; no training clearance.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from .acquire_radio import now
from .collect import file_hash


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()


def prepare(root, max_files=4, max_bytes=512 * 1024**2):
    if not 1 <= max_files <= 100 or max_bytes <= 0:
        raise ValueError('positive bounded limits required (at most 100 recordings)')
    root = Path(root).resolve()
    records, total = [], 0
    for status_path in sorted(root.glob('*/*/status.json')):
        if status_path.is_symlink() or not status_path.resolve().is_relative_to(root):
            raise ValueError('unsafe status path')
        status = json.loads(status_path.read_text())
        if status.get('state') not in ('acquired', 'complete'):
            continue
        audio = status_path.parent / 'capture.mka'
        if audio.is_symlink() or not audio.resolve().is_relative_to(root):
            raise ValueError('unsafe audio path')
        before = audio.stat()
        total += before.st_size
        if total > max_bytes:
            raise ValueError('backup byte bound exceeded')
        sha = file_hash(audio)
        if sha != status.get('audio_sha256') or before.st_size != status.get('bytes'):
            raise ValueError('local audio hash/size differs from finalized status')
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                                 '-of', 'json', str(audio)], capture_output=True, text=True,
                                check=True, timeout=30)
        probe = json.loads(result.stdout)
        duration = float(probe['format']['duration'])
        requested = float(status['requested_seconds'])
        if not math.isfinite(duration) or not math.isfinite(requested) or requested <= 0 or duration < requested - 2:
            raise ValueError('recording is incomplete')
        if not any(s.get('codec_type') == 'audio' for s in probe['streams']):
            raise ValueError('no audio stream')
        if audio.stat() != before:
            raise ValueError('audio changed during validation')
        keys = ('id', 'station', 'source_url', 'source_page', 'source_kind', 'started_at',
                'ended_at', 'requested_seconds', 'candidate_language', 'language_verified',
                'code_switching_expected', 'transcript_status', 'state')
        metadata = {key: status[key] for key in keys if key in status}
        metadata.update(audio_sha256=sha, bytes=before.st_size, duration=duration,
                        rights=dict(training='unknown', commercial='unknown', redistribution='unknown'),
                        training_ready=False,
                        audio_streams=[{k: s[k] for k in ('codec_type', 'codec_name', 'sample_rate', 'channels') if k in s}
                                       for s in probe['streams'] if s.get('codec_type') == 'audio'])
        records.append((audio, sha, encoded(metadata)))
        if len(records) >= max_files:
            break
    if not records:
        raise ValueError('no finalized recordings')
    return records


def verify(api, repo, revision, entries):
    from huggingface_hub import hf_hub_download
    objects = {obj.path: obj for obj in api.get_paths_info(repo, [e['path'] for e in entries],
                                                         repo_type='dataset', revision=revision)}
    for entry in entries:
        obj = objects.get(entry['path'])
        if obj is None or obj.size != entry['bytes']:
            raise ValueError('remote size mismatch; retain local files')
        downloaded = hf_hub_download(repo_id=repo, repo_type='dataset', filename=entry['path'], revision=revision)
        if file_hash(downloaded) != entry['sha256']:
            raise ValueError('remote hash mismatch; retain local files')


def backup(root, repo='sapinsapin/chika', max_files=4, max_bytes=512 * 1024**2):
    from huggingface_hub import HfApi, CommitOperationAdd
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    records = prepare(root, max_files, max_bytes)
    api = HfApi()
    if not api.dataset_info(repo).private:
        raise ValueError('existing private dataset required')
    operations, entries = [], []
    for audio, sha, metadata in records:
        for path, content, digest, size in (
            (f'raw/radio/{sha}/capture.mka', audio, sha, audio.stat().st_size),
            (f'raw/radio/{sha}/metadata.json', metadata, hashlib.sha256(metadata).hexdigest(), len(metadata))):
            operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=content))
            entries.append(dict(path=path, sha256=digest, bytes=size))
    result = api.create_commit(repo_id=repo, repo_type='dataset', operations=operations,
                               commit_message='Archive finalized radio recordings with unverified rights')
    verify(api, repo, result.oid, entries)
    receipt = dict(repo=repo, revision=result.oid, verified_at=now(), recordings=len(records),
                   files=entries, bytes=sum(e['bytes'] for e in entries), training_ready=False)
    payload = encoded(receipt)
    receipt_path = f'checkpoints/radio/{hashlib.sha256(payload).hexdigest()}.json'
    checkpoint = api.upload_file(repo_id=repo, repo_type='dataset', path_in_repo=receipt_path,
                                  path_or_fileobj=payload, commit_message='Checkpoint verified private radio backup')
    verify(api, repo, checkpoint.oid, [dict(path=receipt_path, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))])
    receipt.update(receipt_path=receipt_path, receipt_revision=checkpoint.oid)
    target = Path(root) / 'hub-media-receipt.json'
    fd, tmp = tempfile.mkstemp(prefix='.hub-receipt-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(encoded(receipt)); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('curation/outputs/radio'))
    parser.add_argument('--repo', default='sapinsapin/chika')
    parser.add_argument('--max-files', type=int, default=4)
    parser.add_argument('--max-bytes', type=int, default=512 * 1024**2)
    args = parser.parse_args()
    backup(args.root, args.repo, args.max_files, args.max_bytes)


if __name__ == '__main__':
    main()
