"""Publish a validated export to an empty prefix of an existing private dataset.

Only explicit export artifacts are staged. A parent-commit check prevents races;
remote bytes are verified at the resulting immutable commit before receipt creation.
"""
import argparse
import io
import json
from pathlib import Path
import re
import tempfile

from .schema import VERSION, check_splits, digest, load_manifest


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in {'.', '..'}:
        raise ValueError('unsafe export filename')
    return value


def safe_prefix(prefix):
    if not isinstance(prefix, str) or not prefix or any(not part or part in {'.','..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+', part) for part in prefix.split('/')):
        raise ValueError('unsafe repository prefix')
    return prefix


def stage_export(root, staging, max_files=100, max_bytes=2*1024**3):
    import pyarrow.parquet as pq
    import soundfile as sf
    root, staging = Path(root), Path(staging)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('export root must be a real directory')
    if max_files < 3 or max_bytes < 1:
        raise ValueError('invalid upload bounds')
    total = 0
    files = {}

    def copy(name):
        nonlocal total
        safe_name(name)
        if name in files:
            raise ValueError('duplicate export filename')
        path = root/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('export file missing or symlink')
        size = path.stat().st_size
        if len(files) >= max_files or size > max_bytes-total:
            raise ValueError('export exceeds file/byte bound')
        target = staging/name
        # Bound the read itself in case a source is concurrently enlarged.
        with path.open('rb') as source, target.open('xb') as dest:
            count = 0
            while data := source.read(min(1024**2, max_bytes-total-count+1)):
                count += len(data)
                if total+count > max_bytes:
                    raise ValueError('export exceeds byte bound')
                dest.write(data)
        if size != count or path.is_symlink() or digest(path) != digest(target):
            raise ValueError('export changed during staging')
        target.chmod(0o600)
        total += count
        files[name] = dict(sha256=digest(target), bytes=count)

    for name in ('summary.json', 'manifest.jsonl', 'README.md'):
        copy(name)
    summary = json.loads((staging/'summary.json').read_text())
    if summary.get('schema_version') != VERSION or summary.get('profile') not in {'curation','asr','tts'}:
        raise ValueError('invalid export summary schema/profile')
    rate = 24000 if summary['profile'] == 'tts' else 16000
    if summary.get('sample_rate') != rate:
        raise ValueError('export sample rate differs from profile')
    rows = load_manifest(staging/'manifest.jsonl')
    check_splits(rows)
    if summary['profile'] != 'curation':
        for row in rows:
            if (not row['sentence'].strip() or row['review_status'] != 'audio_verified'
                    or row['rights']['training'] != 'allowed' or row['split'] == 'unassigned'):
                raise ValueError('training publication requires verified text, permission and frozen splits')
    if type(summary.get('rows')) is not int or summary['rows'] != len(rows):
        raise ValueError('summary manifest row count mismatch')
    artifacts = summary.get('artifacts')
    if not isinstance(artifacts, list):
        raise ValueError('missing artifacts')
    ids = []
    by_id = {row['id']: row for row in rows}
    core = ['id', 'sentence', 'duration', 'split', 'review_status', 'source_id',
            'recording_id', 'text_is_prompt', 'language', 'num_words',
            'speaker_id', 'speech_type', 'schema_version', 'prompt']
    for artifact in artifacts:
        name = safe_name(artifact['path'])
        if not name.endswith('.parquet'):
            raise ValueError('only parquet artifacts allowed')
        copy(name)
        if files[name]['sha256'] != artifact['sha256']:
            raise ValueError('artifact checksum mismatch')
        parquet = pq.ParquetFile(staging/name)
        if type(artifact.get('rows')) is not int or parquet.metadata.num_rows != artifact['rows']:
            raise ValueError('artifact row count mismatch')
        if not set(core + ['audio', 'provenance_json']).issubset(parquet.schema_arrow.names):
            raise ValueError('artifact missing core scalar fields')
        for batch in parquet.iter_batches(columns=core + ['audio', 'provenance_json'], batch_size=16):
            for actual in batch.to_pylist():
                expected = by_id.get(actual['id'])
                if expected is None or any(actual[key] != expected[key] for key in core):
                    raise ValueError('artifact core scalar fields differ from manifest')
                provenance = json.loads(actual['provenance_json'])
                if not isinstance(provenance, dict):
                    raise ValueError('artifact provenance must be an object')
                original_rate = provenance.pop('original_sample_rate', None)
                if (provenance.pop('export_sample_rate', None) != rate
                        or type(original_rate) is not int or original_rate <= 0 or provenance != expected):
                    raise ValueError('artifact provenance differs from manifest')
                audio = actual['audio']
                if not isinstance(audio, dict) or not isinstance(audio.get('bytes'), bytes) or not audio['bytes']:
                    raise ValueError('artifact requires embedded audio bytes')
                try:
                    with sf.SoundFile(io.BytesIO(audio['bytes'])) as decoded:
                        if (decoded.format, decoded.subtype, decoded.samplerate, decoded.channels) != ('FLAC', 'PCM_16', rate, 1):
                            raise ValueError('artifact audio format differs from profile')
                        frames = sum(len(block) for block in decoded.blocks(blocksize=65536, dtype='int16'))
                        if frames != decoded.frames or abs(frames/rate - expected['duration']) > .1:
                            raise ValueError('artifact decoded audio duration differs from manifest')
                except RuntimeError as exc:
                    raise ValueError('artifact audio cannot be decoded') from exc
                ids.append(actual['id'])
    if len(ids) != len(rows) or sorted(ids) != sorted(row['id'] for row in rows):
        raise ValueError('artifact manifest IDs mismatch')
    return files, len(rows)


def publish(export_root, prefix, receipt, repo='sapinsapin/chika', max_files=100,
            max_bytes=2*1024**3, api=None, download=None, verify_existing=False):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    prefix = safe_prefix(prefix)
    root = Path(export_root)
    receipt = Path(receipt)
    if receipt.is_symlink() or receipt.exists() or receipt.resolve().is_relative_to(root.resolve()):
        raise ValueError('receipt must be new and outside export root')
    api = api or HfApi()
    download = download or hf_hub_download
    with tempfile.TemporaryDirectory(prefix='speech-publish-') as temp:
        staging = Path(temp)
        files, rows = stage_export(root, staging, max_files, max_bytes)
        conflicts = []
        for attempt in range(3):
            info = api.dataset_info(repo)
            if info.private is not True:
                raise ValueError('publication requires an existing private dataset')
            parent = info.sha
            if not parent:
                raise ValueError('missing immutable parent revision')
            existing = api.list_repo_files(repo_id=repo, repo_type='dataset', revision=parent)
            occupied = {path for path in existing if path == prefix or path.startswith(prefix+'/')}
            if verify_existing:
                if occupied != {prefix+'/'+name for name in files}:
                    raise ValueError('existing prefix files differ from export allowlist')
                revision = parent
                break
            if occupied:
                raise ValueError('target prefix is nonempty; immutable exports cannot be overwritten')
            for name, expected in files.items():
                staged = staging/name
                if staged.stat().st_size != expected['bytes'] or digest(staged) != expected['sha256']:
                    raise ValueError('staged artifact changed before publication')
            operations = [CommitOperationAdd(path_in_repo=prefix+'/'+name, path_or_fileobj=str(staging/name)) for name in sorted(files)]
            try:
                commit = api.create_commit(repo_id=repo, repo_type='dataset', operations=operations,
                                           parent_commit=parent, commit_message='Publish validated speech export '+prefix)
            except Exception as exc:
                # Only explicit parent-precondition rejection proves this commit
                # was not applied. Never retry uncertain transport/auth failures.
                if getattr(getattr(exc, 'response', None), 'status_code', None) != 412 or attempt == 2:
                    raise
                conflicts.append(dict(parent_commit=parent, status_code=412))
                continue
            revision = commit.oid
            break
        if not revision:
            raise ValueError('missing immutable publication revision')
        for name, expected in files.items():
            fetched = Path(download(repo_id=repo, repo_type='dataset', filename=prefix+'/'+name,
                                    revision=revision, force_download=True))
            if fetched.stat().st_size != expected['bytes'] or digest(fetched) != expected['sha256']:
                raise ValueError('remote artifact verification failed; no receipt written')
        result = dict(repo=repo, prefix=prefix, revision=revision, parent_commit=parent,
                      rows=rows, files=files, verified=True, private=True,
                      verification_only=verify_existing, parent_conflicts=conflicts)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids silently replacing earlier publication evidence.
        with receipt.open('x') as out:
            receipt.chmod(0o600)
            out.write(json.dumps(result, indent=2, sort_keys=True)+'\n')
        return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--export-root', required=True, type=Path)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--repo', default='sapinsapin/chika')
    parser.add_argument('--max-files', type=int, default=100)
    parser.add_argument('--max-bytes', type=int, default=2*1024**3)
    parser.add_argument('--verify-existing', action='store_true',
                        help='Read-only recovery: verify identical files at current pinned commit without uploading')
    args = parser.parse_args()
    print(json.dumps(publish(args.export_root, args.prefix, args.receipt, args.repo,
                             args.max_files, args.max_bytes, verify_existing=args.verify_existing), sort_keys=True))


if __name__ == '__main__':
    main()
