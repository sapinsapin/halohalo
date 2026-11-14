"""Deterministic metadata-only checkpoint bundles; local pack/verify/restore.

Only JSON/JSONL ledgers and HTML source snapshots are included; audio, logs,
temporary files and symlinks are excluded. Keep credentials outside these roots. Bundles
contain private source metadata and must be uploaded to a private dataset.
Live process files are historical observations, never proof a worker is alive.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

FORMAT = 'sapinsapin.archive-checkpoint.v1'
SUFFIXES = {'.json', '.jsonl', '.html'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode()


def stable_read(path):
    """A replaced/changed live ledger is retried, never read as a mixed snapshot."""
    for _ in range(3):
        with path.open('rb') as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        current = path.stat()
        signature = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if signature(before) == signature(after) == signature(current):
            return data
    raise RuntimeError('metadata changed while reading: ' + str(path))


def metadata_paths(root):
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlinks are not allowed in checkpoint roots: ' + str(path))
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if any(part.startswith('.') for part in path.relative_to(root).parts):
            continue
        if any(marker in path.name for marker in ('.tmp.', '.part.', '.partial.')):
            continue
        yield path


def pack(roots, output):
    roots = [Path(p).resolve() for p in roots]
    output = Path(output)
    if len({p.name for p in roots}) != len(roots):
        raise ValueError('checkpoint roots must have distinct directory names')
    if output.exists():
        raise FileExistsError('immutable checkpoint already exists: ' + str(output))
    for root in roots:
        if not root.is_dir():
            raise ValueError('missing checkpoint root: ' + str(root))
        if output.resolve().is_relative_to(root):
            raise ValueError('checkpoint output must be outside input roots')
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix='chika-checkpoint-') as work:
        work = Path(work)
        for root in sorted(roots):
            for path in metadata_paths(root):
                data = stable_read(path)
                # Reject truncated metadata; HTML snapshots remain byte-exact.
                if path.suffix == '.json':
                    json.loads(data)
                elif path.suffix == '.jsonl':
                    for line in data.splitlines():
                        if line.strip(): json.loads(line)
                name = (Path(root.name) / path.relative_to(root)).as_posix()
                target = work / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                records.append({'path':name, 'bytes':len(data), 'sha256':sha256(data)})
        records.sort(key=lambda row:row['path'])
        manifest = {'format':FORMAT, 'roots':sorted(p.name for p in roots), 'files':records,
            'audio_included':False, 'process_status_is_historical':True,
            'consistency':'Each file is stable and hashed; multiple live workers are not transactionally frozen.',
            'restore_notes':'Restore into an empty directory. Reconcile audio paths/hashes and check actual process liveness before restarting.'}
        manifest_bytes = canonical(manifest)
        tmp_output = output.with_suffix(output.suffix + '.tmp')
        try:
            with tmp_output.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
                with tarfile.open(mode='w', fileobj=compressed, format=tarfile.PAX_FORMAT) as archive:
                    for name in ['CHECKPOINT.json'] + [r['path'] for r in records]:
                        data = manifest_bytes if name == 'CHECKPOINT.json' else (work / name).read_bytes()
                        info = tarfile.TarInfo(name)
                        info.size = len(data); info.mode = 0o600; info.mtime = 0
                        archive.addfile(info, io.BytesIO(data))
            report = verify(tmp_output)
            tmp_output.replace(output)
        finally:
            tmp_output.unlink(missing_ok=True)
    checksum = sha256(output.read_bytes())
    output.with_suffix(output.suffix + '.sha256').write_text(checksum + '  ' + output.name + '\n')
    return {'path':str(output.resolve()), 'sha256':checksum, 'files':report['files'],
        'metadata_bytes':sum(r['bytes'] for r in records), 'bundle_bytes':output.stat().st_size}


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name) and name != '.' and not path.is_absolute() and '..' not in path.parts and '\\' not in name and str(path) == name


def verify(bundle):
    with tarfile.open(bundle, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive members')
        if any(not m.isfile() or not safe_name(m.name) for m in members):
            raise ValueError('unsafe archive member')
        if any(PurePosixPath(name).suffix not in SUFFIXES for name in names):
            raise ValueError('non-metadata archive member')
        if 'CHECKPOINT.json' not in names:
            raise ValueError('checkpoint manifest missing')
        manifest = json.load(archive.extractfile('CHECKPOINT.json'))
        if manifest.get('format') != FORMAT:
            raise ValueError('unsupported checkpoint format')
        rows = manifest['files']
        if len({r['path'] for r in rows}) != len(rows):
            raise ValueError('duplicate manifest paths')
        if set(names) != {'CHECKPOINT.json'} | {r['path'] for r in rows}:
            raise ValueError('archive and manifest members differ')
        for row in rows:
            data = archive.extractfile(row['path']).read()
            if len(data) != row['bytes'] or sha256(data) != row['sha256']:
                raise ValueError('checksum mismatch: ' + row['path'])
        return {'files':len(rows), 'format':FORMAT, 'valid':True}


def restore(bundle, destination):
    report = verify(bundle)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('restore destination must be empty')
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError('restore destination cannot be a symlink')
    # Never use extractall; only verified regular-file names are written.
    with tarfile.open(bundle, 'r:gz') as archive:
        for member in archive.getmembers():
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(archive.extractfile(member).read())
    return dict(report, destination=str(destination.resolve()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack'); p.add_argument('--root', action='append', required=True); p.add_argument('--output', required=True)
    p = commands.add_parser('verify'); p.add_argument('bundle')
    p = commands.add_parser('restore'); p.add_argument('bundle'); p.add_argument('--destination', required=True)
    args = parser.parse_args()
    if args.command == 'pack': result = pack(args.root, args.output)
    elif args.command == 'verify': result = verify(args.bundle)
    else: result = restore(args.bundle, args.destination)
    print(json.dumps(result))


if __name__ == '__main__': main()
