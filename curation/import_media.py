"""Verify completed archive/radio media into immutable canonical manifests.

No audio is copied or changed. Byte duplicates are excluded, not claimed as unique
speech: music, language, overlap, speakers and rights still require review.
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

from .acquire_archives import FFPROBE
from .schema import VERSION, digest, validate


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode()


def acquisition_record(row):
    """Adapter consumed by prepare; retain canonical annotations/provenance."""
    validate(row)
    return dict(path=row['audio_path'], sha256=row['audio_sha256'],
                id=row['recording_id'], source_id=row['source_id'],
                language=row['language'], duration=row['duration'],
                url=row['provenance']['source_url'], canonical_record=row)


def verify_record(metadata, kind, language='hil'):
    if not isinstance(language, str) or not language.isalpha() or not 2 <= len(language) <= 3:
        raise ValueError('explicit language code required')
    raw = metadata.read_bytes()
    record = json.loads(raw)
    if (record.get('status') if kind == 'archive' else record.get('state')) != ('downloaded' if kind == 'archive' else 'acquired'):
        raise ValueError('not completed')
    path = Path(record['audio_path']).resolve(strict=True)
    if not path.is_file() or any(s in path.name for s in ('.part', '.tmp')):
        raise ValueError('partial or non-file audio')
    before = path.stat()
    if before.st_size <= 0 or before.st_size != record['bytes']:
        raise ValueError('audio size mismatch')
    sha = digest(path)
    if sha != record['audio_sha256']:
        raise ValueError('audio checksum mismatch')
    result = subprocess.run([FFPROBE, '-v', 'error', '-show_format', '-show_streams',
                             '-of', 'json', str(path)], check=True, capture_output=True, timeout=60)
    probe = json.loads(result.stdout)
    if not any(s.get('codec_type') == 'audio' for s in probe.get('streams', [])):
        raise ValueError('no audio stream')
    duration = float(probe['format']['duration'])
    claimed = float(record['duration'])
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(claimed) or abs(duration - claimed) > max(.1, duration * .001):
        raise ValueError('audio duration mismatch')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or metadata.read_bytes() != raw:
        raise ValueError('source changed during verification')
    source = record['source_id'] if kind == 'archive' else record['station']
    recording = record['id']
    ident = hashlib.sha256(encoded([kind, source, recording, sha])).hexdigest()
    return validate(dict(schema_version=VERSION, id=ident, source_id=source,
        recording_id=recording, audio_path=str(path), audio_sha256=sha,
        audio_bytes=before.st_size, sentence='', prompt=None, text_is_prompt=False,
        language=language, languages=[language], language_annotation='provisional source candidate; audio language unverified',
        speech_type='unknown', speaker_id='', duration=duration, num_words=0,
        split='unassigned', review_status='draft',
        rights={k:'unknown' for k in ('training','commercial','redistribution')},
        provenance=dict(source_url=record.get('page_url') or record.get('source_url') or record.get('download_url'),
            import_kind=kind, acquisition_metadata_path=str(metadata.resolve()),
            acquisition_metadata_sha256=hashlib.sha256(raw).hexdigest(),
            acquisition_status='downloaded' if kind == 'archive' else 'acquired'),
        asr_status='pending', llm_status='pending', synthetic=False,
        flags=['speech_presence_unverified','speaker_unknown','language_unverified','rights_unknown','alignment_required_for_tts']))


def build(archive_root=None, radio_root=None, language='hil'):
    rows, excluded = [], []
    for kind, root in [('archive', archive_root), ('radio', radio_root)]:
        if root is None:
            continue
        root = Path(root).resolve(strict=True)
        paths = sorted(root.glob('*/*.json') if kind == 'archive' else root.glob('*/*/status.json'))
        for path in paths:
            if path.name.endswith(('.info.json', '.tmp.json')):
                continue
            try:
                record = json.loads(path.read_bytes())
                if not isinstance(record, dict) or ('status' not in record if kind == 'archive' else 'state' not in record):
                    continue
                rows.append(verify_record(path, kind, language))
            except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
                excluded.append(dict(metadata_path=str(path), reason=str(exc) if not isinstance(exc, subprocess.SubprocessError) else type(exc).__name__))
    rows.sort(key=lambda r:(r['source_id'], r['recording_id'], r['provenance']['import_kind'], r['provenance']['acquisition_metadata_path']))
    unique, seen, duplicates = [], {}, []
    for row in rows:
        previous = seen.get(row['audio_sha256'])
        if previous:
            duplicates.append(dict(id=row['id'], source_id=row['source_id'], recording_id=row['recording_id'],
                metadata_path=row['provenance']['acquisition_metadata_path'], audio_sha256=row['audio_sha256'], duplicate_of=previous['id']))
        else:
            seen[row['audio_sha256']] = row
            unique.append(row)
    sources = {}
    for row in unique:
        stat = sources.setdefault(row['source_id'], dict(recordings=0, measured_media_hours=0))
        stat['recordings'] += 1
        stat['measured_media_hours'] += row['duration']/3600
    report = dict(verified_completed_recordings=len(rows), imported_recordings=len(unique),
        source_count=len(sources), measured_media_hours=sum(r['duration'] for r in unique)/3600,
        sources=sources, exact_byte_duplicates=duplicates, excluded=excluded,
        verified_speech_hours=0, transcribed_hours=0,
        note='Measured media duration only; not unique speech. Exact byte deduplication does not detect decoded repeats, clips, simulcasts, music or language. Rights and speakers unknown.')
    return unique, report


def run(output, archive_root=None, radio_root=None, language='hil'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output/'import.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, report = build(archive_root, radio_root, language)
        manifest = b''.join(encoded(r) for r in rows)
        report_bytes = encoded(report)
        snapshot = hashlib.sha256(manifest + report_bytes).hexdigest()
        snapshots = output/'snapshots'
        snapshots.mkdir(exist_ok=True, mode=0o700)
        destination = snapshots/snapshot
        files = {'manifest.jsonl':manifest, 'report.json':report_bytes}
        if destination.exists():
            if any((destination/name).read_bytes() != data for name, data in files.items()):
                raise ValueError('immutable snapshot modified')
        else:
            with tempfile.TemporaryDirectory(prefix='.import-', dir=snapshots) as tmp:
                staging = Path(tmp)/'snapshot'
                staging.mkdir(mode=0o700)
                for name, data in files.items():
                    target = staging/name
                    target.write_bytes(data)
                    target.chmod(0o600)
                os.rename(staging, destination)
        return dict(manifest=str((destination/'manifest.jsonl').resolve()), report=str((destination/'report.json').resolve()),
                    imported_recordings=report['imported_recordings'], source_count=report['source_count'],
                    measured_media_hours=report['measured_media_hours'], exact_byte_duplicates=len(report['exact_byte_duplicates']),
                    excluded=len(report['excluded']))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--radio-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--language', default='hil', help='provisional language code for this acquisition root')
    args = parser.parse_args()
    if args.archive_root is None and args.radio_root is None:
        parser.error('at least one acquisition root is required')
    print(json.dumps(run(args.output, args.archive_root, args.radio_root, args.language), sort_keys=True))


if __name__ == '__main__':
    main()
