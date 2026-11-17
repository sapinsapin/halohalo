# curation/select_review.py
"""Offline selection of eligible, previously unassigned manifest rows.

Distinct directories do not coordinate claims; the caller must serialize runs and
enumerate all assignments with --exclude-manifest. Every exclusion row is treated
as assigned, whether its subsequent review succeeded or failed. For the first run,
pass an explicitly created empty exclusion manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .review import eligible
from .schema import validate


def _read(path):
    path = Path(path)
    raw = path.read_bytes()  # The sole read; parsing and provenance use these bytes.
    info = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    try:
        lines = raw.decode('utf-8').splitlines()
    except UnicodeDecodeError:
        raise ValueError(f'{path}: manifest is not UTF-8') from None
    rows, ids = [], set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = validate(json.loads(line))
            row_id = row['id']
            if row_id in ids:
                raise ValueError(f'duplicate row ID: {row_id!r}')
            ids.add(row_id)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f'{path}:{number}: {exc}') from None
        rows.append(row)
    info['rows'] = len(rows)
    return rows, info


def _encode(rows):
    return ''.join(
        json.dumps(row, ensure_ascii=False, sort_keys=True,
                   separators=(',', ':'), allow_nan=False) + '\n'
        for row in rows
    ).encode('utf-8')


def content_key(row):
    """Same review input under another ID; never include source questions."""
    return (row['audio_sha256'], row['sentence'], row['language'],
            tuple(row['languages']))


def select(manifest, exclude_manifests, output, limit=100):
    if limit < 1:
        raise ValueError('limit must be positive')
    if not exclude_manifests:
        raise ValueError('at least one explicit exclusion manifest is required')

    rows, source = _read(manifest)
    exclusions, assigned, assigned_content = [], set(), set()
    for path in exclude_manifests:
        excluded_rows, info = _read(path)
        exclusions.append(info)
        assigned.update(row['id'] for row in excluded_rows)
        assigned_content.update(content_key(row) for row in excluded_rows)

    chosen = []
    counts = {'excluded': 0, 'ineligible': 0, 'candidates': 0}
    content_excluded = 0
    seen_content = set()
    for row in rows:  # Validate the whole input above, but never reorder it.
        if row['id'] in assigned:
            counts['excluded'] += 1
            continue
        if not eligible(row):
            counts['ineligible'] += 1
            continue
        key = content_key(row)
        if key in assigned_content or key in seen_content:
            counts['excluded'] += 1
            content_excluded += 1
            continue
        seen_content.add(key)
        counts['candidates'] += 1
        if len(chosen) < limit:
            chosen.append(row)
    if not chosen:
        raise ValueError('empty selection')

    payload = _encode(chosen)
    output = Path(output)
    # Exclusive creation is the concurrency guard; never reuse an existing path.
    output.mkdir(parents=True, exist_ok=False)
    input_path = output / 'input.jsonl'
    input_path.write_bytes(payload)
    counts['selected'] = len(chosen)
    result = {
        'source': source,
        'exclusions': exclusions,
        'output': {
            'path': str(input_path),
            'sha256': hashlib.sha256(payload).hexdigest(),
            'rows': len(chosen),
        },
        'limit': limit,
        'counts': counts,
        'content_aliases_excluded': content_excluded,
    }
    (output / 'selection.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    return result


def _positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True,
                        help='canonical source JSONL')
    parser.add_argument('--exclude-manifest', action='append', required=True,
                        help='prior assigned input.jsonl; repeat for every assignment')
    parser.add_argument('--output', required=True, help='new output directory')
    parser.add_argument('--limit', type=_positive, default=100)
    args = parser.parse_args(argv)
    try:
        select(args.manifest, args.exclude_manifest, args.output, args.limit)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{parser.prog}: error: {exc}\n')


if __name__ == '__main__':
    main()
