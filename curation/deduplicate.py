"""Derive an exact-byte unique manifest without modifying source rows or media."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .schema import digest, load_manifest


def select(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['audio_sha256']].append(row)
    canonical, aliases = [], []
    rank = {'audio_verified': 0, 'text_reviewed': 1, 'draft': 2}
    for audio_hash, group in sorted(groups.items()):
        for key in ('sentence', 'split', 'language', 'languages', 'rights',
                    'duration', 'speaker_id', 'synthetic', 'speech_type'):
            if any(row.get(key) != group[0].get(key) for row in group[1:]):
                raise ValueError(f'conflicting {key}: {audio_hash}')
        chosen = min(group, key=lambda r: (rank[r['review_status']], r['id']))
        canonical.append(chosen)
        for row in group:
            if row['id'] != chosen['id']:
                aliases.append({'canonical_id': chosen['id'],
                                'audio_sha256': audio_hash, 'original_row': row})
    return canonical, aliases


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source_hash = digest(args.manifest)
    rows = load_manifest(args.manifest)
    canonical, aliases = select(rows)
    for row in rows:
        if digest(row['audio_path']) != row['audio_sha256']:
            raise ValueError('audio changed: ' + row['id'])
    if digest(args.manifest) != source_hash:
        raise ValueError('source manifest changed')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for name, data in (('manifest', canonical), ('aliases', aliases)):
        (root / (name + '.jsonl')).write_text(''.join(
            json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    audit = dict(source=args.manifest, source_sha256=source_hash,
                 input_rows=len(rows), canonical_rows=len(canonical),
                 alias_rows=len(aliases), audio_hashes_verified=len(rows),
                 canonical_sha256=digest(root/'manifest.jsonl'),
                 aliases_sha256=digest(root/'aliases.jsonl'),
                 rows_modified=0, audio_deleted=0,
                 review_note='Existing rows preserved verbatim; no new review approval.',
                 limitation='Exact byte deduplication only; acoustic duplicates remain possible.')
    (root/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
