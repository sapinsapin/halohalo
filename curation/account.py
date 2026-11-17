"""Count transcript coverage without double-counting overlapping source windows.

Coverage is not clean speech: language, music, speakers and acoustic duplicates
still need review. This report never declares a collection target achieved.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

from .schema import digest, load_manifest


def union_seconds(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError('invalid source interval')
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return sum((end-start for start, end in merged), Fraction(0))


def source_interval(row):
    p = row['provenance']
    if all(k in p for k in ('original_audio_sha256', 'start_sample', 'end_sample', 'offset_sample_rate')):
        rate, start, end = p['offset_sample_rate'], p['start_sample'], p['end_sample']
        if any(type(n) is not int for n in (rate, start, end)) or rate <= 0 or start < 0 or end <= start:
            raise ValueError('invalid sample provenance')
        interval = Fraction(start, rate), Fraction(end, rate)
        source_hash=p['original_audio_sha256']
        if not isinstance(source_hash,str) or len(source_hash)!=64 or any(c not in '0123456789abcdef' for c in source_hash):
            raise ValueError('invalid original audio hash')
        if abs(interval[1]-interval[0]-Fraction(str(row['duration']))) > Fraction(1,rate):
            raise ValueError('duration differs from source interval')
        return p['original_audio_sha256'], interval
    if 'start' in row or 'end' in row:
        raise ValueError('segmented rows need exact original-source sample offsets')
    return row['audio_sha256'], (Fraction(0), Fraction(str(row['duration'])))


def account(inputs):
    coverage = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    row_counts = defaultdict(int)
    sources_groups = defaultdict(set)
    checked = set()
    for group, manifest in inputs:
        if not group:
            raise ValueError('group required')
        for row in load_manifest(manifest):
            identity = (row['audio_path'], row['audio_sha256'])
            if identity not in checked:
                if digest(row['audio_path']) != row['audio_sha256']:
                    raise ValueError('audio hash mismatch')
                checked.add(identity)
            source, interval = source_interval(row)
            sources_groups[source].add(group)
            row_counts[group] += 1
            stages = ['prepared']
            if row.get('asr_status') == 'transcribed' and row['sentence'].strip():
                stages.append('draft_transcribed')
                if row['review_status'] in {'text_reviewed', 'audio_verified'}:
                    stages.append('review_status_recorded')
                if row['review_status'] == 'audio_verified':
                    stages.append('audio_verified')
                    if row['rights']['training'] == 'allowed' and row['split'] in {'train','validation','test','reserved_test'}:
                        stages.append('training_permission_and_review')
            for stage in stages:
                coverage[group][stage][source].append(interval)
    groups = {}
    for group in row_counts:
        groups[group] = dict(input_rows=row_counts[group], covered_source_hours={
            stage: float(sum((union_seconds(intervals) for intervals in coverage[group][stage].values()), Fraction(0))/3600)
            for stage in ('prepared','draft_transcribed','review_status_recorded','audio_verified','training_permission_and_review')})
    return dict(groups=groups, checked_audio_files=len(checked),
                cross_group_source_hashes={key: sorted(value) for key, value in sources_groups.items() if len(value)>1},
                target_completion_assessed=False,
                limitations=['Overlapping windows and repeated manifests count once per source hash and group.',
                             'Source-hash equality does not detect re-encodings, excerpts or simulcasts.',
                             'Covered duration includes padding/silence/music; it is not unique retained speech.',
                             'Permission/review count is not a train split: evaluation reservations remain reserved.'])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest', action='append', required=True, help='GROUP=/path/to/manifest.jsonl; repeatable')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    inputs = []
    for spec in args.manifest:
        group, sep, path = spec.partition('=')
        if not sep:
            p.error('manifest must be GROUP=PATH')
        inputs.append((group, path))
    result = account(inputs)
    from .review import atomic
    atomic(Path(args.output), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
