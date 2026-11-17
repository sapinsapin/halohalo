"""Audit candidate PCM and offsets against an independently supplied parent manifest.

This checks derivative integrity, not whether the audio is speech or correctly labeled.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .schema import digest, load_manifest


def audit(parents_path, candidates_path):
    parents = load_manifest(parents_path)
    index = {row['id']: row for row in parents}
    if len(index) != len(parents):
        raise ValueError('duplicate parent ID')
    rows = load_manifest(candidates_path)
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate candidate ID')
    groups = defaultdict(list)
    for row in rows:
        groups[row['provenance']['parent_manifest_id']].append(row)
    samples = 0
    for parent_id, children in groups.items():
        if parent_id not in index:
            raise ValueError('parent absent from supplied manifest')
        parent = index[parent_id]
        if digest(parent['audio_path']) != parent['audio_sha256']:
            raise ValueError('parent audio checksum mismatch')
        info = sf.info(parent['audio_path'])
        if (info.samplerate, info.channels, info.subtype) != (16000, 1, 'PCM_16'):
            raise ValueError('invalid parent audio format')
        original, _ = sf.read(parent['audio_path'], dtype='int16')
        offset = parent['provenance']['start_sample']
        previous = 0
        for row in sorted(children, key=lambda r: r['provenance']['parent_start_sample']):
            p = row['provenance']
            start, end = p['parent_start_sample'], p['parent_end_sample']
            if (not isinstance(start, int) or not isinstance(end, int)
                    or not previous <= start < end <= len(original) or end-start > 27*16000):
                raise ValueError('candidate bounds or overlap invalid')
            previous = end
            if p['parent_record'] != parent:
                raise ValueError('embedded parent differs from supplied manifest')
            if (p['start_sample'], p['end_sample'], p['offset_sample_rate']) != (offset+start, offset+end, 16000):
                raise ValueError('absolute sample offsets differ')
            for key, expected in [('start', (offset+start)/16000), ('end', (offset+end)/16000), ('duration', (end-start)/16000)]:
                if abs(row[key]-expected) > 1e-9:
                    raise ValueError('seconds differ from sample offsets')
            for key in ('source_id', 'recording_id', 'language', 'languages', 'split', 'rights', 'prompt', 'text_is_prompt'):
                if row[key] != parent[key]:
                    raise ValueError('inherited metadata differs')
            path = Path(row['audio_path'])
            if path.stat().st_size != row['audio_bytes'] or digest(path) != row['audio_sha256']:
                raise ValueError('candidate audio checksum or size mismatch')
            info = sf.info(path)
            if (info.samplerate, info.channels, info.subtype) != (16000, 1, 'PCM_16'):
                raise ValueError('invalid candidate audio format')
            actual, _ = sf.read(path, dtype='int16')
            if not np.array_equal(actual, original[start:end]):
                raise ValueError('candidate PCM differs from parent slice')
            samples += len(actual)
    return dict(rows=len(rows), parents_checked=len(groups), exact_pcm_samples=samples,
                candidate_hours=samples/16000/3600, parent_manifest_sha256=digest(parents_path),
                candidate_manifest_sha256=digest(candidates_path),
                scope='Derivative PCM, offsets and inherited metadata only; no speech, language or transcript verification.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--parents', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = audit(args.parents, args.candidates)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
