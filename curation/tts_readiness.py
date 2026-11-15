"""Data-only readiness audit for the authorized Filipino MOSS pilot. Never launches training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from .schema import load_manifest, check_splits, digest


def audit(manifest, recipe):
    rows = load_manifest(manifest)
    cfg = recipe['pilot_data_gate']
    reasons = Counter()
    eligible = []
    for row in rows:
        why = []
        if row['language'] != recipe['language']: why.append('wrong_primary_language')
        if not row['sentence'].strip(): why.append('empty_transcript')
        if row['review_status'] != 'audio_verified': why.append('audio_review_pending')
        if row['rights']['training'] != 'allowed': why.append('training_permission_pending')
        if not row['speaker_id']: why.append('speaker_unknown')
        if row['split'] not in ('train', 'validation', 'test', 'reserved_test'): why.append('split_unassigned')
        if not cfg['min_clip_seconds'] <= row['duration'] <= cfg['max_clip_seconds']: why.append('clip_duration')
        if why:
            reasons.update(why)
        else:
            if digest(row['audio_path']) != row['audio_sha256']:
                raise ValueError('eligible audio checksum mismatch')
            eligible.append(row)
    if len({r["audio_sha256"] for r in eligible}) != len(eligible):
        raise ValueError("duplicate eligible audio would inflate training hours")
    check_splits(eligible)
    hours = defaultdict(float)
    for row in eligible:
        hours['test' if row['split'] == 'reserved_test' else row['split']] += row['duration'] / 3600
    deficits = {split: max(0, cfg['min_' + split + '_hours'] - hours[split])
                for split in ('train', 'validation', 'test')}
    return {'manifest_sha256': digest(manifest), 'input_rows': len(rows),
            'eligible_rows': len(eligible), 'excluded_rows': len(rows)-len(eligible),
            'exclusion_reasons': dict(reasons), 'eligible_hours': dict(hours),
            'minimum_hours_missing': deficits, 'data_gate_passed': not any(deficits.values()),
            'training_launched': False,
            'scope': 'Data gate only. Passing does not prove codec/runtime checks, voice quality or commercial clearance.'}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--recipe', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    result = audit(a.manifest, json.loads(Path(a.recipe).read_text()))
    Path(a.report).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['data_gate_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
