"""Versioned working manifest; export retains halohalo's speech core columns."""
import hashlib
import json

VERSION = 'sapinsapin.speech.v1'
REQUIRED = {'schema_version', 'id', 'source_id', 'recording_id', 'audio_path',
            'audio_sha256', 'sentence', 'prompt', 'text_is_prompt', 'language',
            'languages', 'speech_type', 'speaker_id', 'duration', 'num_words',
            'split', 'review_status', 'rights', 'provenance'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def validate(row):
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f'missing fields: {sorted(missing)}')
    if row['schema_version'] != VERSION:
        raise ValueError('unsupported schema version')
    if row['text_is_prompt'] is not False:
        raise ValueError('sentence must never contain a source question')
    if not isinstance(row['sentence'], str) or row['num_words'] != len(row['sentence'].split()):
        raise ValueError('sentence/num_words mismatch')
    if not isinstance(row['duration'], (float, int)) or not 0 < row['duration'] < float('inf'):
        raise ValueError('invalid duration')
    if row['split'] not in {'unassigned', 'train', 'validation', 'test', 'reserved_test'}:
        raise ValueError('invalid split')
    if row['review_status'] not in {'draft', 'text_reviewed', 'audio_verified'}:
        raise ValueError('invalid review status')
    if row['review_status'] == 'audio_verified':
        decision = row.get('audio_review', {})
        if not all(decision.get(k) for k in ('reviewer', 'reviewed_at', 'audio_sha256', 'sentence_sha256')):
            raise ValueError('audio verification requires attributable, content-bound review')
        if decision['audio_sha256'] != row['audio_sha256'] or decision['sentence_sha256'] != hashlib.sha256(row['sentence'].encode()).hexdigest():
            raise ValueError('stale audio review')
    if 'translation' in row:
        validate_translation(row)
    for key in ('training', 'commercial', 'redistribution'):
        if row['rights'].get(key) not in {'allowed', 'denied', 'unknown'}:
            raise ValueError(f'invalid rights.{key}')
    if row['rights']['training'] == 'allowed' and not row['rights'].get('evidence'):
        raise ValueError('training permission requires evidence')
    if len(row['audio_sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['audio_sha256']):
        raise ValueError('invalid audio hash')
    if not isinstance(row['languages'], list) or row['language'] not in row['languages']:
        raise ValueError('languages must include primary language')
    return row


TRANSLATION_STATUSES = {'machine_draft', 'human_reviewed'}


def validate_translation(row):
    """Optional English translation of `sentence`, bound to the exact sentence it translates (never the reverse)."""
    t = row['translation']
    if not isinstance(t, dict) or t.get('language') != 'en' or not isinstance(t.get('text'), str):
        raise ValueError('translation must be {language: en, text: str, ...}')
    if t.get('source_sentence_sha256') != hashlib.sha256(row['sentence'].encode()).hexdigest():
        raise ValueError('stale translation: sentence changed since it was translated')
    if bool(t['text'].strip()) != bool(row['sentence'].strip()):
        raise ValueError('translation must be empty exactly when the sentence is empty')
    if t.get('status') not in TRANSLATION_STATUSES or not t.get('model'):
        raise ValueError('translation needs a status and the producing model or translator')
    if t['status'] == 'human_reviewed' and not (t.get('reviewer') and t.get('reviewed_at')):
        raise ValueError('human-reviewed translation must be attributable')


def translation_en(row):
    """Flat export value: the English text, or None when absent."""
    t = row.get('translation')
    return t['text'] if t else None


def load_manifest(path):
    with open(path) as f:
        rows = [validate(json.loads(line)) for line in f if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate row IDs')
    return rows


def check_splits(rows):
    """No recording, known speaker or exact source bytes across evaluation/train."""
    seen = {}
    for r in rows:
        split = 'test' if r['split'] == 'reserved_test' else r['split']
        if split == 'unassigned':
            continue
        keys = [('recording', r['source_id'], r['recording_id']), ('audio', r['audio_sha256'])]
        if r['speaker_id']:
            keys.append(('speaker', r['source_id'], r['speaker_id']))
        for key in keys:
            if key in seen and seen[key] != split:
                raise ValueError(f'cross-split leakage: {key}')
            seen[key] = split
