"""Join content-bound text reviews into a new manifest without changing transcripts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .cli import write_jsonl
from .review import eligible, input_identity, parse_response, sha, validate_review
from .schema import digest, load_manifest, validate


def attach(rows, directory):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    result, counts = [], {'attached': 0, 'missing': 0, 'ineligible': 0}
    for row in rows:
        out = copy.deepcopy(row)
        if not eligible(row):
            counts['ineligible'] += 1
            result.append(out)
            continue
        identity = input_identity(row)
        fingerprint = sha({'input': identity, 'config': config})
        path = directory / (fingerprint + '.json')
        if not path.exists():
            counts['missing'] += 1
            result.append(out)
            continue
        review = json.loads(path.read_text())
        if (review.get('cache_key') != fingerprint or review.get('config') != config
                or review.get('input_sha256') != sha(identity)
                or review.get('audio_sha256') != row['audio_sha256']
                or review.get('sentence_sha256') != hashlib.sha256(row['sentence'].encode()).hexdigest()):
            raise ValueError('stale or inconsistent text review')
        if (review.get('status') != 'text_review_only' or review.get('audio_verified') is not False
                or review.get('human_approved') is not False):
            raise ValueError('invalid text-only review status')
        raw = json.loads((directory / (fingerprint + '.response.json')).read_text())
        if sha(raw) != review.get('response_sha256') or parse_response(raw, row) != review.get('review'):
            raise ValueError('saved response differs from review')
        validate_review(review['review'], row)
        if digest(row['audio_path']) != row['audio_sha256']:
            raise ValueError('reviewed audio bytes changed')
        out['text_review'] = review
        out['llm_status'] = 'text_reviewed'
        if out['review_status'] != 'audio_verified':
            out['review_status'] = 'text_reviewed'
        out['provenance']['text_review_sha256'] = sha(review)
        counts['attached'] += 1
        result.append(validate(out))
    return result, counts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--reviews', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows, counts = attach(load_manifest(args.manifest), args.reviews)
    target = Path(args.output)
    if target.exists():
        if load_manifest(target) != rows:
            raise ValueError('output already exists with different content; choose a new snapshot')
    else:
        write_jsonl(target, rows)
    print(json.dumps(dict(rows=len(rows), **counts, transcript_changes=0, audio_verifications_granted=0)))


if __name__ == '__main__':
    main()
