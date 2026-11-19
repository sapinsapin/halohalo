"""English translation column for Chika manifests: write a translation job, then attach finished translations.

The job file is what gets handed to Antigravity, ChatGPT or an API script. Each line carries the row id, the exact
sentence to translate, its hash and the expected source language. Translators return one JSON line per row:

    {"id": "<row id>", "source_sentence_sha256": "<from the job>", "translation_en": "...", "model": "<who/what>"}

Attaching never edits `sentence`: a translation whose hash no longer matches the row's current sentence is rejected,
so translations cannot drift from (or overwrite) transcripts. Output is a new manifest; inputs are untouched.

    python -m curation.translations job --manifest M.jsonl --output job.jsonl
    python -m curation.translations attach --manifest M.jsonl --translations done.jsonl --output M-translated.jsonl
"""
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .cli import write_jsonl
from .schema import load_manifest, validate

LANGUAGE_NAMES = {'fil': 'Filipino/Tagalog (often mixed with English)', 'hil': 'Hiligaynon', 'ceb': 'Cebuano',
                  'bik': 'Bikol', 'bcl': 'Bikol', 'ilo': 'Ilocano', 'en': 'English'}
RULES = ('Translate `sentence` into faithful English. Keep names, places and numbers; keep meaning and tone; parts '
         'already in English stay as they are; no notes or explanations; keep unclear or untranslatable words in '
         '[square brackets]. Never change `sentence` itself. An empty sentence gets an empty translation.')


def sentence_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def job(rows):
    """Rows with a nonempty sentence and no current translation."""
    return [{'id': r['id'], 'language': r['language'], 'language_name': LANGUAGE_NAMES.get(r['language'], r['language']),
             'sentence': r['sentence'], 'source_sentence_sha256': sentence_sha(r['sentence']), 'rules': RULES}
            for r in rows if r['sentence'].strip() and 'translation' not in r]


def attach(rows, translations, status='machine_draft', created_at=None):
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    by_id = {}
    for t in translations:
        if t['id'] in by_id:
            raise ValueError(f'duplicate translation for {t["id"]}')
        if not isinstance(t.get('translation_en'), str) or not t.get('model'):
            raise ValueError(f'{t["id"]}: translation_en and model are required')
        by_id[t['id']] = t
    known = {r['id'] for r in rows}
    unknown = sorted(set(by_id) - known)
    if unknown:
        raise ValueError(f'translations for unknown rows: {unknown[:3]}')
    out, counts = [], {'attached': 0, 'already_translated': 0, 'missing': 0, 'empty_sentence': 0}
    for row in rows:
        new = copy.deepcopy(row)
        t = by_id.get(row['id'])
        if not row['sentence'].strip():
            counts['empty_sentence'] += 1
        elif t is None:
            counts['already_translated' if 'translation' in row else 'missing'] += 1
        elif 'translation' in row:
            raise ValueError(f'{row["id"]}: already translated; build a new snapshot from the untranslated manifest')
        else:
            if t.get('source_sentence_sha256') != sentence_sha(row['sentence']):
                raise ValueError(f'{row["id"]}: translation was made from a different sentence')
            text = ' '.join(t['translation_en'].split())
            if not text:
                raise ValueError(f'{row["id"]}: empty translation for a nonempty sentence')
            new['translation'] = {'language': 'en', 'text': text, 'status': status, 'model': t['model'],
                                  'source_sentence_sha256': t['source_sentence_sha256'], 'created_at': created_at}
            counts['attached'] += 1
        out.append(validate(new))
    return out, counts


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('job')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('attach')
    p.add_argument('--manifest', required=True)
    p.add_argument('--translations', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    target = Path(args.output)
    if target.exists():
        raise SystemExit('output exists; outputs are immutable')
    if args.command == 'job':
        items = job(rows)
        write_jsonl(target, items)
        print(json.dumps({'rows': len(rows), 'to_translate': len(items)}))
        return
    translations = [json.loads(l) for l in Path(args.translations).read_text().splitlines() if l.strip()]
    result, counts = attach(rows, translations)
    write_jsonl(target, result)
    print(json.dumps(dict(rows=len(result), **counts, transcript_changes=0)))


if __name__ == '__main__':
    main()
