# curation/test_select_review.py
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from curation import select_review
from curation.schema import VERSION


def make_row(row_id, sentence='Maayong aga'):
    return {
        'schema_version': VERSION,
        'id': row_id,
        'source_id': 'source',
        'recording_id': f'recording-{row_id}',
        'audio_path': f'{row_id}.wav',
        'audio_sha256': hashlib.sha256(row_id.encode()).hexdigest(),
        'sentence': sentence,
        'prompt': 'An unsourced prompt',
        'text_is_prompt': False,
        'language': 'hil',
        'languages': ['hil'],
        'speech_type': 'spontaneous',
        'speaker_id': 'speaker',
        'duration': 1.25,
        'num_words': len(sentence.split()),
        'split': 'unassigned',
        'review_status': 'draft',
        'rights': {'training': 'unknown', 'commercial': 'unknown',
                   'redistribution': 'unknown'},
        'provenance': {},
    }


class SelectReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.jsonl'
        self.exclusion = self.root / 'assigned.jsonl'
        self.output = self.root / 'selection'
        self.exclusion.write_bytes(b'')

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path, rows, prefix=b'', suffix=b''):
        payload = b''.join(
            (json.dumps(row, ensure_ascii=False) + '\n').encode()
            for row in rows)
        path.write_bytes(prefix + payload + suffix)
        return payload

    def run_select(self, limit=100):
        return select_review.select(
            self.manifest, [self.exclusion], self.output, limit)

    def test_exclusion_blanks_order_rows_and_limit(self):
        rows = [make_row(name) for name in ('a', 'b', 'c', 'd')]
        blank = make_row('blank', '   ')
        self.write(self.manifest, [rows[0], blank, *rows[1:]],
                   prefix=b'\n  \n', suffix=b'\n')
        # Mere presence in a prior assignment excludes b; no success is required.
        self.write(self.exclusion, [rows[1]])
        result = self.run_select(limit=2)

        selected = [json.loads(line) for line in
                    (self.output / 'input.jsonl').read_text().splitlines()]
        self.assertEqual(selected, [rows[0], rows[2]])  # Full rows, source order.
        self.assertEqual(result['counts'],
                         {'excluded': 1, 'ineligible': 1,
                          'candidates': 3, 'selected': 2})

    def test_hashes_paths_and_counts_cover_source_exclusions_and_output(self):
        source_bytes = self.write(self.manifest, [make_row('a'), make_row('b')])
        exclusion_bytes = self.write(self.exclusion, [make_row('a')])
        result = self.run_select()
        saved = json.loads((self.output / 'selection.json').read_text())
        output_bytes = (self.output / 'input.jsonl').read_bytes()
        digest = lambda value: hashlib.sha256(value).hexdigest()

        self.assertEqual(saved, result)
        self.assertEqual(result['source'], {
            'path': str(self.manifest), 'sha256': digest(source_bytes), 'rows': 2})
        self.assertEqual(result['exclusions'], [{
            'path': str(self.exclusion), 'sha256': digest(exclusion_bytes),
            'rows': 1}])
        self.assertEqual(result['output'], {
            'path': str(self.output / 'input.jsonl'),
            'sha256': digest(output_bytes), 'rows': 1})

    def test_duplicate_ids_are_rejected_in_either_kind_of_input(self):
        for target in ('manifest', 'exclusion'):
            with self.subTest(target=target):
                output = self.root / f'out-{target}'
                self.write(self.manifest, [make_row('source')])
                duplicate_path = getattr(self, target)
                self.write(duplicate_path, [make_row('dup'), make_row('dup')])
                with self.assertRaisesRegex(ValueError, 'duplicate row ID'):
                    select_review.select(
                        self.manifest, [self.exclusion], output)
                self.assertFalse(output.exists())

    def test_empty_selection_fails_without_creating_output(self):
        self.write(self.manifest, [make_row('blank', '  ')])
        with self.assertRaisesRegex(ValueError, 'empty selection'):
            self.run_select()
        self.assertFalse(self.output.exists())

    def test_existing_output_and_contents_are_preserved(self):
        self.write(self.manifest, [make_row('a')])
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_bytes(b'do not replace')
        with self.assertRaises(FileExistsError):
            self.run_select()
        self.assertEqual(sentinel.read_bytes(), b'do not replace')
        self.assertEqual(list(self.output.iterdir()), [sentinel])


if __name__ == '__main__':
    unittest.main()
