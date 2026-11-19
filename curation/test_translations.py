import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave

from .cli import export, write_jsonl
from .schema import VERSION, digest, validate
from .translations import attach, job, sentence_sha


def make_row(audio, sentence='Maayong aga sa inyo tanan.', rid='one'):
    return dict(schema_version=VERSION, id=rid, source_id='source', recording_id='rec-' + rid, audio_path=str(audio),
                audio_sha256=digest(audio), sentence=sentence, prompt=None, text_is_prompt=False, language='hil',
                languages=['hil'], speech_type='unknown', speaker_id='', duration=1., num_words=len(sentence.split()),
                split='unassigned', review_status='draft',
                rights=dict(training='unknown', commercial='unknown', redistribution='unknown'), provenance={})


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.audio = self.root / 'a.wav'
        with wave.open(str(self.audio), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b'\0\0' * 16000)

    def tearDown(self):
        self.tmp.cleanup()

    def test_job_attach_and_export_column(self):
        import pyarrow.parquet as pq
        rows = [make_row(self.audio), make_row(self.audio, sentence='', rid='empty')]
        items = job(rows)
        self.assertEqual([i['id'] for i in items], ['one'])
        done = [{'id': 'one', 'source_sentence_sha256': items[0]['source_sentence_sha256'],
                 'translation_en': ' Good  morning to all of you. ', 'model': 'test-model'}]
        out, counts = attach(rows, done, created_at='2026-11-19T00:00:00+00:00')
        self.assertEqual(counts['attached'], 1)
        self.assertEqual(counts['empty_sentence'], 1)
        self.assertEqual(out[0]['sentence'], rows[0]['sentence'])  # transcript untouched
        self.assertEqual(out[0]['translation']['text'], 'Good morning to all of you.')
        self.assertNotIn('translation', out[1])
        manifest = self.root / 'm.jsonl'
        write_jsonl(manifest, out)
        args = SimpleNamespace(manifest=str(manifest), audio_root=None, output=str(self.root / 'export'),
                               profile='curation', shard_size=10)
        with contextlib.redirect_stdout(io.StringIO()):
            export(args)
        table = pq.read_table(next((self.root / 'export').glob('*.parquet')))
        got = {r['id']: r['translation_en'] for r in table.to_pylist()}
        self.assertEqual(got, {'one': 'Good morning to all of you.', 'empty': None})
        self.assertIn('translation_en', table.schema.names)

    def test_rejects_translation_of_a_different_sentence(self):
        rows = [make_row(self.audio)]
        bad = [{'id': 'one', 'source_sentence_sha256': sentence_sha('something else'),
                'translation_en': 'Good morning.', 'model': 'm'}]
        with self.assertRaisesRegex(ValueError, 'different sentence'):
            attach(rows, bad)

    def test_rejects_unknown_ids_empty_text_and_retranslation(self):
        rows = [make_row(self.audio)]
        sha = sentence_sha(rows[0]['sentence'])
        with self.assertRaisesRegex(ValueError, 'unknown rows'):
            attach(rows, [{'id': 'nope', 'source_sentence_sha256': sha, 'translation_en': 'x', 'model': 'm'}])
        with self.assertRaisesRegex(ValueError, 'empty translation'):
            attach(rows, [{'id': 'one', 'source_sentence_sha256': sha, 'translation_en': '  ', 'model': 'm'}])
        out, _ = attach(rows, [{'id': 'one', 'source_sentence_sha256': sha, 'translation_en': 'Hi.', 'model': 'm'}])
        with self.assertRaisesRegex(ValueError, 'already translated'):
            attach(out, [{'id': 'one', 'source_sentence_sha256': sha, 'translation_en': 'Hello.', 'model': 'm'}])

    def test_schema_rejects_stale_translation(self):
        out, _ = attach([make_row(self.audio)], [{'id': 'one', 'source_sentence_sha256': sentence_sha(
            'Maayong aga sa inyo tanan.'), 'translation_en': 'Good morning.', 'model': 'm'}])
        edited = copy.deepcopy(out[0])
        edited['sentence'] = 'Maayong gab-i.'
        edited['num_words'] = 2
        with self.assertRaisesRegex(ValueError, 'stale translation'):
            validate(edited)
        unattributed = copy.deepcopy(out[0])
        unattributed['translation']['status'] = 'human_reviewed'
        with self.assertRaisesRegex(ValueError, 'attributable'):
            validate(unattributed)


if __name__ == '__main__':
    unittest.main()
