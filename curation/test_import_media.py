import json
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

from .import_media import acquisition_record, build, run
from .schema import digest, load_manifest


@unittest.skipUnless(shutil.which('ffprobe'), 'ffprobe required')
class ImportMediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root/'archive'
        self.radio = self.root/'radio'
        self.archive.mkdir(); self.radio.mkdir()

    def record(self, source='a', ident='one', kind='archive', state=None):
        folder = (self.archive/source if kind == 'archive' else self.radio/source/ident)
        folder.mkdir(parents=True, exist_ok=True)
        audio = folder/(ident+'.wav')
        with wave.open(str(audio), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(b'\0\0'*16000)
        row = dict(id=ident, audio_path=str(audio), audio_sha256=digest(audio),
                   bytes=audio.stat().st_size, duration=1, page_url='https://example.org/audio',
                   question='DO NOT IMPORT', sentence='DO NOT IMPORT')
        row.update(dict(source_id=source, status=state or 'downloaded') if kind == 'archive'
                   else dict(station=source, state=state or 'acquired'))
        metadata = folder/(ident+'.json' if kind == 'archive' else 'status.json')
        metadata.write_text(json.dumps(row))
        return metadata, audio

    def test_schema_adapter_and_resume(self):
        self.record()
        first = run(self.root/'out', self.archive, self.radio)
        second = run(self.root/'out', self.archive, self.radio)
        self.assertEqual(first, second)
        rows = load_manifest(first['manifest'])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['sentence'], '')
        self.assertIsNone(row['prompt'])
        self.assertNotIn('question', row)
        self.assertEqual(row['language'], 'hil')
        self.assertEqual(row['asr_status'], 'pending')
        self.assertEqual(set(row['rights'].values()), {'unknown'})
        self.assertEqual(acquisition_record(row)['canonical_record'], row)
        Path(first['manifest']).write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'immutable snapshot modified'):
            run(self.root/'out', self.archive, self.radio)

    def test_duplicates_and_partial(self):
        self.record('z', 'last')
        self.record('a', 'first', 'radio')
        self.record('b', 'partial', 'radio', 'partial')
        rows, report = build(self.archive, self.radio)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['source_id'], 'a')
        self.assertEqual(report['verified_completed_recordings'], 2)
        self.assertEqual(len(report['exact_byte_duplicates']), 1)
        self.assertEqual(report['exact_byte_duplicates'][0]['duplicate_of'], rows[0]['id'])
        self.assertEqual(len(report['excluded']), 1)

    def test_tamper_size_and_duration(self):
        metadata, audio = self.record()
        original = audio.read_bytes()
        audio.write_bytes(original[:-1]+b'1')
        self.assertIn('checksum', build(self.archive)[1]['excluded'][0]['reason'])
        audio.write_bytes(original+b'1')
        self.assertIn('size', build(self.archive)[1]['excluded'][0]['reason'])
        audio.write_bytes(original)
        row = json.loads(metadata.read_text()); row['duration'] = 20
        metadata.write_text(json.dumps(row))
        self.assertIn('duration', build(self.archive)[1]['excluded'][0]['reason'])

    def test_partial_filename_and_missing_file(self):
        metadata, audio = self.record()
        partial = audio.with_suffix('.part.wav'); audio.rename(partial)
        row = json.loads(metadata.read_text()); row['audio_path'] = str(partial)
        metadata.write_text(json.dumps(row))
        self.assertEqual(build(self.archive)[0], [])
        partial.unlink()
        self.assertEqual(build(self.archive)[0], [])


if __name__ == '__main__':
    unittest.main()
