import array
import json
import math
import shutil
import unittest
import wave

from curation.schema import digest
from curation import test_tts_readiness
from curation.tts_export import export


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_tts_readiness.ReadinessTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        for i, row in enumerate(self.fixture.rows):
            path = self.root / f'{i}.wav'
            with wave.open(str(path), 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes(array.array('h', (int(1000 * math.sin(2*math.pi*(220+i*20)*n/16000))
                                                 for n in range(64000))).tobytes())
            row.update(audio_path=str(path), audio_sha256=digest(path))
            self.fixture.bind_review(row)
        self.manifest = self.root / 'manifest.jsonl'
        self.recipe = self.root / 'recipe.json'
        self.recipe.write_text(json.dumps(self.fixture.recipe))

    def run_export(self):
        self.manifest.write_text(''.join(json.dumps(r)+'\n' for r in self.fixture.rows))
        return export(self.manifest, self.recipe, self.root / 'export')

    def test_real_waveform_export_preserves_text_splits_and_hashes(self):
        result = self.run_export()
        self.assertEqual(result['counts'], {'train': 1, 'validation': 1, 'test': 1})
        for original in self.fixture.rows:
            row = json.loads((self.root / 'export' / (original['split']+'.jsonl')).read_text())
            self.assertEqual(row['conversations'][0]['text'], original['sentence'])
            self.assertEqual(digest(row['conversations'][0]['wav']), row['provenance']['audio_sha256'])
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_export()

    def test_unreviewed_input_produces_no_export(self):
        self.fixture.rows[0]['review_status'] = 'text_reviewed'
        with self.assertRaisesRegex(ValueError, 'only eligible'):
            self.run_export()
        self.assertFalse((self.root / 'export').exists())

    def test_duration_mismatch_removes_partial_output(self):
        self.fixture.rows[0]['duration'] = 5
        with self.assertRaisesRegex(ValueError, 'duration'):
            self.run_export()
        self.assertFalse(list(self.root.glob('.tts-export-*')))


if __name__ == '__main__':
    unittest.main()
