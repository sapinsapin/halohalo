import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from curation.schema import digest
from curation.test_review import row
from curation.tts_readiness import audit


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.recipe = {'language': 'fil', 'pilot_data_gate': {
            'min_train_hours': .001, 'min_validation_hours': .001,
            'min_test_hours': .001, 'min_clip_seconds': 2, 'max_clip_seconds': 15}}
        self.rows = []
        for i, split in enumerate(('train', 'validation', 'test')):
            r = row()
            audio = self.root / f'{i}.flac'
            audio.write_bytes(f'independent audio fixture {i}'.encode())
            r.update(id=str(i), source_id='source', recording_id=str(i), speaker_id=str(i),
                     language='fil', languages=['fil', 'en'], sentence=f'Test code switching {i}',
                     num_words=4, duration=4, split=split, audio_path=str(audio),
                     audio_sha256=digest(audio), review_status='audio_verified',
                     rights={'training':'allowed','commercial':'unknown','redistribution':'unknown','evidence':'fixture'})
            self.bind_review(r)
            self.rows.append(r)

    def tearDown(self):
        self.tmp.cleanup()

    def bind_review(self, r):
        r['audio_review']={'reviewer':'fixture','reviewed_at':'2026-11-15',
                           'audio_sha256':r['audio_sha256'],
                           'sentence_sha256':hashlib.sha256(r['sentence'].encode()).hexdigest()}

    def run_audit(self):
        p=self.root/'manifest.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        return audit(p,self.recipe)

    def test_reviewed_codeswitching_data_passes_data_gate_only(self):
        result=self.run_audit()
        self.assertTrue(result['data_gate_passed'])
        self.assertFalse(result['training_launched'])

    def test_text_review_does_not_replace_audio_review(self):
        self.rows[0]['review_status']='text_reviewed'
        result=self.run_audit()
        self.assertFalse(result['data_gate_passed'])
        self.assertEqual(result['exclusion_reasons']['audio_review_pending'],1)

    def test_duplicate_audio_cannot_inflate_hours(self):
        r=copy.deepcopy(self.rows[0]);r['id']='duplicate';self.rows.append(r)
        with self.assertRaisesRegex(ValueError,'duplicate eligible audio'): self.run_audit()

    def test_transcript_leakage_normalizes_case_and_whitespace(self):
        self.rows[1]['sentence']=' TEST  CODE SWITCHING 0 '
        self.bind_review(self.rows[1])
        with self.assertRaisesRegex(ValueError,'transcript leakage'): self.run_audit()

    def test_changed_audio_fails_hash_check(self):
        Path(self.rows[0]['audio_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'): self.run_audit()

    def test_stale_review_fails_schema(self):
        self.rows[0]['sentence']='Changed transcript here now'
        with self.assertRaisesRegex(ValueError,'stale audio review'): self.run_audit()

    def test_unknown_training_rights_excluded(self):
        self.rows[0]['rights']['training']='unknown'
        self.assertFalse(self.run_audit()['data_gate_passed'])


if __name__ == '__main__':
    unittest.main()
