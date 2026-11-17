import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from curation.attach_reviews import attach
from curation.review import Config, input_identity, sha
from curation.schema import digest
from curation.test_review import row, valid_review, envelope


class AttachReviewsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.r = row()
        self.r['num_words'] = len(self.r['sentence'].split())
        audio = self.root / 'audio.flac'
        audio.write_bytes(b'fixture audio bytes')
        self.r.update(audio_path=str(audio), audio_sha256=digest(audio))
        config = Config().identity()
        (self.root / 'config.json').write_text(json.dumps(config))
        fingerprint = sha({'input': input_identity(self.r), 'config': config})
        self.path = self.root / (fingerprint + '.json')
        self.raw_path = self.root / (fingerprint + '.response.json')
        raw = envelope(valid_review(self.r))
        self.review = dict(cache_key=fingerprint, config=config, input_sha256=sha(input_identity(self.r)),
            audio_sha256=self.r['audio_sha256'], sentence_sha256=hashlib.sha256(self.r['sentence'].encode()).hexdigest(),
            status='text_review_only', audio_verified=False, human_approved=False,
            response_sha256=sha(raw), review=valid_review(self.r))
        self.path.write_text(json.dumps(self.review))
        self.raw_path.write_text(json.dumps(raw))

    def tearDown(self):
        self.tmp.cleanup()

    def test_join_preserves_transcript_prompt_rights_and_split(self):
        original = copy.deepcopy(self.r)
        rows, counts = attach([self.r], self.root)
        self.assertEqual(counts['attached'], 1)
        self.assertEqual(self.r, original)
        for field in ['sentence', 'prompt', 'rights', 'split', 'languages', 'audio_sha256']:
            self.assertEqual(rows[0][field], original[field])
        self.assertEqual(rows[0]['review_status'], 'text_reviewed')
        self.assertNotIn('audio_review', rows[0])

    def test_changed_text_does_not_receive_stale_review(self):
        self.r['sentence'] += ' Hello.'
        self.r['num_words'] += 1
        rows, counts = attach([self.r], self.root)
        self.assertEqual(counts['missing'], 1)
        self.assertNotIn('text_review', rows[0])

    def test_tampered_review_raw_and_audio_rejected(self):
        for change in ['status', 'raw', 'audio']:
            with self.subTest(change=change):
                original = self.path.read_bytes(), self.raw_path.read_bytes(), Path(self.r['audio_path']).read_bytes()
                if change == 'status':
                    review = dict(self.review, audio_verified=True)
                    self.path.write_text(json.dumps(review))
                elif change == 'raw':
                    self.raw_path.write_text('{}')
                else:
                    Path(self.r['audio_path']).write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    attach([self.r], self.root)
                self.path.write_bytes(original[0]); self.raw_path.write_bytes(original[1])
                Path(self.r['audio_path']).write_bytes(original[2])

    def test_pending_rows_not_reviewed(self):
        self.r.update(sentence='', num_words=0, asr_status='pending')
        rows, counts = attach([self.r], self.root)
        self.assertEqual(counts['ineligible'], 1)
        self.assertEqual(rows[0], self.r)


if __name__ == '__main__':
    unittest.main()
