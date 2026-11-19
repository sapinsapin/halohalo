import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave

from .ingest_contributions import ingest
from .schema import VERSION, digest, load_manifest

UID = '6f9619ff-8b86-4011-b42d-00cf4fc964ff'


def stage(root, uid=UID, frames=32000, **override):
    folder = Path(root) / uid
    folder.mkdir(parents=True)
    with wave.open(str(folder / 'audio.wav'), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b'\1\0' * frames)
    sentence = 'Nag-meeting kami kahapon.'
    row = dict(schema_version=VERSION, id=uid, source_id='contributor-self', recording_id='r1', audio_path='audio.wav',
               audio_sha256=digest(folder / 'audio.wav'), sentence=sentence, prompt='Ano ginawa mo?',
               text_is_prompt=False, language='fil', languages=['fil', 'en'], variety='Tagalog', speech_type='spontaneous',
               speaker_id=None, duration=frames / 16000, num_words=3, split='unassigned', review_status='draft',
               rights=dict(training='unknown', commercial='unknown', redistribution='unknown', evidence=''),
               consent=dict(version='chika-private-v1', private_deposit=True, evidence='self-recorded'),
               provenance=dict(sentence_sha256=hashlib.sha256(sentence.encode()).hexdigest(),
                               transcript_method='human', review_basis='contributor_supplied_unverified'))
    row.update(override)
    (folder / 'manifest.jsonl').write_text(json.dumps(row) + '\n')
    return folder


class IngestTests(unittest.TestCase):
    def test_good_contribution_becomes_valid_draft_row(self):
        with tempfile.TemporaryDirectory() as d:
            stage(Path(d) / 'stage')
            report = ingest(Path(d) / 'stage', Path(d) / 'out')
            self.assertEqual((report['ingested'], report['rejected']), (1, []))
            rows = load_manifest(Path(d) / 'out' / 'manifest.jsonl')
            self.assertEqual(rows[0]['review_status'], 'draft')
            self.assertEqual(rows[0]['prompt'], 'Ano ginawa mo?')
            self.assertTrue(Path(rows[0]['audio_path']).is_file())
            self.assertEqual(rows[0]['provenance']['staging_prefix'], f'contributor-staging/v1/{UID}')

    def test_bad_contributions_are_rejected_with_reasons(self):
        cases = {
            'no consent': dict(consent=dict(version='chika-private-v1', private_deposit=False, evidence='x')),
            'promoted': dict(review_status='audio_verified'),
            'wrong hash': dict(audio_sha256='0' * 64),
            'edited sentence': dict(sentence='Iba na.', num_words=2),
            'wrong duration': dict(duration=9.0),
        }
        with tempfile.TemporaryDirectory() as d:
            for i, (name, override) in enumerate(cases.items()):
                stage(Path(d) / 'stage', uid=f'{UID[:-2]}{i:02d}', **override)
            report = ingest(Path(d) / 'stage', Path(d) / 'out')
            self.assertEqual(report['ingested'], 0)
            reasons = ' | '.join(r['reason'] for r in report['rejected'])
            for needle in ('consent', 'unassigned drafts', 'audio_sha256', 'sentence changed', 'duration'):
                self.assertIn(needle, reasons)

    def test_folder_id_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            stage(Path(d) / 'stage', id='someone-else')
            report = ingest(Path(d) / 'stage', Path(d) / 'out')
            self.assertIn('folder', report['rejected'][0]['reason'])


if __name__ == '__main__':
    unittest.main()
