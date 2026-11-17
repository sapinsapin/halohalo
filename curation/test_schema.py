import copy
import hashlib
import unittest
from .schema import VERSION, validate, check_splits


def row():
    return dict(schema_version=VERSION,id='one',source_id='source',recording_id='recording',
                audio_path='/test.wav',audio_sha256='a'*64,sentence='Actually, na-miss ko ang pagkaon sa balay.',
                prompt='What do you miss?',text_is_prompt=False,language='hil',languages=['hil','eng'],
                speech_type='spontaneous',speaker_id='speaker',duration=2.,num_words=7,
                split='train',review_status='draft',rights=dict(training='unknown',commercial='unknown',redistribution='unknown'),provenance={})


class SchemaTests(unittest.TestCase):
    def test_codeswitch_preserved(self):
        r=row(); self.assertEqual(validate(r)['sentence'],r['sentence'])
    def test_prompt_rejected(self):
        r=row();r['text_is_prompt']=True
        with self.assertRaises(ValueError):validate(r)
    def test_claimed_verification_needs_evidence(self):
        r=row();r['review_status']='audio_verified'
        with self.assertRaises(ValueError):validate(r)
        r['audio_review']=dict(reviewer='tester',reviewed_at='2026-11-14',audio_sha256=r['audio_sha256'],sentence_sha256=hashlib.sha256(r['sentence'].encode()).hexdigest())
        validate(r)
        r['sentence']='Changed';r['num_words']=1
        with self.assertRaises(ValueError):validate(r)
    def test_reserved_test_speaker_leak(self):
        a=row();b=copy.deepcopy(a);b.update(id='two',recording_id='other',audio_sha256='b'*64,split='reserved_test')
        with self.assertRaises(ValueError):check_splits([a,b])
    def test_identical_audio_leak_across_sources(self):
        a=row();b=copy.deepcopy(a);b.update(id='two',source_id='other',speaker_id='',split='test')
        with self.assertRaises(ValueError):check_splits([a,b])


if __name__=='__main__':unittest.main()
