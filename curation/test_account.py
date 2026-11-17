import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from curation.account import account, source_interval, union_seconds
from curation.cli import write_jsonl
from curation.schema import digest
from curation.test_review import row

class AccountTests(unittest.TestCase):
    def test_interval_union_and_bad_offsets(self):
        self.assertEqual(union_seconds([(Fraction(0),Fraction(20)),(Fraction(10),Fraction(30)),(Fraction(30),Fraction(40))]),40)
        r=row();r['duration']=1;r['provenance'].update(original_audio_sha256='a'*64,start_sample=0,end_sample=16001,offset_sample_rate=16000)
        source_interval(r)
        r['provenance']['end_sample']=17000
        with self.assertRaises(ValueError):source_interval(r)

    def test_repeated_manifests_overlaps_pending_and_language_conflicts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audio=root/'audio';audio.write_bytes(b'fixture')
            r=row();r.update(audio_path=str(audio),audio_sha256=digest(audio),duration=20,num_words=len(r['sentence'].split()))
            r['provenance'].update(original_audio_sha256='b'*64,start_sample=0,end_sample=320000,offset_sample_rate=16000)
            other=copy.deepcopy(r);other['id']='second';other['provenance'].update(start_sample=160000,end_sample=480000)
            pending=copy.deepcopy(r);pending.update(id='pending',sentence='',num_words=0,asr_status='pending');pending['provenance'].update(start_sample=480000,end_sample=800000)
            manifest=root/'manifest.jsonl';write_jsonl(manifest,[r,other,pending])
            result=account([('hil',manifest),('hil',manifest),('ceb',manifest)])
            self.assertEqual(result['groups']['hil']['covered_source_hours']['draft_transcribed'],30/3600)
            self.assertEqual(result['groups']['hil']['covered_source_hours']['prepared'],50/3600)
            self.assertEqual(result['groups']['hil']['covered_source_hours']['audio_verified'],0)
            self.assertEqual(result['cross_group_source_hashes']['b'*64],['ceb','hil'])
            self.assertFalse(result['target_completion_assessed'])
            audio.write_bytes(b'tampered')
            with self.assertRaises(ValueError):account([('hil',manifest)])

if __name__=='__main__':unittest.main()
