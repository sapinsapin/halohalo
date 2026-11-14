import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from curation.schema import VERSION, digest
from curation.vad import intervals, run


class VadTests(unittest.TestCase):
    def test_continuous_coverage_cap_and_padding(self):
        spans=intervals([.9]*2000,1024000)
        self.assertEqual(spans[0][0],0)
        self.assertEqual(spans[-1][1],1024000)
        self.assertEqual(sum(b-a for a,b in spans),1024000)
        self.assertTrue(all(b-a<=27*16000 for a,b in spans))
        self.assertTrue(all(a[1]==b[0] for a,b in zip(spans,spans[1:])))

    def test_silence_and_merged_padding(self):
        self.assertEqual(intervals([0]*100,51200),[])
        spans=intervals([.9]*20+[0]*4+[.9]*20,44*512)
        self.assertEqual(spans,[(0,44*512)])
        self.assertEqual(intervals([.9]*4+[0]*20,24*512),[])

    def test_gap_cut(self):
        probs=[.9]*1000
        probs[820]=.1  # 26.24 seconds: short gap inside continuous region
        spans=intervals(probs,512000)
        self.assertEqual(spans[0][1],820*512)
        self.assertEqual(spans[1][0],820*512)

    def test_snapshot_review_offsets_and_exact_pcm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); audio=root/'parent.flac'
            data=np.arange(32000,dtype=np.int16)
            sf.write(audio,data,16000,subtype='PCM_16')
            row=dict(schema_version=VERSION,id='parent',source_id='s',recording_id='r',
                     audio_path=str(audio),audio_sha256=digest(audio),sentence='old transcript',
                     num_words=2,prompt='original question?',text_is_prompt=False,language='hil',
                     languages=['hil'],speech_type='unknown',speaker_id='',duration=2,
                     split='reserved_test',review_status='audio_verified',
                     rights={'training':'unknown','commercial':'unknown','redistribution':'unknown'},
                     audio_review={'reviewer':'test','reviewed_at':'today','audio_sha256':digest(audio),
                                   'sentence_sha256':hashlib.sha256(b'old transcript').hexdigest()},
                     provenance={'start_sample':480000,'end_sample':512000,'offset_sample_rate':16000})
            manifest=root/'input.jsonl'; manifest.write_text(json.dumps(row)+'\n')
            with patch('curation.vad.Silero') as model:
                model.return_value.probabilities.return_value=[.9]*63
                first=run(manifest,root/'out','unused',1,1)
                self.assertEqual(first,run(manifest,root/'out','unused',1,1))
                out=json.loads((root/'out/manifest.jsonl').read_text())
                self.assertEqual(out['start'],30)
                self.assertEqual(out['end'],32)
                self.assertEqual(out['split'],'reserved_test')
                self.assertEqual(out['prompt'],row['prompt'])
                self.assertEqual(out['sentence'],'')
                self.assertEqual(out['review_status'],'draft')
                self.assertNotIn('audio_review',out)
                self.assertEqual(out['provenance']['parent_record'],row)
                np.testing.assert_array_equal(sf.read(out['audio_path'],dtype='int16')[0],data)
                with self.assertRaisesRegex(ValueError,'input/config changed'):
                    run(manifest,root/'out','unused',1,1,.6)
                row['rights']['redistribution']='denied'
                manifest.write_text(json.dumps(row)+'\n')
                with self.assertRaisesRegex(ValueError,'input/config changed'):
                    run(manifest,root/'out','unused',1,1)
                row['rights']['redistribution']='unknown'; manifest.write_text(json.dumps(row)+'\n')
                Path(out['audio_path']).write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'stale output audio'):
                    run(manifest,root/'out','unused',1,1)


if __name__=='__main__': unittest.main()
