import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from .prepare import config,connect,prepare_one,export_manifest
from .schema import digest,load_manifest


class PrepareTests(unittest.TestCase):
    def test_sample_exact_windows_resume_and_decoded_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);audio=root/'original.wav'
            # Nonintegral final window; exact sample coverage must survive FLAC encoding.
            pcm=b''.join(struct.pack('<h',(i%100)*100) for i in range(40501))
            with wave.open(str(audio),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm)
            record={'id':'r1','source_id':'show','path':str(audio),'sha256':digest(audio),
                    'duration':40501/16000,'language':'hil','url':'https://example.org/audio.wav'}
            out=root/'prepared';db=connect(out);cfg=config(1)
            r=prepare_one(record,out,db,cfg,0);self.assertEqual(r['windows'],3)
            self.assertEqual(prepare_one(record,out,db,cfg,0)['status'],'cached')
            export_manifest(db,out);rows=load_manifest(out/'manifest.jsonl')
            bystart=sorted(rows,key=lambda r:r['start'])
            self.assertEqual(bystart[0]['start'],0)
            self.assertEqual(bystart[-1]['end'],40501/16000)
            self.assertEqual(sum(round(r['duration']*16000) for r in rows),40501)
            self.assertTrue(all(r['sentence']=='' and r['asr_status']=='pending' for r in rows))
            for a,b in zip(bystart,bystart[1:]):self.assertEqual(a['end'],b['start'])
            second=dict(record,id='r2');prepare_one(second,out,db,cfg,0)
            summary=export_manifest(db,out);self.assertEqual(summary['exact_decoded_duplicate_groups'][0]['recordings'],2)
            # Missing derivative invalidates cache rather than pretending success.
            Path(rows[0]['audio_path']).unlink()
            self.assertEqual(prepare_one(record,out,db,cfg,0)['status'],'prepared')
            record['sha256']='0'*64
            with self.assertRaises(ValueError):prepare_one(record,out,db,cfg,0)
            db.close()


if __name__=='__main__':unittest.main()
