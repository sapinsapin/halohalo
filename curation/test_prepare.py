import contextlib
import io
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from .prepare import config,connect,prepare_one,export_manifest,parser,round_robin_sources
from .schema import VERSION,digest,load_manifest,validate


class PrepareTests(unittest.TestCase):
    def test_round_robin_preserves_within_source_order_without_dropping(self):
        records=[{'source_id':s,'id':i} for s,i in [('b',1),('b',2),('a',3),('c',4),('a',5),('b',6)]]
        result=list(round_robin_sources(iter(records)))
        self.assertEqual([r['id'] for r in result],[3,1,4,5,2,6])
        self.assertEqual(sorted(r['id'] for r in result),sorted(r['id'] for r in records))
        self.assertEqual(list(round_robin_sources([])),[])
        self.assertTrue(parser().parse_args(['--rss-root','r','--output','o','--round-robin-sources']).round_robin_sources)

    def test_input_selection(self):
        p=parser()
        for args in (['--output','out'], ['--output','out','--manifest','m','--rss-root','r']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p.parse_args(args)
        self.assertEqual(p.parse_args(['--output','out','--manifest','m']).manifest,'m')
        self.assertEqual(p.parse_args(['--output','out','--rss-root','r']).rss_root,'r')

    def test_canonical_metadata_survives_windows(self):
        from .import_media import acquisition_record
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);audio=root/'original.wav'
            with wave.open(str(audio),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
                w.writeframes(b'\0\0'*32501)
            canonical=dict(schema_version=VERSION,id='manifest-row',source_id='archive',
                recording_id='recording',audio_path=str(audio),audio_sha256=digest(audio),
                sentence='',prompt='What did the interviewer ask?',text_is_prompt=False,language='unknown',
                audio_bytes=audio.stat().st_size,alignment=[{'start':99,'end':100,'word':'stale'}],
                start=99,end=100,
                languages=['unknown','hil'],language_annotation='unverified source claim',
                speech_type='unknown',speaker_id='',duration=32501/16000,num_words=0,
                split='reserved_test',review_status='draft',
                rights={'training':'unknown','commercial':'denied','redistribution':'unknown',
                        'evidence':'source metadata only'},
                provenance={'source_url':'https://example.org/original','import_kind':'archive',
                            'acquisition_metadata_sha256':'a'*64,'nested':{'preserved':True}},
                source_title='Archive title',flags=['source_claim_unverified'])
            before=copy.deepcopy(canonical)
            record=acquisition_record(canonical)
            out=root/'prepared';db=connect(out);cfg=config(1)
            result=prepare_one(record,out,db,cfg,0)
            self.assertEqual(result['windows'],3)
            self.assertEqual(prepare_one(record,out,db,cfg,0)['status'],'cached')
            export_manifest(db,out)
            for row in load_manifest(out/'manifest.jsonl'):
                validate(row)
                self.assertEqual(digest(row['audio_path']),row['audio_sha256'])
                for key in ('rights','split','language','languages','language_annotation','source_title'):
                    self.assertEqual(row[key],canonical[key])
                self.assertEqual(row['recording_id'],'recording')
                self.assertEqual(row['provenance']['nested'],{'preserved':True})
                self.assertEqual(row['provenance']['parent_record_provenance'],canonical['provenance'])
                self.assertEqual(row['provenance']['parent_manifest_id'],'manifest-row')
                self.assertEqual(row['sentence'],'')
                self.assertIsNone(row['prompt'])
                self.assertIs(row['text_is_prompt'],False)
                self.assertEqual(row['audio_bytes'],Path(row['audio_path']).stat().st_size)
                self.assertNotIn('alignment',row)
                self.assertLess(row['end'],3)
                parent=row['provenance']['parent_record']
                self.assertEqual(parent['prompt'],canonical['prompt'])
                self.assertEqual(parent['audio_bytes'],canonical['audio_bytes'])
                self.assertEqual(parent['alignment'],canonical['alignment'])
                self.assertEqual(row['asr_status'],'pending')
            self.assertEqual(canonical,before)
            changed=copy.deepcopy(canonical);changed['split']='train'
            with self.assertRaisesRegex(ValueError,'metadata changed'):
                prepare_one(acquisition_record(changed),out,db,cfg,0)
            invalid=copy.deepcopy(canonical);invalid['split']='unknown'
            with self.assertRaises(ValueError):acquisition_record(invalid)
            db.close()

    def test_sample_exact_windows_resume_and_decoded_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);audio=root/'original.wav'
            # Nonintegral final window; exact sample coverage must survive FLAC encoding.
            pcm=b''.join(struct.pack('<h',(i%100)*100) for i in range(40501))
            with wave.open(str(audio),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm)
            record={'id':'r1','source_id':'show','path':str(audio),'sha256':digest(audio),
                    'duration':40501/16000,'language':'bik','url':'https://example.org/audio.wav',
                    'source_metadata':json.dumps({'language_varieties':['bcl','bto'],'language_verified':False})}
            out=root/'prepared';db=connect(out);cfg=config(1)
            r=prepare_one(record,out,db,cfg,0);self.assertEqual(r['windows'],3)
            self.assertEqual(prepare_one(record,out,db,cfg,0)['status'],'cached')
            export_manifest(db,out);rows=load_manifest(out/'manifest.jsonl')
            self.assertTrue(all(r['provenance']['acquisition_source_metadata']['language_varieties']==['bcl','bto'] for r in rows))
            self.assertTrue(all(r['language']=='bik' for r in rows))
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
