import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave

from .cli import export,write_jsonl
from .schema import VERSION,digest


class ExportTests(unittest.TestCase):
    def test_pld_filter_and_pending_label_guard(self):
        import pyarrow.parquet as pq
        import soundfile as sf
        from datasets import load_dataset,Audio
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audio=root/'audio.wav'
            with wave.open(str(audio),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0\0'*16000)
            row=dict(schema_version=VERSION,id='one',source_id='source',recording_id='recording',audio_path=str(audio),
                     audio_sha256=digest(audio),sentence='Maayong aga.',prompt='Source question',text_is_prompt=False,
                     language='hil',languages=['hil'],speech_type='spontaneous',speaker_id='',duration=1.,num_words=2,
                     split='reserved_test',review_status='draft',rights=dict(training='unknown',commercial='unknown',redistribution='unknown'),provenance={})
            manifest=root/'manifest.jsonl';write_jsonl(manifest,[row])
            args=SimpleNamespace(manifest=str(manifest),audio_root=None,output=str(root/'export'),profile='curation',shard_size=10)
            with contextlib.redirect_stdout(io.StringIO()):export(args)
            file=next((root/'export').glob('*.parquet'))
            result=pq.read_table(file).to_pylist()[0]
            self.assertIs(result['text_is_prompt'],False)
            self.assertEqual(result['prompt'],'Source question')
            self.assertEqual(result['sentence'],'Maayong aga.')
            ds=load_dataset('parquet',data_files=str(file),split='train',streaming=True).cast_column('audio',Audio(decode=False))
            filtered=ds.filter(lambda x:not x['text_is_prompt'])
            loaded=next(iter(filtered));samples,rate=sf.read(io.BytesIO(loaded['audio']['bytes']))
            self.assertEqual(rate,16000);self.assertEqual(len(samples),16000)
            # Even an attributed review cannot promote empty pending text to training.
            row.update(sentence='',num_words=0,review_status='audio_verified')
            row['rights'].update(training='allowed',evidence='test permission')
            row['audio_review']=dict(reviewer='tester',reviewed_at='2026-11-14',audio_sha256=row['audio_sha256'],sentence_sha256=hashlib.sha256(b'').hexdigest())
            write_jsonl(manifest,[row]);args.profile='asr';args.output=str(root/'rejected')
            with self.assertRaisesRegex(ValueError,'nonempty verified transcript'):export(args)
            self.assertFalse(Path(args.output).exists())


if __name__=='__main__':unittest.main()
