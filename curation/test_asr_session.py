import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import soundfile as sf

from .test_asr import ASRTest
from .asr_session import Config, Session, transcript
from .schema import digest


class PersistentTest(ASRTest):
    # Reuse only the fixture, not the single-window test cases.
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.cfg = Config(endpoint=self.cfg.endpoint, timeout=2)
        self.session = Session(self.cfg, 'secret-key', self.root/'persistent')
        self.clock = 0
        self.finals = 0

    async def asyncTearDown(self):
        await self.session.close()
        await super().asyncTearDown()

    async def handler(self, ws):
        self.connections += 1
        await ws.send(json.dumps(dict(type='hello', model='wrong' if self.mode == 'model' else self.cfg.model)))
        count = 0
        try:
            async for message in ws:
                self.requests.append(message)
                if isinstance(message, bytes):
                    count += len(message)//2
                    continue
                msg = json.loads(message)
                if msg['type'] != 'finalize':
                    continue
                self.finals += 1
                if self.mode == 'usage':
                    await ws.send(json.dumps(dict(type='usage',audio_ms=(self.clock+count)//16,credits=0)))
                if self.mode == 'disconnect':
                    return
                if self.mode in {'fatal', 'nonfatal', 'quota'}:
                    await ws.send(json.dumps(dict(type='error', code='quota_exhaustion' if self.mode=='quota' else 'final_failed', fatal=self.mode=='fatal')))
                async def result(fid, start, n, text):
                    await ws.send(json.dumps(dict(type='result', final=True, text=text, start_ms=start//16,
                        duration_ms=n//16, metadata=dict(model='canonical-model', finalize_id=fid))))
                    if self.mode == 'usage':
                        await ws.send(json.dumps(dict(type='usage',audio_ms=(self.clock+count)//16,credits=0)))
                    await ws.send(json.dumps(dict(type='finalize_end', id=fid)))
                if self.mode == 'auto':
                    half = count//2
                    await result('auto-'+str(self.finals), self.clock, half, 'Una')
                    await result(msg['id'], self.clock+half, count-half, '')
                elif self.mode == 'duplicate':
                    await result('auto-1', self.clock, count//2, 'x')
                    await ws.send(json.dumps(dict(type='finalize_end', id='auto-1')))
                    await result(msg['id'], self.clock+count//2, count-count//2, 'y')
                else:
                    await result('wrong' if self.mode=='unmatched' else msg['id'], self.clock,
                                 count//2 if self.mode=='truncated' else count, '' if self.mode=='empty' else 'Maayong aga')
                self.clock += count
                count = 0
        except Exception:
            pass

    def row2(self, frames=1607):
        path = self.root/'second.flac'
        sf.write(path, np.ones(frames, dtype=np.int16), 16000, subtype='PCM_16')
        return {**self.row, 'id': 'two', 'audio_path': str(path), 'audio_sha256': digest(path), 'duration': frames/16000}

    async def test_persistent_clock_and_cache(self):
        firstrow = self.row2()
        first = await self.session.transcribe_one(firstrow)
        self.assertEqual(await self.session.transcribe_one(firstrow), first)
        second = await self.session.transcribe_one(self.row)
        self.assertEqual(self.connections, 1)
        self.assertEqual(self.finals, 2)
        self.assertEqual(self.session.samples, 3207)
        raw = json.loads((self.session.root/(second['cache_key']+'.response.json')).read_text())
        self.assertEqual(raw['session_start_sample'], 1607)
        self.assertEqual(raw['events'][0]['event']['start_ms'], 100)
        self.assertEqual(transcript(raw, self.cfg)[1][0]['start_ms'], -.4375)
        self.assertNotIn('SECRET QUESTION', str(self.requests))

    async def test_usage_is_retained_without_becoming_coverage(self):
        self.mode = 'usage'
        result = await self.session.transcribe_one(self.row)
        raw = json.loads((self.session.root/(result['cache_key']+'.response.json')).read_text())
        self.assertEqual(raw['events'][0]['event']['type'], 'usage')
        self.assertEqual(transcript(raw, self.cfg)[0], 'Maayong aga')
        raw['events'][1]['event']['duration_ms'] = 1
        with self.assertRaisesRegex(ValueError, 'truncated coverage'):
            transcript(raw, self.cfg)
        raw['events'][0]['event']['truncated'] = True
        with self.assertRaisesRegex(ValueError, 'server truncation'):
            transcript(raw, self.cfg)

    async def test_periodic_manifest_flush_and_cache_only_replay(self):
        from .asr_session import run
        from .cli import write_jsonl
        manifest=self.root/'batch.jsonl'
        rows=[self.row,self.row2(),{**self.row,'id':'three'}]
        write_jsonl(manifest,rows)
        key=self.root/'key';key.write_text('secret-key')
        args=SimpleNamespace(endpoint=self.cfg.endpoint,model=self.cfg.model,language=self.cfg.language,
            timeout=self.cfg.timeout,pace=0,max_items=3,manifest=str(manifest),output=str(self.root/'batch'),
            key_file=str(key),checkpoint_every=2)
        writes=[]
        def tracked(path,values):
            writes.append(len(values));write_jsonl(path,values)
        with patch('curation.asr_session.write_jsonl',side_effect=tracked):
            await run(args)
        self.assertEqual(writes,[2,3])
        summary=json.loads((self.root/'batch/summary.json').read_text())
        self.assertEqual(summary['manifest_rows'],3)
        with patch('curation.asr_session.websockets.connect',side_effect=AssertionError('network forbidden')):
            await run(args)
        self.mode='fatal';args.output=str(self.root/'failed-batch')
        with self.assertRaises(SystemExit):await run(args)
        summary=json.loads((self.root/'failed-batch/summary.json').read_text())
        self.assertEqual(summary['manifest_rows'],0)
        self.assertTrue(summary['stopped_early'])
        self.assertEqual((self.root/'failed-batch/manifest.jsonl').read_text(),'')

    async def test_handshake_binding_and_word_clock_separation(self):
        result=await self.session.transcribe_one(self.row)
        raw=json.loads((self.session.root/(result['cache_key']+'.response.json')).read_text())
        raw['events'][0]['event']['words']=[{'start_ms':1234,'end_ms':1250,'word':'fixture'}]
        self.assertNotIn('words',transcript(raw,self.cfg)[1][0])
        raw['session_id']='another-session'
        with self.assertRaisesRegex(ValueError,'handshake identity'):
            transcript(raw,self.cfg)

    async def test_recover_paid_commit(self):
        first = await self.session.transcribe_one(self.row)
        (self.session.root/(first['cache_key']+'.response.json')).unlink()
        (self.session.root/(first['cache_key']+'.json')).unlink()
        self.assertEqual(await self.session.transcribe_one(self.row), first)
        self.assertEqual(self.finals, 1)
        self.assertEqual(self.session.samples, 1600)

    async def test_all_failures_discard_session(self):
        for mode in ['model', 'fatal', 'nonfatal', 'quota', 'disconnect', 'unmatched', 'duplicate', 'truncated']:
            with self.subTest(mode=mode):
                self.mode = mode
                session = Session(self.cfg, 'secret-key', self.root/mode)
                with self.assertRaises(RuntimeError):
                    await session.transcribe_one(self.row)
                self.assertTrue(session.broken)
                self.assertIsNone(session.ws)
                with self.assertRaises(RuntimeError):
                    await session.transcribe_one(self.row2())
                self.assertFalse(list((self.root/mode).glob('*.response.json')))
                # A new process must not repay an uncertain prior attempt.
                replacement = Session(self.cfg, 'secret-key', self.root/mode)
                with self.assertRaises(RuntimeError):
                    await replacement.transcribe_one(self.row)

    async def test_auto_and_empty(self):
        self.mode = 'auto'
        self.assertEqual((await self.session.transcribe_one(self.row2()))['sentence'], 'Una')
        self.mode = 'empty'
        self.assertEqual((await self.session.transcribe_one(self.row))['status'], 'empty')

    async def test_previous_success_survives_later_failure(self):
        first = await self.session.transcribe_one(self.row)
        self.mode = 'quota'
        with self.assertRaises(RuntimeError):
            await self.session.transcribe_one(self.row2())
        self.assertTrue((self.session.root/(first['cache_key']+'.response.json')).exists())
        self.assertEqual(self.session.samples, 1600)

    async def test_per_item_lock(self):
        import fcntl
        from .review import sha
        fingerprint = sha(dict(audio_sha256=self.row['audio_sha256'], config=self.cfg.identity()))
        with (self.session.root/(fingerprint+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                await self.session.transcribe_one(self.row)
        self.assertEqual(self.connections, 0)

    async def test_cache_audio_verification(self):
        await self.session.transcribe_one(self.row)
        sf.write(self.row['audio_path'], np.ones(1600, dtype=np.int16), 16000, subtype='PCM_16')
        with self.assertRaises(ValueError):
            await self.session.transcribe_one(self.row)
        self.assertEqual(self.finals, 1)

    async def test_complete_before_response_write_crash(self):
        from .review import atomic as real_atomic
        def interrupted(path, value):
            if str(path).endswith('.response.json'):
                raise OSError('simulated crash')
            real_atomic(path, value)
        with patch('curation.asr_session.atomic', side_effect=interrupted):
            with self.assertRaises(OSError):
                await self.session.transcribe_one(self.row)
        self.assertEqual((await self.session.transcribe_one(self.row))['sentence'], 'Maayong aga')
        self.assertEqual(self.finals, 1)


# Fixture inheritance should not execute the legacy transport's tests here.
for name in list(ASRTest.__dict__):
    if name.startswith('test_') and name not in PersistentTest.__dict__:
        setattr(PersistentTest, name, None)
del ASRTest

if __name__ == '__main__':
    unittest.main()
