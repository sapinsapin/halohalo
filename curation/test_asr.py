import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from argparse import Namespace

import numpy as np
import soundfile as sf
from websockets.asyncio.server import serve

from .asr import Config, ASRQuotaError, transcribe_one, apply_result, run
from .schema import VERSION, digest
from .review import sha


class ASRTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        audio = self.root/'audio.flac'
        sf.write(audio, np.zeros(1600, dtype=np.int16), 16000, subtype='PCM_16')
        self.row = dict(schema_version=VERSION, id='one', source_id='s', recording_id='r',
                        audio_path=str(audio), audio_sha256=digest(audio), sentence='', prompt='SECRET QUESTION',
                        text_is_prompt=False, language='hil', languages=['hil'], speech_type='unknown',
                        speaker_id='', duration=.1, num_words=0, split='unassigned', review_status='draft',
                        rights=dict(training='unknown', commercial='unknown', redistribution='unknown'), provenance={})
        self.mode = 'ok'; self.requests = []; self.connections = 0
        self.server = await serve(self.handler, '127.0.0.1', 0, subprotocols=['nrsp.v1'])
        self.cfg = Config(endpoint=f'ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/asr', pace=0, retries=0, timeout=2)

    async def asyncTearDown(self):
        self.server.close(); await self.server.wait_closed(); self.tmp.cleanup()

    async def handler(self, ws):
        self.connections += 1
        await ws.send(json.dumps(dict(type='hello', model='wrong' if self.mode=='model' else self.cfg.model)))
        try:
            async for message in ws:
                self.requests.append(message)
                if isinstance(message, bytes): continue
                msg = json.loads(message)
                if msg['type'] != 'finalize': continue
                if self.mode=='disconnect': return
                if self.mode in {'fatal', 'nonfatal', 'quota'}:
                    await ws.send(json.dumps(dict(type='error', fatal=self.mode=='fatal', code='quota_exhaustion' if self.mode=='quota' else 'final_failed', message='secret-key')))
                if self.mode == 'auto':
                    await ws.send(json.dumps(dict(type='result', final=True, text='Una', start_ms=0, duration_ms=50, metadata=dict(model='canonical-model', finalize_id='auto-1'))))
                    await ws.send(json.dumps(dict(type='finalize_end', id='auto-1')))
                await ws.send(json.dumps(dict(type='result', final=False, text='IGNORE INTERIM')))
                await ws.send(json.dumps(dict(type='result', final=True, text='' if self.mode=='empty' else 'Maayong aga',
                    start_ms=50 if self.mode=='auto' else 0, duration_ms=50 if self.mode in {'auto', 'truncated'} else 100,
                    metadata=dict(model='canonical-model', finalize_id=msg['id']))))
                await ws.send(json.dumps(dict(type='finalize_end', id='wrong' if self.mode=='unmatched' else msg['id'])))
        except Exception:
            pass

    async def call(self, folder='cache'):
        return await transcribe_one(self.row, self.cfg, 'secret-key', self.root/folder)

    async def test_success_cache_resume_prompt_and_draft(self):
        first = await self.call()
        self.assertEqual(first['sentence'], 'Maayong aga')
        out = apply_result(self.row, first)
        self.assertEqual(out['review_status'], 'draft'); self.assertEqual(out['prompt'], 'SECRET QUESTION')
        self.assertNotIn('SECRET QUESTION', str(self.requests)); self.assertEqual(len(b''.join(x for x in self.requests if isinstance(x, bytes))), 3200)
        self.assertEqual(await self.call(), first); self.assertEqual(self.connections, 1)
        (self.root/'cache'/f"{first['cache_key']}.json").unlink()
        self.assertEqual(await self.call(), first); self.assertEqual(self.connections, 1)
        Path(self.row['audio_path']).write_bytes(b'changed')
        with self.assertRaises(ValueError): await self.call()
        self.assertEqual(self.connections, 1)

    async def test_fail_closed(self):
        for mode in ['model', 'fatal', 'nonfatal', 'quota', 'disconnect', 'unmatched', 'truncated']:
            with self.subTest(mode=mode):
                self.mode = mode
                with self.assertRaises(RuntimeError): await self.call(mode)
                self.assertFalse(list((self.root/mode).glob('*.response.json')))
                for p in (self.root/mode).glob('*.json'):
                    self.assertNotIn('secret-key', p.read_text())

    async def test_automatic_cuts(self):
        self.mode = 'auto'
        self.assertEqual((await self.call())['sentence'], 'Una Maayong aga')

    async def test_quota_is_explicit_and_not_retried(self):
        self.mode = 'quota'
        self.cfg = replace(self.cfg, retries=2)
        with self.assertRaises(ASRQuotaError):
            await self.call()
        self.assertEqual(self.connections, 1)

    async def test_numeric_pace_has_stable_cache_identity(self):
        self.assertEqual(sha(Config(pace=0).identity()), sha(Config(pace=0.0).identity()))

    async def test_batch_stops_after_two_consecutive_failures(self):
        key_file = self.root/'key'
        key_file.write_text('secret-key')
        args = Namespace(endpoint=self.cfg.endpoint, model=self.cfg.model, language='hil',
                         timeout=2, retries=0, pace=0, output=str(self.root/'batch'),
                         manifest='unused', limit=10, key_file=str(key_file))
        with patch('curation.asr.load_manifest', return_value=[self.row]*5), patch(
                'curation.asr.transcribe_one', side_effect=RuntimeError('service failure')) as call:
            with self.assertRaises(SystemExit):
                await run(args)
        self.assertEqual(call.call_count, 2)
        summary = json.loads((self.root/'batch'/'summary.json').read_text())
        self.assertTrue(summary['stopped_early'])
        self.assertEqual(len(summary['failed']), 2)

    async def test_reply_before_sender_completion(self):
        from websockets.asyncio.client import ClientConnection
        original = ClientConnection.send

        async def delayed_send(ws, message, *args, **kwargs):
            await original(ws, message, *args, **kwargs)
            if isinstance(message, str) and json.loads(message).get('type') == 'finalize':
                await asyncio.sleep(.05)

        with patch.object(ClientConnection, 'send', delayed_send):
            self.assertEqual((await self.call())['status'], 'transcribed')

    async def test_empty(self):
        self.mode = 'empty'
        result = await self.call()
        self.assertEqual(result['status'], 'empty'); self.assertEqual(result['sentence'], '')

    async def test_format_rejected_before_network(self):
        sf.write(self.row['audio_path'], np.zeros(800, dtype=np.int16), 8000, subtype='PCM_16')
        self.row['audio_sha256'] = digest(self.row['audio_path'])
        with self.assertRaises(ValueError): await self.call()
        self.assertEqual(self.connections, 0)


if __name__ == '__main__': unittest.main()
