import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from . import asr_batch as ab
from .schema import VERSION, digest

REV = 'a' * 40


class FakeModel:
    def __init__(self, texts=None):
        self.calls, self.texts = [], texts or {}

    def __call__(self, audios):
        self.calls.append(len(audios))
        return [self.texts.get(len(a), f'salita {len(a)}') for a in audios]


class BatchASRTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.rows = []
        for i, seconds in enumerate([1.0, 2.0, 0.5, 3.0, 31.0, 1.5]):
            wav = self.root / f'a{i}.wav'
            sf.write(wav, np.zeros(int(seconds * 16000), dtype='int16'), 16000, subtype='PCM_16')
            self.rows.append(dict(
                schema_version=VERSION, id=f'r{i}', source_id='s', recording_id='rec', audio_path=str(wav),
                audio_sha256=digest(wav), sentence='', prompt=None, text_is_prompt=False, language='fil',
                languages=['fil'], speech_type='unknown', speaker_id='', duration=seconds, num_words=0,
                split='unassigned', review_status='draft',
                rights={k: 'unknown' for k in ('training', 'commercial', 'redistribution')},
                provenance={'source_url': 'u'}, asr_status='pending'))
        self.manifest = self.root / 'in.jsonl'
        self.write_manifest()
        self.config = ab.make_config('sapinsapin/m', REV, 'fil')
        self.lock = str(self.root / 'gpu.lock')

    def write_manifest(self):
        self.manifest.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))

    def run_batch(self, model, **kw):
        return ab.run(self.manifest, self.root / 'out', self.config, batch_size=2, chunk_rows=2, transcriber=model,
                      gpu_lock=self.lock, **kw)

    def test_complete_run_keeps_input_order_and_contract(self):
        model = FakeModel({8000: ''})  # the 0.5 s clip comes back empty
        summary = self.run_batch(model)
        out = [json.loads(l) for l in (self.root / 'out/manifest.jsonl').read_text().splitlines()]
        self.assertEqual([r['id'] for r in out], ['r0', 'r1', 'r2', 'r3', 'r5'])
        self.assertEqual(summary['not_attempted_over_30s'], ['r4'])
        self.assertEqual((summary['transcribed'], summary['empty'], summary['network_requests']), (4, 1, 0))
        r2 = out[2]
        self.assertEqual((r2['asr_status'], r2['num_words'], r2['review_status']), ('empty', 0, 'draft'))
        self.assertEqual(out[0]['sentence'], 'salita 16000')
        p = out[0]['provenance']
        self.assertEqual(p['asr_config']['engine'], 'local-batch')
        self.assertEqual(p['asr_cache_key'], ab.cache_key(self.rows[0], self.config))
        self.assertEqual(p['source_url'], 'u')
        self.assertEqual(ab.audit(self.manifest, self.root / 'out')['rows'], 5)

    def test_resume_skips_finished_chunks(self):
        self.run_batch(FakeModel())
        (self.root / 'out/chunks/chunk-00002.jsonl').unlink()
        model = FakeModel()
        self.run_batch(model)
        self.assertEqual(model.calls, [1])  # only the removed one-row chunk was recomputed
        self.assertEqual(ab.audit(self.manifest, self.root / 'out')['rows'], 5)

    def test_changed_input_is_refused(self):
        self.run_batch(FakeModel())
        self.rows = self.rows[:-1]
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.run_batch(FakeModel())
        with self.assertRaises(ValueError):
            ab.audit(self.manifest, self.root / 'out')

    def test_audio_mismatch_stops_before_gpu_work(self):
        sf.write(self.rows[1]['audio_path'], np.ones(32000, dtype='int16'), 16000, subtype='PCM_16')
        model = FakeModel()
        with self.assertRaises(ValueError):
            self.run_batch(model)
        self.assertEqual(model.calls, [])  # chunk 0 holds r1, so nothing was decoded
        state = json.loads((self.root / 'out/worker.json').read_text())
        self.assertEqual(state['status'], 'stopped_for_inspection')
        self.assertFalse((self.root / 'out/chunks/chunk-00000.jsonl').exists())

    def test_memory_floor_stops(self):
        with self.assertRaises(RuntimeError):
            self.run_batch(FakeModel(), mem=lambda: 10.0)

    def test_audit_detects_edited_transcript_fields(self):
        self.run_batch(FakeModel())
        path = self.root / 'out/manifest.jsonl'
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        rows[0]['speaker_id'] = 'x'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with self.assertRaises(ValueError):
            ab.audit(self.manifest, self.root / 'out')

    def test_path_map_reads_mirror_but_keeps_original_paths(self):
        mirror = self.root / 'mirror'
        mirror.mkdir()
        for r in self.rows:
            Path(r['audio_path']).rename(mirror / Path(r['audio_path']).name)
        path_map = ((str(self.root) + '/', str(mirror) + '/'),)
        self.run_batch(FakeModel(), path_map=path_map)
        out = [json.loads(l) for l in (self.root / 'out/manifest.jsonl').read_text().splitlines()]
        self.assertEqual(out[0]['audio_path'], self.rows[0]['audio_path'])
        self.assertEqual(ab.audit(self.manifest, self.root / 'out', path_map)['rows'], 5)
        with self.assertRaises(FileNotFoundError):
            ab.audit(self.manifest, self.root / 'out')  # originals are gone; only the mirror holds them

    def test_fallback_replaces_loops_and_flags_unresolved(self):
        loop = 'nagapanguna ' * 30
        calls = []

        def regenerate(indices, temperature):
            calls.append((list(indices), temperature))
            # row 1 is fixed at 0.4; row 2 loops at every temperature
            return ['maayo nga aga' if (i == 1 and temperature >= 0.4) else loop for i in indices]
        out = ab.with_fallback(['salamat po', loop, loop], regenerate, self.config)
        self.assertEqual([t for t, _ in out], ['salamat po', 'maayo nga aga', loop])
        self.assertEqual(out[0][1]['temperature'], 0.0)
        self.assertEqual(out[1][1]['temperature'], 0.4)
        self.assertGreater(out[1][1]['fallback_from_ratio'], 2.4)
        self.assertTrue(out[2][1]['repetition_unresolved'])
        self.assertEqual(calls[0], ([1, 2], 0.2))
        self.assertEqual([c[0] for c in calls[2:]], [[2]] * 3)  # fixed rows are not retried again

    def test_decode_meta_reaches_provenance_and_summary(self):
        class MetaModel:
            def __call__(self, audios):
                return [('x', {'temperature': 0.2, 'compression_ratio': 1.0}) for _ in audios]
        summary = self.run_batch(MetaModel())
        self.assertEqual(summary['temperature_fallbacks'], 5)
        out = json.loads((self.root / 'out/manifest.jsonl').read_text().splitlines()[0])
        self.assertEqual(out['provenance']['asr_decode']['temperature'], 0.2)
        self.assertEqual(ab.audit(self.manifest, self.root / 'out')['rows'], 5)

    def test_config_requires_pinned_revision(self):
        with self.assertRaises(ValueError):
            ab.make_config('sapinsapin/m', 'main', 'fil')


if __name__ == '__main__':
    unittest.main()
