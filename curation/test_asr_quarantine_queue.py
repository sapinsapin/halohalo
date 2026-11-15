import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from . import asr_quarantine_queue as qq
from .schema import VERSION, digest


def row(i):
    return dict(schema_version=VERSION, id=f'r{i}', source_id='s', recording_id='rec', audio_path='/unused.wav',
                audio_sha256=f'{i:064d}', sentence='', prompt=None, text_is_prompt=False, language='hil',
                languages=['hil'], speech_type='unknown', speaker_id='', duration=1.0, num_words=0, split='unassigned',
                review_status='draft', rights={k: 'unknown' for k in ('training', 'commercial', 'redistribution')},
                provenance={}, asr_status='pending')


class QuarantineQueueTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.items = []
        for b in range(4):
            d = self.root / f'b{b}'
            d.mkdir()
            (d / 'input.jsonl').write_text(''.join(json.dumps(row(b * 10 + i)) + '\n' for i in range(3)))
            (d / 'worker.json').write_text(json.dumps({'status': 'queued'}))
            self.items.append({'root': d.name, 'rows': 3, 'input_sha256': digest(d / 'input.jsonl')})
        self.plan = self.root / 'plan.json'
        self.plan.write_text(json.dumps({'version': 1, 'batches': self.items}))
        self.outcomes = {}  # batch name -> 'ok' | 'timeout' | 'timeout_first' | 'closed'
        self.launched = []

    def fake_launch(self, command, **kw):
        root = Path(command[command.index('--manifest') + 1]).parent
        self.launched.append(root.name)
        outcome = self.outcomes.get(root.name, 'ok')
        rows = [json.loads(l) for l in (root / 'input.jsonl').read_text().splitlines()]
        out = root / 'asr'
        out.mkdir()
        done = rows if outcome == 'ok' else ([] if outcome in ('timeout_first',) else rows[:1])
        for r in done:
            r.update(sentence='x', num_words=1, asr_status='transcribed')
        (out / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in done))
        failed = [] if outcome == 'ok' else [{'id': rows[len(done)]['id'], 'error_kind': 'RuntimeError'}]
        (out / 'summary.json').write_text(json.dumps({'selected': 3, 'completed': len(done), 'failed': failed,
                                                       'stopped_early': outcome != 'ok'}))
        if outcome != 'ok':
            if outcome == 'closed':
                events = []
            else:
                events = [{'elapsed_seconds': 3.5, 'event': {'type': 'error', 'code': 'finalize_timeout', 'fatal': False}}]
            (out / 'x.attempt-1.json').write_text(json.dumps({'status': 'failed', 'events': events}))
        return SimpleNamespace(pid=1, wait=lambda: 0 if outcome == 'ok' else 1)

    def run_queue(self, **kw):
        return qq.run(self.plan, self.root / 'state', self.root / 'key', 'hil', 'm', min_free_gib=0,
                      launch=self.fake_launch, sleep=lambda s: None, **kw)

    def test_timeout_is_quarantined_and_queue_continues(self):
        self.outcomes = {'b1': 'timeout'}
        state = self.run_queue()
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(self.launched, ['b0', 'b1', 'b2', 'b3'])
        self.assertEqual((state['completed_batches'], state['quarantined']), (3, 1))
        q = [json.loads(l) for l in (self.root / 'state/quarantine.jsonl').read_text().splitlines()]
        self.assertEqual([r['id'] for r in q], ['r11'])
        left = [json.loads(l)['id'] for l in (self.root / 'state/unattempted.jsonl').read_text().splitlines()]
        self.assertEqual(left, ['r12'])  # never the quarantined row, never completed rows

    def test_other_failure_stops_queue(self):
        self.outcomes = {'b1': 'closed'}
        with self.assertRaises(RuntimeError):
            self.run_queue()
        self.assertEqual(self.launched, ['b0', 'b1'])
        state = json.loads((self.root / 'state/worker.json').read_text())
        self.assertEqual(state['status'], 'stopped_after_error')

    def test_consecutive_first_row_timeouts_stop_queue(self):
        self.outcomes = {'b0': 'timeout_first', 'b1': 'timeout_first'}
        with self.assertRaises(RuntimeError):
            self.run_queue(max_consecutive=2)
        self.assertEqual(self.launched, ['b0', 'b1'])

    def test_quarantine_budget(self):
        self.outcomes = {'b0': 'timeout', 'b1': 'timeout'}
        with self.assertRaises(RuntimeError):
            self.run_queue(max_quarantined=2)
        self.assertEqual(self.launched, ['b0', 'b1'])

    def test_result_event_is_not_quarantinable(self):
        self.outcomes = {'b0': 'timeout'}
        orig = self.fake_launch

        def launch(command, **kw):
            p = orig(command, **kw)
            f = self.root / 'b0/asr/x.attempt-1.json'
            a = json.loads(f.read_text())
            a['events'].append({'event': {'type': 'result'}})
            f.write_text(json.dumps(a))
            return p
        with self.assertRaises(RuntimeError):
            qq.run(self.plan, self.root / 'state', self.root / 'key', 'hil', 'm', min_free_gib=0, launch=launch,
                   sleep=lambda s: None)

    def test_finished_batches_are_skipped_on_recovery(self):
        self.outcomes = {'b1': 'timeout'}
        self.run_queue()
        self.launched.clear()
        (self.root / 'state/worker.json').unlink()
        state = self.run_queue()
        self.assertEqual(self.launched, [])
        self.assertEqual(state['skipped_finished'], 4)


if __name__ == '__main__':
    unittest.main()
