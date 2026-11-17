import json
import fcntl
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import asr_queue as queue
from .schema import VERSION, digest


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.plan=self.root/'plan.json';self.state=self.root/'queue'
        self.row=dict(schema_version=VERSION,id='a',source_id='source',recording_id='record',audio_path='/unused.wav',
                      audio_sha256='0'*64,sentence='',prompt=None,text_is_prompt=False,language='fil',languages=['fil'],
                      speech_type='unknown',speaker_id='',duration=1,num_words=0,split='unassigned',review_status='draft',
                      rights={k:'unknown' for k in ('training','commercial','redistribution')},provenance={},asr_status='pending')
        self.items=[]
        for name in ['a','b']:
            root=self.root/name;root.mkdir()
            (root/'input.jsonl').write_text(json.dumps({**self.row,'id':name})+'\n')
            (root/'worker.json').write_text(json.dumps({'status':'queued'}))
            self.items.append(dict(root=name,rows=1,input_sha256=digest(root/'input.jsonl')))
        self.save_plan()

    def save_plan(self, **extra):
        self.plan.write_text(json.dumps(dict(version=1,batches=self.items,**extra)))

    def run_queue(self):
        return queue.run(self.plan,self.state,self.root/'key','fil','model',min_free_gib=0)

    def child(self, command, **kwargs):
        root=Path(command[command.index('--manifest')+1]).parent
        self.assertEqual(json.loads((root/'worker.json').read_text())['status'],'launching')
        def wait():
            out=root/'asr';out.mkdir()
            row=json.loads((root/'input.jsonl').read_text());row.update(sentence='text',num_words=1,asr_status='transcribed')
            (out/'manifest.jsonl').write_text(json.dumps(row)+'\n')
            (out/'summary.json').write_text(json.dumps(dict(selected=1,completed=1,failed=[],stopped_early=False)))
            return 0
        return SimpleNamespace(pid=123,wait=wait)

    def test_sequential_completion_and_no_replay(self):
        with patch.object(queue.subprocess,'Popen',side_effect=self.child) as spawn:
            self.run_queue()
            self.assertEqual(spawn.call_count,2)
        self.assertEqual(json.loads((self.state/'worker.json').read_text())['status'],'complete')
        with patch.object(queue.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(ValueError,'already started'):self.run_queue()
            spawn.assert_not_called()

    def test_failed_stage_does_not_launch_next(self):
        with patch.object(queue.subprocess,'Popen',return_value=SimpleNamespace(pid=123,wait=lambda:1)) as spawn:
            with self.assertRaises((ValueError,FileNotFoundError)):self.run_queue()
            self.assertEqual(spawn.call_count,1)
        self.assertEqual(json.loads((self.root/'b/worker.json').read_text())['status'],'queued')
        self.assertEqual(json.loads((self.state/'worker.json').read_text())['status'],'stopped_after_error')

    def test_changed_input_or_duplicate_ids_prevent_launch(self):
        (self.root/'a/input.jsonl').write_text(json.dumps({**self.row,'sentence':'changed','num_words':1})+'\n')
        with self.assertRaisesRegex(ValueError,'changed'):queue.prepare_plan(self.plan)
        self.items[0]['input_sha256']=digest(self.root/'a/input.jsonl')
        (self.root/'b/input.jsonl').write_bytes((self.root/'a/input.jsonl').read_bytes())
        self.items[1]['input_sha256']=digest(self.root/'b/input.jsonl');self.save_plan()
        with self.assertRaisesRegex(ValueError,'duplicate'):queue.prepare_plan(self.plan)

    def test_absent_dependency_is_not_treated_as_complete(self):
        dep=self.root/'dependency';dep.mkdir();(dep/'worker.json').write_text(json.dumps(dict(status='running',pid=123)))
        self.save_plan(dependency=dict(root='dependency',rows=1))
        with patch.object(queue.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='')),patch.object(queue.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(ValueError,'process absent'):self.run_queue()
            spawn.assert_not_called()

    def test_incomplete_or_wrong_output_blocks_next(self):
        def corrupt(command,**kwargs):
            process=self.child(command,**kwargs)
            original=process.wait
            def wait():
                code=original();root=Path(command[command.index('--manifest')+1]).parent
                row={**self.row,'id':'wrong','asr_status':'transcribed'}
                (root/'asr/manifest.jsonl').write_text(json.dumps(row)+'\n');return code
            process.wait=wait;return process
        with patch.object(queue.subprocess,'Popen',side_effect=corrupt) as spawn:
            with self.assertRaisesRegex(ValueError,'identities'):self.run_queue()
            self.assertEqual(spawn.call_count,1)

    def test_storage_floor_and_uncertain_launch_fail_closed(self):
        with patch.object(queue.shutil,'disk_usage',return_value=SimpleNamespace(free=-1)),patch.object(queue.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(ValueError,'storage floor'):self.run_queue()
            spawn.assert_not_called()
        with patch.object(queue.subprocess,'Popen',side_effect=OSError('spawn interrupted')):
            with self.assertRaisesRegex(OSError,'spawn'):self.run_queue()
        self.assertEqual(json.loads((self.root/'a/worker.json').read_text())['status'],'launching')
        with self.assertRaisesRegex(ValueError,'already started'):queue.prepare_plan(self.plan)

    def test_competing_batch_claim_prevents_spawn(self):
        with (self.root/'a/queue-launch.lock').open('a') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(queue.subprocess,'Popen') as spawn:
                with self.assertRaises(BlockingIOError):self.run_queue()
                spawn.assert_not_called()
        self.assertEqual(json.loads((self.root/'a/worker.json').read_text())['status'],'queued')

    def test_plan_mutation_during_stage_blocks_next(self):
        def mutate(command,**kwargs):
            process=self.child(command,**kwargs);original=process.wait
            def wait():
                code=original();self.plan.write_text(self.plan.read_text()+' ');return code
            process.wait=wait;return process
        with patch.object(queue.subprocess,'Popen',side_effect=mutate) as spawn:
            with self.assertRaisesRegex(ValueError,'changed during stage'):self.run_queue()
            self.assertEqual(spawn.call_count,1)
        self.assertEqual(json.loads((self.root/'b/worker.json').read_text())['status'],'queued')

    def test_queue_state_cannot_overwrite_batch_worker(self):
        with self.assertRaisesRegex(ValueError,'separate'):
            queue.run(self.plan,self.root/'a',self.root/'key','fil','model',min_free_gib=0)
        self.assertEqual(json.loads((self.root/'a/worker.json').read_text())['status'],'queued')


if __name__=='__main__':unittest.main()
