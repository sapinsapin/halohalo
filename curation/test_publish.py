import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq

from .publish import publish
from .schema import VERSION, digest


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.export = self.root/'export'; self.export.mkdir()
        self.receipt = self.root/'receipt.json'
        row = dict(schema_version=VERSION,id='one',source_id='source',recording_id='record',
                   audio_path='/unused/source.wav',audio_sha256='0'*64,sentence='',prompt=None,
                   text_is_prompt=False,language='hil',languages=['hil'],speech_type='unknown',speaker_id='',
                   duration=1.,num_words=0,split='unassigned',review_status='draft',
                   rights={k:'unknown' for k in ('training','commercial','redistribution')},provenance={})
        (self.export/'manifest.jsonl').write_text(json.dumps(row)+'\n')
        (self.export/'README.md').write_text('Export')
        self.row = {k:v for k,v in row.items() if k != 'provenance'}
        pq.write_table(pa.Table.from_pylist([self.row]), self.export/'curation-unassigned-00000-of-00001.parquet')
        self.artifact = self.export/'curation-unassigned-00000-of-00001.parquet'
        self.summary = dict(schema_version=VERSION,profile='curation',rows=1,
                            artifacts=[dict(path=self.artifact.name,sha256=digest(self.artifact),rows=1)])
        self.write_summary()
        (self.export/'secret.env').write_text('not uploaded')
        self.api = Mock()
        self.api.dataset_info.return_value = SimpleNamespace(private=True,sha='parent-sha')
        self.api.list_repo_files.return_value = ['README.md','other/data.parquet']
        self.api.create_commit.return_value = SimpleNamespace(oid='immutable-sha')
        self.remote = self.root/'remote'; self.remote.mkdir()
        def commit(**kwargs):
            for op in kwargs['operations']:
                (self.remote/Path(op.path_in_repo).name).write_bytes(Path(op.path_or_fileobj).read_bytes())
            return SimpleNamespace(oid='immutable-sha')
        self.api.create_commit.side_effect = commit
        self.download = Mock(side_effect=lambda **kwargs:str(self.remote/Path(kwargs['filename']).name))

    def call(self, **kwargs):
        return publish(self.export, 'exports/pilot', self.receipt, api=self.api, download=self.download, **kwargs)

    def write_summary(self):
        (self.export/'summary.json').write_text(json.dumps(self.summary))

    def test_allowlist_cas_and_immutable_verification(self):
        result = self.call()
        self.assertTrue(result['verified'])
        self.assertEqual(len(result['files']), 4)
        self.assertNotIn('secret.env', result['files'])
        self.assertEqual(self.api.create_commit.call_args.kwargs['parent_commit'], 'parent-sha')
        for call in self.download.call_args_list:
            self.assertEqual(call.kwargs['revision'], 'immutable-sha')
            self.assertTrue(call.kwargs['force_download'])
        self.assertEqual(json.loads(self.receipt.read_text()), result)

    def test_recover_existing_and_reject_changed_bytes(self):
        self.download.side_effect = RuntimeError('interrupted verification')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.call()
        self.assertFalse(self.receipt.exists())
        self.api.create_commit.reset_mock()
        self.api.dataset_info.return_value.sha = 'existing-commit'
        self.api.list_repo_files.return_value = ['exports/pilot/'+p.name for p in self.remote.iterdir()]
        self.download.side_effect = lambda **kwargs:str(self.remote/Path(kwargs['filename']).name)
        result = self.call(verify_existing=True)
        self.assertTrue(result['verification_only'])
        self.assertEqual(result['revision'], 'existing-commit')
        self.api.create_commit.assert_not_called()
        self.receipt.unlink()
        (self.remote/'README.md').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'remote artifact'):
            self.call(verify_existing=True)
        self.assertFalse(self.receipt.exists())
        self.api.list_repo_files.return_value.append('exports/pilot/unexpected.txt')
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            self.call(verify_existing=True)
        self.api.create_commit.assert_not_called()

    def test_core_scalar_mismatch_rejected_even_with_updated_hash(self):
        changes = dict(sentence='changed transcript',duration=2.,split='train',review_status='text_reviewed',
                       source_id='other',recording_id='other',text_is_prompt=True,language='en',num_words=99)
        for key, value in changes.items():
            with self.subTest(field=key):
                pq.write_table(pa.Table.from_pylist([{**self.row, key:value}]), self.artifact)
                self.summary['artifacts'][0]['sha256'] = digest(self.artifact)
                self.write_summary()
                with self.assertRaisesRegex(ValueError, 'core scalar'):
                    self.call()
        self.api.create_commit.assert_not_called()

    def test_remote_corruption_has_no_receipt(self):
        corrupt = self.root/'corrupt'; corrupt.write_text('bad')
        self.download.side_effect = lambda **kwargs:str(corrupt)
        with self.assertRaisesRegex(ValueError,'remote artifact'):
            self.call()
        self.assertFalse(self.receipt.exists())

    def test_private_and_conflicting_prefix(self):
        self.api.dataset_info.return_value.private = False
        with self.assertRaisesRegex(ValueError,'private'):
            self.call()
        self.api.dataset_info.return_value.private = True
        self.api.list_repo_files.return_value = ['exports/pilot/old.parquet']
        with self.assertRaisesRegex(ValueError,'nonempty'):
            self.call()
        self.api.create_commit.assert_not_called()

    def test_race_not_retried(self):
        self.api.create_commit.side_effect = RuntimeError('parent conflict')
        with self.assertRaisesRegex(RuntimeError,'parent conflict'):
            self.call()
        self.assertEqual(self.api.create_commit.call_count,1)
        self.assertFalse(self.receipt.exists())

    def test_tamper_and_rows(self):
        self.summary['artifacts'][0]['sha256'] = '0'*64; self.write_summary()
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.call()
        self.summary['artifacts'][0]['sha256'] = digest(self.artifact)
        self.summary['artifacts'][0]['rows'] = 2; self.write_summary()
        with self.assertRaisesRegex(ValueError,'row count'):
            self.call()
        self.api.create_commit.assert_not_called()

    def test_traversal_symlink_bounds_and_schema(self):
        with self.assertRaisesRegex(ValueError,'unsafe'):
            publish(self.export,'../bad',self.receipt,api=self.api)
        self.summary['artifacts'][0]['path'] = '../secret.env'; self.write_summary()
        with self.assertRaisesRegex(ValueError,'unsafe'):
            self.call()
        self.summary['artifacts'][0]['path'] = self.artifact.name; self.write_summary()
        with self.assertRaisesRegex(ValueError,'bound'):
            self.call(max_files=3)
        with self.assertRaisesRegex(ValueError,'bound'):
            self.call(max_bytes=1)
        self.artifact.unlink(); self.artifact.symlink_to(self.export/'secret.env')
        with self.assertRaisesRegex(ValueError,'symlink'):
            self.call()
        (self.export/'manifest.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'missing fields'):
            self.call()
        with self.assertRaisesRegex(ValueError,'outside'):
            publish(self.export,'safe',self.export/'receipt.json',api=self.api)
        self.api.create_commit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
