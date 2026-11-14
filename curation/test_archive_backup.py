import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from curation import archive_backup as b


class ArchiveBackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.audio = self.root / 'audio.webm'; self.audio.write_bytes(b'audio')
        self.sha = b.digest(self.audio)
        self.record = dict(audio=self.audio, stat=self.audio.stat(), metadata=b'{"training_ready":false}\n', sha256=self.sha, bytes=5)
        self.objects = {}; self.version = 0; self.fail = False; self.conflict = False; self.downloads = 0
        self.api = NS(dataset_info=lambda repo: NS(private=True, sha=str(self.version)),
                      get_paths_info=self.paths, create_commit=self.commit)
    def paths(self, repo, paths, **kwargs):
        return [self.objects[p] for p in paths if p in self.objects]
    def commit(self, **kwargs):
        if self.conflict:
            raise RuntimeError('parent commit conflict')
        if self.fail and self.version == 1:
            raise RuntimeError('network interrupted')
        self.assertEqual(kwargs['parent_commit'], str(self.version))
        op = kwargs['operations'][0]; value = op.path_or_fileobj
        blob = value.read_bytes() if isinstance(value, Path) else value
        self.objects[op.path_in_repo] = NS(path=op.path_in_repo, size=len(blob), lfs={'sha256': hashlib.sha256(blob).hexdigest()})
        self.version += 1
        return NS(oid=str(self.version))
    def run_backup(self):
        with patch.object(b, 'prepare', return_value=[self.record]):
            return b.backup('unused', self.root / 'state', api=self.api, operation=lambda **kw: NS(**kw), download=self.download)
    def download(self, **kwargs):
        self.downloads += 1
        raise AssertionError('trusted LFS must avoid download')
    def test_success_and_idempotent_resume(self):
        first = self.run_backup(); self.assertEqual(self.version, 2)
        self.assertEqual(first, self.run_backup()); self.assertEqual(self.version, 2)
        receipt = json.loads(Path(first[0]).read_text())
        self.assertFalse(receipt['local_deletion_authorized']); self.assertEqual(receipt['revision'], '2')
        self.assertTrue(self.audio.exists()); self.assertEqual(self.downloads, 0)
    def test_network_error_retains_progress_and_resumes(self):
        self.fail = True
        with self.assertRaisesRegex(RuntimeError, 'network'):
            self.run_backup()
        self.assertEqual(len(list((self.root / 'state/progress').glob('*.json'))), 1)
        self.assertFalse((self.root / 'state/receipts').exists())
        self.fail = False; self.run_backup(); self.assertEqual(self.version, 2)
    def test_parent_race_does_not_overwrite(self):
        self.conflict = True
        with self.assertRaisesRegex(RuntimeError, 'parent commit'):
            self.run_backup()
        self.assertEqual(self.objects, {})
    def test_conflicting_remote_is_rejected(self):
        path = f'raw/archive/sha256/{self.sha}/audio'
        self.objects[path] = NS(size=5, lfs={'sha256': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'remote hash conflict'):
            self.run_backup()
        self.assertEqual(self.version, 0)
    def test_private_required(self):
        self.api.dataset_info = lambda repo: NS(private=False, sha='0')
        with self.assertRaisesRegex(ValueError, 'private'):
            self.run_backup()
    def test_local_change_before_upload(self):
        self.audio.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed before upload'):
            self.run_backup()
    def test_non_lfs_fallback_is_pinned(self):
        entry = dict(path='raw/object', bytes=5, sha256=self.sha)
        self.objects['raw/object'] = NS(size=5, lfs=None)
        revisions = []
        def download(**kw):
            revisions.append(kw['revision']); return self.audio
        self.assertTrue(b.verify(self.api, 'private/repo', 'immutable', entry, download))
        self.assertEqual(revisions, ['immutable'])
    def test_checkpoint_conflict(self):
        path = self.root / 'receipt.json'; b.durable(path, b'a')
        with self.assertRaisesRegex(ValueError, 'conflicting local'):
            b.durable(path, b'b')
        self.assertEqual(path.read_bytes(), b'a')
    def prepare_fixture(self):
        source = self.root / 'source.json'
        source.write_text(json.dumps(dict(title='Full episode', token='secret', audio_path='/private/local')))
        row = dict(provenance=dict(import_kind='archive', acquisition_status='downloaded',
                   acquisition_metadata_path=str(source), acquisition_metadata_sha256=b.digest(source),
                   source_url='https://example.com/watch?v=id&token=secret'),
                   audio_path=str(self.audio), audio_bytes=5, audio_sha256=self.sha, duration=10,
                   schema_version='sapinsapin.speech.v1', id='id', source_id='source', recording_id='recording',
                   language='hil', languages=['hil'], speech_type='unknown', review_status='draft',
                   rights=dict(training='unknown', commercial='unknown', redistribution='unknown'))
        return row
    def test_prepare_validates_and_limits_metadata(self):
        row = self.prepare_fixture()
        probe = NS(stdout=json.dumps(dict(format=dict(duration=10), streams=[dict(codec_type='audio')])))
        with patch.object(b, 'load_manifest', return_value=[row]), patch.object(b.subprocess, 'run', return_value=probe):
            records = b.prepare('manifest')
        meta = json.loads(records[0]['metadata'])
        self.assertEqual(meta['source']['title'], 'Full episode')
        self.assertNotIn('token', meta['source']); self.assertNotIn('audio_path', meta['source'])
        self.assertEqual(meta['provenance']['source_url'], 'https://example.com/watch?v=id')
    def test_prepare_rejects_tamper_and_bounds(self):
        row = self.prepare_fixture()
        with patch.object(b, 'load_manifest', return_value=[row]):
            with self.assertRaisesRegex(ValueError, 'within bounds'):
                b.prepare('manifest', max_bytes=1)
            self.audio.write_bytes(b'tamper')
            with self.assertRaisesRegex(ValueError, 'hash/size'):
                b.prepare('manifest')
    def test_prepare_rejects_bad_duration(self):
        row = self.prepare_fixture()
        probe = NS(stdout=json.dumps(dict(format=dict(duration=1), streams=[dict(codec_type='audio')])))
        with patch.object(b, 'load_manifest', return_value=[row]), patch.object(b.subprocess, 'run', return_value=probe):
            with self.assertRaisesRegex(ValueError, 'duration'):
                b.prepare('manifest')
    def test_url_credentials_removed(self):
        self.assertEqual(b.safe_url('https://user:password@example.com/watch?v=123&token=secret#x'), 'https://example.com/watch?v=123')


if __name__ == '__main__':
    unittest.main()
