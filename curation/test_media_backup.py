import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from curation import media_backup as backup


class MediaBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'station' / 'interval'
        self.directory.mkdir(parents=True)
        self.audio = self.directory / 'capture.mka'
        self.audio.write_bytes(b'fixture audio')
        self.status = dict(state='acquired', requested_seconds=1800, bytes=self.audio.stat().st_size,
                           audio_sha256=hashlib.sha256(self.audio.read_bytes()).hexdigest(),
                           runner_pid=123, secret='exclude', rights={'training': 'cleared'})
        self.save()

    def save(self):
        (self.directory / 'status.json').write_text(json.dumps(self.status))

    def probe(self, duration=1800):
        return patch.object(backup.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(
            dict(format={'duration': duration}, streams=[dict(codec_type='audio', codec_name='aac')]))))

    def test_finalized_metadata_is_restricted_and_rights_unknown(self):
        with self.probe():
            result = backup.prepare(self.root)
        metadata = json.loads(result[0][2])
        self.assertNotIn('secret', metadata)
        self.assertNotIn('runner_pid', metadata)
        self.assertFalse(metadata['training_ready'])
        self.assertEqual(metadata['rights']['training'], 'unknown')

    def test_partial_is_excluded(self):
        self.status['state'] = 'partial'; self.save()
        with self.assertRaisesRegex(ValueError, 'no finalized'):
            backup.prepare(self.root)

    def test_tampered_audio_rejected(self):
        self.audio.write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError, 'hash/size'):
            backup.prepare(self.root)

    def test_short_recording_rejected(self):
        with self.probe(10), self.assertRaisesRegex(ValueError, 'incomplete'):
            backup.prepare(self.root)

    def test_byte_bound(self):
        with self.assertRaisesRegex(ValueError, 'byte bound'):
            backup.prepare(self.root, max_bytes=1)

    def test_symlink_audio_rejected(self):
        self.audio.unlink(); self.audio.symlink_to(self.directory / 'status.json')
        with self.assertRaisesRegex(ValueError, 'unsafe audio'):
            backup.prepare(self.root)

    def test_remote_hash_mismatch(self):
        api = SimpleNamespace(get_paths_info=lambda *a, **kw: [SimpleNamespace(path='raw/audio', size=13)])
        with patch('huggingface_hub.hf_hub_download', return_value=str(self.audio)):
            with self.assertRaisesRegex(ValueError, 'remote hash mismatch'):
                backup.verify(api, 'private/repo', 'immutable', [dict(path='raw/audio', bytes=13, sha256='wrong')])


if __name__ == '__main__':
    unittest.main()
