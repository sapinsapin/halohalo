import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from curation.collect import connect
from curation.supervise import Supervisor, Profile, GIB, lock, inventory


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'rss'
        db = connect(self.root)
        db.execute("INSERT INTO items(id,source_id,status,attempts) VALUES('a','show','pending',0)")
        db.commit()
        db.close()
        self.profile = Profile(self.root, 'test', 300, 50)
        self.supervisor = Supervisor(Path(self.temp.name)/'state', profiles=(self.profile,), apply=True)
        self.disk = patch('curation.supervise.shutil.disk_usage', return_value=Mock(free=50*GIB)).start()
        self.ps = patch('curation.supervise.processes', return_value=[]).start()
        self.spawn = patch('curation.supervise.subprocess.Popen', return_value=Mock(pid=123)).start()
        self.backup = patch('curation.supervise.hub.backup', side_effect=self.checkpoint).start()
        self.evict = patch('curation.supervise.evict.evict', return_value={'evicted_files': 1}).start()
        self.addCleanup(patch.stopall)

    def checkpoint(self, *args, **kwargs):
        (self.root/'hub-checkpoint.json').write_text(json.dumps({'revision': 'verified'}))

    def downloaded(self):
        with connect(self.root) as db:
            db.execute("UPDATE items SET status='downloaded',duration=3600 WHERE id='a'")

    def entry(self):
        return self.supervisor.tick()['roots'][str(self.root)]

    def test_observe_default_no_mutations(self):
        self.supervisor.apply = False
        self.entry()
        self.spawn.assert_not_called()
        self.backup.assert_not_called()
        self.evict.assert_not_called()

    def test_active_worker_can_checkpoint_never_evict_or_resume(self):
        self.downloaded()
        self.disk.return_value.free = 30*GIB
        with lock(self.root/'download.lock'):
            entry = self.entry()
        self.backup.assert_called_once()
        self.evict.assert_not_called()
        self.spawn.assert_not_called()
        self.assertTrue(entry['downloader_active'])

    def test_threshold_and_bounded_eviction(self):
        self.downloaded()
        self.entry()
        self.backup.assert_not_called()
        self.disk.return_value.free = 30*GIB
        # A new heartbeat after the persisted cooldown.
        with patch('curation.supervise.time.time', return_value=10**12):
            entry = self.entry()
        self.backup.assert_called_once()
        self.assertEqual(self.evict.call_args.kwargs['max_files'], 20)
        self.assertEqual(entry['checkpoint']['revision'], 'verified')

    def test_crash_restart_launch_guard_prevents_duplicate(self):
        # Simulate an inherited launch FD whose parent has crashed, before the
        # child has acquired download.lock or recorded any PID receipt.
        with lock(self.root/'supervisor-launch.lock'):
            self.entry()
        self.spawn.assert_not_called()

    def test_launch_records_pid_and_inherits_guard_and_throttles(self):
        entry = self.entry()
        self.assertEqual(entry['launched_worker']['pid'], 123)
        self.assertTrue(self.spawn.call_args.kwargs['pass_fds'])
        self.assertIn('--max-source-hours', entry['launched_worker']['command'])
        self.entry()
        self.spawn.assert_called_once()

    def test_resume_retries_stop_without_progress(self):
        for n in range(7):
            with patch('curation.supervise.time.time', return_value=10**10+n*1000):
                entry = self.entry()
        self.assertEqual(self.spawn.call_count, 5)
        self.assertIn('retry limit', entry['decision'])

    def test_caps_include_remote_hours(self):
        with connect(self.root) as db:
            db.execute("INSERT INTO items(id,source_id,status,duration) VALUES('b','show','remote',180000)")
        self.assertEqual(inventory(self.profile)['eligible'], 0)
        self.entry()
        self.spawn.assert_not_called()
        self.assertEqual(inventory(Profile(self.root, 'test', 0, 0))['eligible'], 1)
        with connect(self.root) as db:
            db.execute("UPDATE items SET duration=1080000 WHERE id='b'")
        self.assertEqual(inventory(Profile(self.root, 'test', 300, 0))['eligible'], 0)

    def test_failure_retains_audio_and_error_and_backoff(self):
        self.downloaded()
        media = self.root/'example.media'
        media.write_bytes(b'keep')
        self.disk.return_value.free = 30*GIB
        self.backup.side_effect = RuntimeError('upload failed')
        entry = self.entry()
        self.assertEqual(entry['error']['message'], 'upload failed')
        self.evict.assert_not_called()
        self.spawn.assert_not_called()
        self.assertEqual(media.read_bytes(), b'keep')
        self.entry()
        self.backup.assert_called_once()

    def test_global_lock_serializes(self):
        self.supervisor.state_dir.mkdir()
        with lock(self.supervisor.state_dir/'supervisor.lock'):
            self.assertEqual(self.supervisor.tick(), {'busy': True})
        self.spawn.assert_not_called()

    def test_exhausted_attempts_no_restart(self):
        with connect(self.root) as db:
            db.execute('UPDATE items SET attempts=3')
        self.assertEqual(self.entry()['decision'], 'no eligible work')
        self.spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
