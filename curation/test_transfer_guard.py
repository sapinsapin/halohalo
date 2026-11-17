import io
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import transfer_guard as guard


class TransferTests(unittest.TestCase):
    def run_fake(self, *, end=20, exitcode=0, change=None, free=None, **bounds):
        clock = SimpleNamespace(now=0)
        process = SimpleNamespace(pid=12345)
        process.poll = lambda: exitcode if clock.now >= end else None
        process.wait = Mock(return_value=exitcode)
        def sleep(seconds):
            clock.now += seconds
            if change:
                change(clock.now)
        patches = (
            patch.object(guard.time, 'monotonic', side_effect=lambda: clock.now),
            patch.object(guard.time, 'sleep', side_effect=sleep),
            patch.object(guard.subprocess, 'Popen', return_value=process),
            patch.object(guard.os, 'killpg'),
            patch.object(guard.shutil, 'disk_usage', side_effect=lambda _: SimpleNamespace(free=free(clock.now) if free else 100*1024**3)),
        )
        with patches[0], patches[1], patches[2] as popen, patches[3] as kill, patches[4]:
            try:
                result = guard.run_transfer(['downloader', 'secret'], self.root, 'item', io.StringIO(), **bounds)
            except guard.TransferFailure as exc:
                result = exc.code
            return result, clock.now, popen, kill

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_progress_past_old_hour_limit(self):
        result, elapsed, popen, kill = self.run_fake(end=4000, poll_seconds=100,
            change=lambda t: (self.root/'item.webm.part').write_bytes(b'x'*int(t)))
        self.assertEqual(result, 0)
        self.assertEqual(elapsed, 4000)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertFalse(popen.call_args.kwargs['shell'])
        self.assertEqual(popen.call_args.kwargs['stderr'], subprocess.STDOUT)
        kill.assert_not_called()

    def test_exit_preserved(self):
        self.assertEqual(self.run_fake(exitcode=7)[0], 7)

    def test_idle_terminates_and_keeps_partial(self):
        partial = self.root/'item.m4a.part'
        partial.write_bytes(b'partial')
        result, _, _, kill = self.run_fake(end=100, idle_timeout=10)
        self.assertEqual(result, 'idle_timeout')
        self.assertEqual(partial.read_bytes(), b'partial')
        self.assertEqual([c.args for c in kill.call_args_list],
                         [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])

    def test_hard_limit_even_with_growth(self):
        result, elapsed, _, _ = self.run_fake(end=100, max_seconds=12,
            change=lambda t: (self.root/'item.m4a.part').write_bytes(b'x'*int(t)))
        self.assertEqual(result, 'hard_timeout')
        self.assertEqual(elapsed, 14)  # Includes bounded TERM grace.

    def test_disk_floor_before_start_and_during_transfer(self):
        result, _, popen, _ = self.run_fake(free=lambda t: 0)
        self.assertEqual(result, 'disk_floor')
        popen.assert_not_called()
        self.assertEqual(self.run_fake(free=lambda t: (100 if t < 5 else 0)*1024**3)[0], 'disk_floor')

    def test_preexisting_overbudget_does_not_launch(self):
        (self.root/'item.webm.part').write_bytes(b'x'*11)
        result, _, popen, kill = self.run_fake(max_transfer_bytes=10)
        self.assertEqual(result, 'transfer_budget')
        popen.assert_not_called()
        kill.assert_not_called()

    def test_budget_crossing_stops_reaps_and_preserves_partial(self):
        partial = self.root/'item.webm.part'
        result, _, popen, kill = self.run_fake(end=100, max_transfer_bytes=10,
            change=lambda t: partial.write_bytes(b'x'*11))
        self.assertEqual(result, 'transfer_budget')
        popen.return_value.wait.assert_called_once_with(timeout=5)
        self.assertEqual([c.args for c in kill.call_args_list],
                         [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])
        self.assertEqual(partial.read_bytes(), b'x'*11)

    def test_oversized_successful_exit_is_rejected(self):
        partial = self.root/'item.webm.part'
        result, _, popen, kill = self.run_fake(end=5, max_transfer_bytes=10,
            change=lambda t: partial.write_bytes(b'x'*11))
        self.assertEqual(result, 'transfer_budget')
        popen.return_value.wait.assert_called_once_with()
        kill.assert_not_called()
        self.assertEqual(partial.read_bytes(), b'x'*11)

    def test_default_has_no_transfer_budget(self):
        partial = self.root/'item.webm.part'
        result, _, _, kill = self.run_fake(end=5,
            change=lambda t: partial.write_bytes(b'x'*11))
        self.assertEqual(result, 0)
        kill.assert_not_called()

    def test_budget_sums_distinct_media_inodes(self):
        def change(t):
            (self.root/'item.audio.webm').write_bytes(b'x'*6)
            if t >= 10:
                (self.root/'item.audio.flac').write_bytes(b'y'*5)
        result, _, _, _ = self.run_fake(end=100, change=change, max_transfer_bytes=10)
        self.assertEqual(result, 'transfer_budget')

    def test_rename_does_not_double_count_budget(self):
        partial = self.root/'item.webm.part'
        partial.write_bytes(b'x'*6)
        def change(t):
            if partial.exists():
                partial.rename(self.root/'item.webm')
        result, _, _, _ = self.run_fake(end=5, change=change, max_transfer_bytes=6)
        self.assertEqual(result, 0)

    def test_invalid_transfer_budgets(self):
        for budget in (True, False, 0, -1, 1.5, '1'):
            with self.subTest(budget=budget):
                result, _, popen, _ = self.run_fake(max_transfer_bytes=budget)
                self.assertEqual(result, 'invalid_bounds')
                popen.assert_not_called()

    def test_unrelated_metadata_and_mtime_are_not_progress(self):
        partial = self.root/'item.m4a.part'
        partial.write_bytes(b'x')
        def change(t):
            for name in ('other.m4a.part', 'itemExtra.m4a.part',
                         'item.download.log', 'item.info.json', 'item.m4a.ytdl'):
                (self.root/name).write_bytes(b'x'*int(t))
            partial.touch()
        self.assertEqual(self.run_fake(end=100, idle_timeout=10, change=change)[0], 'idle_timeout')

    def test_deletion_and_rename_are_not_progress(self):
        a = self.root/'item.webm.part'
        a.write_bytes(b'x'*100)
        def change(t):
            if t == 5:
                a.rename(self.root/'item.webm')
            elif t == 10:
                (self.root/'item.webm').unlink()
        self.assertEqual(self.run_fake(end=100, idle_timeout=15, change=change)[0], 'idle_timeout')

    def test_stop_kills_before_reaping_and_bounds_wait(self):
        events = []
        process = SimpleNamespace(pid=99, wait=lambda timeout: events.append(('wait', timeout)))
        with patch.object(guard.os, 'killpg', side_effect=lambda p, s: events.append(('signal', s))), patch.object(guard.time, 'sleep'):
            guard._stop(process)
        self.assertEqual(events, [('signal', signal.SIGTERM), ('signal', signal.SIGKILL), ('wait', 5)])

    def test_permission_error_only_accepted_for_terminated_child(self):
        for running in (False, True):
            def wait(timeout):
                if running:
                    raise subprocess.TimeoutExpired('owned-child', timeout)
                return -15
            process = SimpleNamespace(pid=99, wait=wait)
            with patch.object(guard.os, 'killpg', side_effect=[None, PermissionError()]), patch.object(guard.time, 'sleep'):
                if running:
                    with self.assertRaises(guard.TransferFailure) as caught:
                        guard._stop(process)
                    self.assertEqual(caught.exception.code, 'cleanup_timeout')
                else:
                    guard._stop(process)

    def test_safe_error_does_not_include_command(self):
        with patch.object(guard.subprocess, 'Popen', side_effect=OSError('secret token')):
            with self.assertRaises(guard.TransferFailure) as caught:
                guard.run_transfer(['secret'], self.root, 'item', io.StringIO(), min_free_gib=0)
        self.assertEqual(str(caught.exception), 'transfer_monitor_error')


if __name__ == '__main__':
    unittest.main()
