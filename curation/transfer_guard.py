"""Bound external transfers by real media progress, disk space, and wall time."""
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import time


class TransferFailure(RuntimeError):
    """A safe, machine-readable failure; never contains command or log contents."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


_MEDIA = re.compile(r'\.(?:m4a|webm|opus|ogg|mp3|mp4|wav|flac|aac|ts|mkv)(?:\.|$)')


def _snapshot(directory, item_id):
    """Return inode-keyed media sizes; ignore unrelated items and metadata."""
    sizes = {}
    for path in directory.iterdir():
        name = path.name
        if not name.startswith(item_id + '.') or not _MEDIA.search(name):
            continue
        if any(suffix in name.split('.')[1:] for suffix in ('log', 'json', 'ytdl')):
            continue
        try:
            info = path.lstat()
        except FileNotFoundError:  # Atomic rename/removal during scan.
            continue
        if stat.S_ISREG(info.st_mode):
            sizes[(info.st_dev, info.st_ino)] = info.st_size
    return sizes


def _stop(process):
    # Do not poll/wait between TERM and KILL: keeping the leader unreaped
    # reserves its PID, so the process group ID cannot be reused meanwhile.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS can reject a signal to an already terminated, unreaped group.
        # Accept this only when the owned child has actually terminated.
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            raise TransferFailure('cleanup_timeout') from None
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        raise TransferFailure('cleanup_timeout') from None


def run_transfer(command, output_directory, item_id, log_handle,
                 min_free_gib=25, idle_timeout=300, max_seconds=21600,
                 poll_seconds=5):
    """Return the child exit code or raise TransferFailure, preserving partials.

    Only increases above each media inode's previous byte high-water mark reset
    inactivity. Renames, deletion, truncation, mtimes, and log writes do not.
    Commands run without a shell in their own process group. Bounds are seconds.
    """
    if (not isinstance(item_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', item_id)
            or isinstance(command, (str, bytes)) or not command):
        raise TransferFailure('invalid_arguments')
    bounds = (min_free_gib, idle_timeout, max_seconds, poll_seconds)
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds):
        raise TransferFailure('invalid_bounds')
    if min_free_gib < 0 or min(idle_timeout, max_seconds, poll_seconds) <= 0:
        raise TransferFailure('invalid_bounds')
    directory = Path(output_directory)
    process = None
    try:
        if shutil.disk_usage(directory).free < min_free_gib * 1024**3:
            raise TransferFailure('disk_floor')
        high_water = _snapshot(directory, item_id)
        started = last_progress = time.monotonic()
        process = subprocess.Popen(command, stdout=log_handle,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True, shell=False)
        while True:
            code = process.poll()
            if code is not None:
                return process.wait()
            now = time.monotonic()
            if shutil.disk_usage(directory).free < min_free_gib * 1024**3:
                raise TransferFailure('disk_floor')
            current = _snapshot(directory, item_id)
            if any(size > high_water.get(key, 0) for key, size in current.items()):
                last_progress = now
            for key, size in current.items():
                high_water[key] = max(size, high_water.get(key, 0))
            if now - started >= max_seconds:
                raise TransferFailure('hard_timeout')
            if now - last_progress >= idle_timeout:
                raise TransferFailure('idle_timeout')
            time.sleep(min(poll_seconds, max_seconds - (now - started),
                           idle_timeout - (now - last_progress)))
    except BaseException as exc:
        if process is not None:
            _stop(process)
        if isinstance(exc, (TransferFailure, KeyboardInterrupt, SystemExit)):
            raise
        raise TransferFailure('transfer_monitor_error') from None
