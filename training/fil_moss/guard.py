"""Run one GPU stage under the shared lock, a host-memory floor and a time bound. Standard library only.

    python -m training.fil_moss.guard --max-seconds 14400 -- python -m training.fil_moss.stage1 train ...

Same rules as the Hiligaynon pilot's guard.py: refuse to start below the floor, stop the stage's own process
group (never anything else, never resident services) if MemAvailable falls below it, and hold
/tmp/sapinsapin-gpu.lock for the whole stage so stages on the shared GB10 never overlap.
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

GIB = 2 ** 30


def mem_available_bytes(meminfo="/proc/meminfo"):
    with open(meminfo) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from %s" % meminfo)


def should_stop(available_bytes, elapsed_seconds, floor_bytes, max_seconds):
    """Reason to stop, or None."""
    if available_bytes < floor_bytes:
        return "memory floor"
    if elapsed_seconds > max_seconds:
        return "time bound"
    return None


def run(command, floor_gib, max_seconds, lock_path, poll=2.0, meminfo="/proc/meminfo"):
    floor = floor_gib * GIB
    with open(lock_path, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("GPU lock %s is held by another stage; refusing to start" % lock_path)
        if mem_available_bytes(meminfo) < floor:
            raise SystemExit("less than %s GiB available; refusing to start" % floor_gib)
        proc = subprocess.Popen(command, start_new_session=True)
        start, reason, lowest = time.monotonic(), None, None
        try:
            while proc.poll() is None:
                avail = mem_available_bytes(meminfo)
                lowest = avail if lowest is None else min(lowest, avail)
                reason = should_stop(avail, time.monotonic() - start, floor, max_seconds)
                if reason:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(poll)
        finally:
            code = proc.wait()
            print(json.dumps({"command": command, "returncode": code, "stop_reason": reason,
                              "elapsed_seconds": time.monotonic() - start,
                              "lowest_available_gib": None if lowest is None else lowest / GIB}), flush=True)
    return code or (1 if reason else 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--min-available-gib", type=float, default=25.0)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--lock", default="/tmp/sapinsapin-gpu.lock")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("no command")
    return run(command, a.min_available_gib, a.max_seconds, a.lock)


if __name__ == "__main__":
    raise SystemExit(main())
