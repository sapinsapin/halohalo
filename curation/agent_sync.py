"""Claim board so several agents (Claude Code, Antigravity, people) can share dwnd-01 and the Hub without colliding.

State lives in one directory (default ~/chika-curation/AGENT_SYNC on dwnd-01): `board.json` holds active claims,
`log.jsonl` is an append-only history of every claim, release and note. All writes hold an exclusive file lock.
Claims expire after their TTL unless renewed, so a crashed agent cannot block others forever.

Resources are free-form strings; use these conventions:
  gpu:dwnd            the GB10 GPU (batch ASR, evals)          hub:sapinsapin/chika   commits to the dataset repo
  data:<output-dir>   a curation output directory you write    lang:<code>            a language's next batch
  job:<name>          anything else long-running

  python3 agent_sync.py status
  python3 agent_sync.py claim gpu:dwnd --agent claude --note "ceb batch v2" --ttl 6
  python3 agent_sync.py renew gpu:dwnd --agent claude --ttl 6
  python3 agent_sync.py release gpu:dwnd --agent claude --note "done; output remote-ceb-asr-v2"
  python3 agent_sync.py note --agent claude "published ceb_dwnd_batch_asr_32311_v1"
  python3 agent_sync.py log [-n 30]

`claim` exits 1 (and changes nothing) when another agent holds an unexpired claim. Standard library only.
"""
import argparse
import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get('AGENT_SYNC_DIR', Path.home() / 'chika-curation' / 'AGENT_SYNC'))


def now():
    return time.time()


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


@contextmanager
def locked(root=None):
    root = Path(root or ROOT)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'board.json'
        board = json.loads(path.read_text()) if path.exists() else {'claims': {}}
        yield board
        tmp = root / 'board.json.tmp'
        tmp.write_text(json.dumps(board, indent=2, sort_keys=True) + '\n')
        tmp.replace(path)


def log(event, root=None):
    root = Path(root or ROOT)
    event = dict(event, at=now())
    with open(root / 'log.jsonl', 'a') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')


def active(board):
    t = now()
    return {k: v for k, v in board['claims'].items() if v['expires'] > t}


def claim(resource, agent, note='', ttl=4.0, root=None):
    with locked(root) as board:
        holder = active(board).get(resource)
        if holder and holder['agent'] != agent:
            return False, holder
        board['claims'] = active(board)
        entry = {'agent': agent, 'note': note, 'since': holder['since'] if holder else now(),
                 'expires': now() + ttl * 3600, 'host': os.uname().nodename}
        board['claims'][resource] = entry
    log({'type': 'claim', 'resource': resource, 'agent': agent, 'note': note, 'ttl_h': ttl}, root)
    return True, entry


def release(resource, agent, note='', root=None):
    with locked(root) as board:
        holder = board['claims'].get(resource)
        if holder and holder['agent'] != agent and holder['expires'] > now():
            return False, holder
        board['claims'].pop(resource, None)
        board['claims'] = active(board)
    log({'type': 'release', 'resource': resource, 'agent': agent, 'note': note}, root)
    return True, None


def note(agent, text, root=None):
    with locked(root):
        pass
    log({'type': 'note', 'agent': agent, 'note': text}, root)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    for name in ('claim', 'renew'):
        c = sub.add_parser(name)
        c.add_argument('resource')
        c.add_argument('--agent', required=True)
        c.add_argument('--note', default='')
        c.add_argument('--ttl', type=float, default=4.0, help='hours')
    r = sub.add_parser('release')
    r.add_argument('resource')
    r.add_argument('--agent', required=True)
    r.add_argument('--note', default='')
    n = sub.add_parser('note')
    n.add_argument('text')
    n.add_argument('--agent', required=True)
    lg = sub.add_parser('log')
    lg.add_argument('-n', type=int, default=30)
    a = p.parse_args()
    if a.cmd == 'status':
        with locked() as board:
            claims = active(board)
            board['claims'] = claims
        if not claims:
            print('no active claims')
        for k, v in sorted(claims.items()):
            print(f"{k:32} {v['agent']:12} until {iso(v['expires'])}  {v['note']}")
    elif a.cmd in ('claim', 'renew'):
        ok, entry = claim(a.resource, a.agent, a.note, a.ttl)
        if not ok:
            print(f"HELD by {entry['agent']} until {iso(entry['expires'])}: {entry['note']}")
            sys.exit(1)
        print(f"{a.cmd}ed {a.resource} until {iso(entry['expires'])}")
    elif a.cmd == 'release':
        ok, entry = release(a.resource, a.agent, a.note)
        if not ok:
            print(f"HELD by {entry['agent']}; not released")
            sys.exit(1)
        print(f'released {a.resource}')
    elif a.cmd == 'note':
        note(a.agent, a.text)
        print('noted')
    else:
        path = ROOT / 'log.jsonl'
        lines = path.read_text().splitlines()[-a.n:] if path.exists() else []
        for line in lines:
            e = json.loads(line)
            print(f"{iso(e['at'])} {e['agent']:12} {e['type']:8} {e.get('resource', ''):28} {e.get('note', '')}")


if __name__ == '__main__':
    main()
