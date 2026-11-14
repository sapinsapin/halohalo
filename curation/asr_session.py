"""Bounded persistent NRSP transcription; any uncertain attempt stops the session."""
import argparse
import asyncio
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import time
from urllib.parse import urlencode
import uuid

import websockets
from .asr import Config as BaseConfig, ASRQuotaError, check_event, decode, redact, apply_result
from .schema import validate, load_manifest
from .review import atomic, sha, utc
from .cli import write_jsonl


@dataclass(frozen=True)
class Config(BaseConfig):
    timeout: float = 60
    retries: int = 0
    pace: float = 0

    def check(self):
        super().check()
        if self.retries != 0:
            raise ValueError('persistent sessions never retry')

    def identity(self):
        return {**super().identity(), 'transport': 'nrsp.v1.persistent', 'client_version': 2}


def transcript(raw, cfg):
    """Validate absolute wire clocks before deriving an item-relative view."""
    if raw.get('status') != 'complete' or raw.get('config') != cfg.identity():
        raise ValueError('incomplete ASR response')
    handshake = raw['session_handshake']
    if handshake.get('session_id') != raw.get('session_id'):
        raise ValueError('session handshake identity mismatch')
    if sha(handshake) != raw['session_handshake_sha256']:
        raise ValueError('handshake checksum mismatch')
    hello = handshake['hello']
    check_event(hello)
    if hello.get('type') != 'hello' or hello.get('model') != cfg.model or handshake['config'] != cfg.wire_config():
        raise ValueError('hello model mismatch')
    start, count = raw['session_start_sample'], raw['pcm_samples']
    if type(start) is not int or start < 0 or type(count) is not int or not 0 < count <= 960000:
        raise ValueError('invalid sample clock')
    lo = hi = start
    pending = None
    seen = set()
    finals, normalized = [], []
    events = raw['events']
    for entry in events:
        event = entry['event']
        check_event(event)
        kind = event.get('type')
        if kind == 'result':
            if event.get('final') is not True or pending is not None:
                raise ValueError('unexpected result')
            meta = event.get('metadata', {})
            fid = meta.get('finalize_id')
            if not isinstance(fid, str) or fid in seen or (fid != raw['finalize_id'] and not fid.startswith('auto-')):
                raise ValueError('unexpected result finalize id')
            if not meta.get('model') or not isinstance(event.get('text'), str):
                raise ValueError('missing result identity')
            s, d = event.get('start_ms'), event.get('duration_ms')
            if type(s) is not int or type(d) is not int or s < 0 or d < 0:
                raise ValueError('invalid wire clock')
            # Both fields are independently floored by the server. Track all
            # feasible sample boundaries rather than accumulating rounded ms.
            lo, hi = max(lo, s * 16), min(hi, s * 16 + 15)
            if lo > hi:
                raise ValueError('noncontiguous coverage')
            lo, hi = lo + d * 16, hi + d * 16 + 15
            if lo > start + count:
                raise ValueError('excess coverage')
            pending = fid
            finals.append(event['text'].strip())
            # Only result-level clocks are normalized; raw word timestamps stay
            # in the immutable wire events and must not be mixed into this view.
            normalized.append({'text':event['text'], 'start_ms':s-start/16,
                               'duration_ms':d, 'finalize_id':fid})
        elif kind == 'finalize_end':
            fid = event.get('id')
            if fid != pending or fid in seen:
                raise ValueError('duplicate or mismatched finalization')
            seen.add(fid)
            pending = None
            if fid == raw['finalize_id'] and entry is not events[-1]:
                raise ValueError('events after item finalization')
        else:
            raise ValueError('unexpected wire event')
    if not finals or pending is not None or raw['finalize_id'] not in seen or not lo <= start + count <= hi:
        raise ValueError('missing finalization or truncated coverage')
    return ' '.join(t for t in finals if t), normalized


class Session:
    def __init__(self, cfg, key, output):
        cfg.check()
        self.cfg, self.key, self.root = cfg, key.strip(), Path(output)
        if not self.key or any(c.isspace() for c in self.key):
            raise ValueError('invalid key')
        self.root.mkdir(parents=True, exist_ok=True)
        self.ws = None
        self.samples = 0
        self.broken = False
        self.session_id = uuid.uuid4().hex
        self.handshake = None

    async def close(self):
        if self.ws is not None:
            ws, self.ws = self.ws, None
            await ws.close()

    async def connect(self):
        url = self.cfg.endpoint + '?' + urlencode(dict(model=self.cfg.model, language=self.cfg.language, sample_rate=16000))
        self.ws = await websockets.connect(url, subprotocols=['nrsp.v1', 'bearer.' + self.key], max_size=4*1024*1024, open_timeout=min(30, self.cfg.timeout))
        hello = redact(json.loads(await self.ws.recv()), self.key)
        self.handshake = dict(session_id=self.session_id, hello=hello, config=self.cfg.wire_config(), started_at=utc())
        atomic(self.root / (self.session_id + '.session.json'), self.handshake)
        check_event(hello)
        if hello.get('type') != 'hello' or hello.get('model') != self.cfg.model:
            raise ValueError('hello model mismatch')
        await self.ws.send(json.dumps(self.cfg.wire_config()))

    async def exchange(self, pcm, fingerprint, path):
        raw = dict(config=self.cfg.identity(), cache_key=fingerprint, finalize_id=uuid.uuid4().hex,
                   status='running', started_at=utc(), session_id=self.session_id,
                   session_start_sample=self.samples, pcm_samples=len(pcm)//2, events=[])
        atomic(path, raw)
        try:
            async with asyncio.timeout(self.cfg.timeout):
                if self.ws is None:
                    await self.connect()
                raw.update(session_handshake=self.handshake, session_handshake_sha256=sha(self.handshake))
                atomic(path, raw)
                started = time.monotonic()
                async def sender():
                    for offset in range(0, len(pcm), self.cfg.frame_bytes):
                        await self.ws.send(pcm[offset:offset+self.cfg.frame_bytes])
                        if self.cfg.pace:
                            await asyncio.sleep(max(0, started + min(offset+self.cfg.frame_bytes, len(pcm))/32000*self.cfg.pace-time.monotonic()))
                    await self.ws.send(json.dumps(dict(type='finalize', id=raw['finalize_id'])))
                task = asyncio.create_task(sender())
                try:
                    while True:
                        event = redact(json.loads(await self.ws.recv()), self.key)
                        raw['events'].append(dict(elapsed_seconds=time.monotonic()-started, event=event))
                        atomic(path, raw)
                        check_event(event)
                        if event.get('type') == 'finalize_end' and event.get('id') == raw['finalize_id']:
                            await task
                            break
                        if event.get('type') == 'finalize_end' and not str(event.get('id', '')).startswith('auto-'):
                            raise ValueError('unexpected finalize')
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            raw['status'] = 'complete'
            transcript(raw, self.cfg)
            atomic(path, raw)
            self.samples += len(pcm)//2
            return raw
        except BaseException as exc:
            self.broken = True
            raw.update(status='failed', error_kind=type(exc).__name__)
            atomic(path, raw)
            await self.close()
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, ASRQuotaError):
                raise ASRQuotaError('ASR quota exhausted; session stopped') from None
            raise RuntimeError('ASR session failed; inspect response ledger') from None

    async def transcribe_one(self, row):
        if self.broken:
            raise RuntimeError('session is unusable after failure')
        validate(row)
        pcm = decode(row)
        fingerprint = sha(dict(audio_sha256=row['audio_sha256'], config=self.cfg.identity()))
        with (self.root / (fingerprint+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = self.root / (fingerprint+'.json')
            response = self.root / (fingerprint+'.response.json')
            raw = None
            if response.exists():
                raw = json.loads(response.read_text())
            else:
                attempts = sorted(self.root.glob(fingerprint+'.attempt-*.json'))
                for candidate in attempts:
                    saved = json.loads(candidate.read_text())
                    if saved.get('status') == 'complete':
                        raw = saved
                        break
                if raw is None and attempts:
                    raise RuntimeError('uncertain prior paid attempt; manual reconciliation required')
            if raw is None:
                raw = await self.exchange(pcm, fingerprint, self.root / f'{fingerprint}.attempt-{uuid.uuid4().hex}.json')
            if raw.get('cache_key') != fingerprint or raw.get('pcm_samples') != len(pcm)//2:
                raise ValueError('cache identity mismatch')
            text, normalized = transcript(raw, self.cfg)
            atomic(response, raw)
            result = dict(cache_key=fingerprint, config=self.cfg.identity(), audio_sha256=row['audio_sha256'],
                          response_sha256=sha(raw), sentence=text, status='transcribed' if text else 'empty')
            if target.exists() and json.loads(target.read_text()) != result:
                raise ValueError('cache result mismatch')
            atomic(target, result)
            return result


async def _run_locked(args):
    cfg = Config(endpoint=args.endpoint, model=args.model, language=args.language, timeout=args.timeout, pace=args.pace)
    if not 1 <= args.max_items <= 10000:
        raise ValueError('max-items must be 1..10000')
    rows = load_manifest(args.manifest)[:args.max_items]
    session = Session(cfg, Path(args.key_file).read_text(), args.output)
    completed, failed = [], []
    try:
        for row in rows:
            try:
                completed.append(apply_result(row, await session.transcribe_one(row)))
            except Exception as exc:
                failed.append(dict(id=row['id'], error_kind=type(exc).__name__))
            write_jsonl(session.root/'manifest.jsonl', completed)
            atomic(session.root/'summary.json', dict(selected=len(rows), completed=len(completed), failed=failed,
                   stopped_early=bool(failed), review_status='draft', transcribed_hours=sum(r['duration'] for r in completed if r['asr_status']=='transcribed')/3600,
                   empty=sum(r['asr_status']=='empty' for r in completed)))
            if failed:
                break
    finally:
        await session.close()
    if failed:
        raise SystemExit(1)


async def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await _run_locked(args)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--key-file', required=True)
    p.add_argument('--endpoint', default=Config.endpoint); p.add_argument('--model', default=Config.model)
    p.add_argument('--language', choices=['hil', 'fil'], default='hil')
    p.add_argument('--max-items', type=int, default=2)
    p.add_argument('--timeout', type=float, default=60); p.add_argument('--pace', type=float, default=0)
    asyncio.run(run(p.parse_args()))


if __name__ == '__main__':
    main()
