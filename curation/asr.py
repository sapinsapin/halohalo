"""Bounded NRSP draft transcription. Never transmits source prompts or grants audio review."""
import argparse
import asyncio
import copy
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import time
from urllib.parse import urlencode, urlsplit
import uuid

import soundfile as sf
import websockets

from .schema import digest, load_manifest, validate
from .review import atomic, sha, utc
from .cli import write_jsonl


class ASRQuotaError(RuntimeError):
    """Service-declared quota exhaustion; never retry this request."""


def check_event(event):
    if event.get('type') == 'error':
        if 'quota' in str(event.get('code', '')).lower():
            raise ASRQuotaError('ASR quota exhausted')
        raise ValueError('server error')
    if event.get('truncated') or event.get('metadata', {}).get('truncated'):
        raise ValueError('server truncation')


@dataclass(frozen=True)
class Config:
    endpoint: str = 'wss://d1ve4x2dfkyese.cloudfront.net/v1/asr'
    model: str = 'fw/whisper-large-v3-pld-hil-norm'
    language: str = 'hil'
    timeout: float = 180
    retries: int = 1
    pace: float = 1
    frame_bytes: int = 3200

    def check(self):
        u = urlsplit(self.endpoint)
        if not u.hostname or u.username or u.password or u.query or u.fragment:
            raise ValueError('invalid ASR endpoint')
        if u.scheme != 'wss' and not (u.scheme == 'ws' and u.hostname in {'localhost', '127.0.0.1', '::1'}):
            raise ValueError('ASR requires WSS')
        if self.language not in {'hil', 'fil'} or not self.model or not 0 <= self.retries <= 3 or self.timeout <= 0 or not 0 <= self.pace <= 10 or not 2 <= self.frame_bytes <= 32768 or self.frame_bytes % 2:
            raise ValueError('invalid ASR bounds')

    def identity(self):
        return dict(endpoint=self.endpoint, model=self.model, language=self.language,
                    transport='nrsp.v1', client_version=1, sample_rate=16000,
                    channels=1, encoding='pcm_s16le', frame_bytes=self.frame_bytes,
                    pace=float(self.pace), decoder_config=self.wire_config())

    def wire_config(self):
        return dict(type='config', language=self.language, sample_rate=16000,
                    interim=False, endpointing_ms=0)


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]')
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    if isinstance(value, dict):
        return {redact(k, key): redact(v, key) for k, v in value.items()}
    return value


def decode(row):
    # Verify even on cache hits. Never silently resample or downmix.
    if digest(row['audio_path']) != row['audio_sha256']:
        raise ValueError('audio checksum mismatch')
    info = sf.info(row['audio_path'])
    if info.samplerate != 16000 or info.channels != 1 or info.subtype != 'PCM_16':
        raise ValueError('prepared audio must be mono PCM16 at 16000 Hz')
    if not 0 < info.frames <= 60 * 16000 or abs(info.frames / 16000 - row['duration']) > 1 / 16000:
        raise ValueError('invalid prepared window duration')
    data, rate = sf.read(row['audio_path'], dtype='int16')
    return data.astype('<i2').tobytes()


def transcript(raw, cfg):
    if raw.get('status') != 'complete' or raw.get('config') != cfg.identity():
        raise ValueError('incomplete ASR response')
    events = [x['event'] for x in raw['events']]
    if not events or events[0].get('type') != 'hello' or events[0].get('model') != cfg.model:
        raise ValueError('hello model mismatch')
    finals = []
    coverage_ms = 0
    for event in events:
        if event.get('type') == 'error' or event.get('truncated') or event.get('metadata', {}).get('truncated'):
            raise ValueError('ASR error or truncation')
        if event.get('type') == 'result' and event.get('final') is True:
            meta = event.get('metadata', {})
            if not meta.get('model') or not isinstance(event.get('text'), str):
                raise ValueError('final result identity mismatch')
            start, duration = event.get('start_ms'), event.get('duration_ms')
            if not isinstance(start, (int, float)) or not isinstance(duration, (int, float)) or duration < 0 or abs(start-coverage_ms) > 2:
                raise ValueError('invalid final audio coverage')
            coverage_ms = start + duration
            finals.append(event['text'].strip())
    if not finals or events[-1].get('type') != 'finalize_end' or events[-1].get('id') != raw['finalize_id']:
        raise ValueError('missing matching finalization')
    if abs(coverage_ms - raw['pcm_duration_ms']) > 2:
        raise ValueError('truncated final audio coverage')
    return ' '.join(t for t in finals if t)


async def exchange(pcm, cfg, key, path, fingerprint):
    fid = uuid.uuid4().hex
    raw = dict(config=cfg.identity(), cache_key=fingerprint, finalize_id=fid,
               status='running', started_at=utc(), pcm_duration_ms=len(pcm)/32, events=[])
    started = time.monotonic()
    atomic(path, raw)
    def save(event):
        raw['events'].append(dict(elapsed_seconds=time.monotonic()-started, event=redact(event, key)))
        atomic(path, raw)
    try:
        async with asyncio.timeout(cfg.timeout):
            url = cfg.endpoint + '?' + urlencode(dict(model=cfg.model, language=cfg.language, sample_rate=16000))
            async with websockets.connect(url, subprotocols=['nrsp.v1', 'bearer.' + key], max_size=4*1024*1024, open_timeout=min(cfg.timeout, 30)) as ws:
                hello = json.loads(await ws.recv())
                save(hello)
                check_event(hello)
                if hello.get('type') != 'hello' or hello.get('model') != cfg.model:
                    raise ValueError('hello model mismatch')
                await ws.send(json.dumps(cfg.wire_config()))
                async def sender():
                    sent_at = time.monotonic()
                    for offset in range(0, len(pcm), cfg.frame_bytes):
                        await ws.send(pcm[offset:offset+cfg.frame_bytes])
                        if cfg.pace:
                            await asyncio.sleep(max(0, sent_at + min(offset+cfg.frame_bytes, len(pcm))/32000*cfg.pace-time.monotonic()))
                    await ws.send(json.dumps(dict(type='finalize', id=fid)))
                task = asyncio.create_task(sender())
                try:
                    while True:
                        event = json.loads(await ws.recv())
                        save(event)
                        check_event(event)
                        if event.get('type') == 'finalize_end':
                            if event.get('id') != fid:
                                if str(event.get('id', '')).startswith('auto-'):
                                    continue
                                raise ValueError('unexpected finalization')
                            # The peer can reply before send() returns locally.
                            # Awaiting also propagates sender failures.
                            await task
                            break
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        raw['status'] = 'complete'
        transcript(raw, cfg)
        atomic(path, raw)
        return raw
    except Exception as exc:
        raw['status'] = 'failed'
        raw['error_kind'] = type(exc).__name__
        atomic(path, raw)
        if isinstance(exc, ASRQuotaError):
            raise ASRQuotaError('ASR quota exhausted; inspect safe response ledger') from None
        raise RuntimeError('ASR request failed; inspect safe response ledger') from None


async def transcribe_one(row, cfg, key, output):
    validate(row); cfg.check()
    key = key.strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError('invalid key')
    pcm = decode(row)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    fingerprint = sha(dict(audio_sha256=row['audio_sha256'], config=cfg.identity()))
    with (root / (fingerprint+'.lock')).open('a') as lock:
        # Fail instead of blocking the event loop on an overlapping worker.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = root / (fingerprint+'.json')
        raw_path = root / (fingerprint+'.response.json')
        if not raw_path.exists():
            # Recover a completed paid attempt if interrupted before cache commit.
            for candidate in sorted(root.glob(fingerprint+'.attempt-*.json')):
                saved = json.loads(candidate.read_text())
                if saved.get('status') == 'complete' and saved.get('cache_key') == fingerprint:
                    transcript(saved, cfg)
                    atomic(raw_path, saved)
                    break
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
            if raw.get('cache_key') != fingerprint:
                raise ValueError('cache identity mismatch')
            text = transcript(raw, cfg)
        else:
            raw = None
            for attempt in range(cfg.retries+1):
                attempt_path = root / f'{fingerprint}.attempt-{uuid.uuid4().hex}.json'
                try:
                    raw = await exchange(pcm, cfg, key, attempt_path, fingerprint)
                    break
                except RuntimeError:
                    # Quota and all server-declared errors must not trigger paid retries.
                    failed = json.loads(attempt_path.read_text())
                    if any(e['event'].get('type') == 'error' for e in failed['events']) or attempt == cfg.retries:
                        raise
                    await asyncio.sleep(1)
            text = transcript(raw, cfg)
            atomic(raw_path, raw)
        result = dict(cache_key=fingerprint, config=cfg.identity(), audio_sha256=row['audio_sha256'],
                      response_sha256=sha(raw), sentence=text, status='transcribed' if text else 'empty')
        if target.exists() and json.loads(target.read_text()) != result:
            raise ValueError('cache result mismatch')
        atomic(target, result)
        return result


def apply_result(row, result):
    out = copy.deepcopy(row)
    out.update(sentence=result['sentence'], num_words=len(result['sentence'].split()),
               asr_status=result['status'], review_status='draft', llm_status='pending')
    out.pop('audio_review', None)
    out['provenance'].update(asr_config=result['config'], asr_config_sha256=sha(result['config']),
                             asr_cache_key=result['cache_key'], asr_audio_sha256=result['audio_sha256'])
    return validate(out)


async def run(args):
    cfg = Config(endpoint=args.endpoint, model=args.model, language=args.language,
                 timeout=args.timeout, retries=args.retries, pace=args.pace)
    cfg.check()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)[:args.limit]
    key = Path(args.key_file).read_text().strip()
    completed, failed = [], []
    consecutive_failures = 0
    for row in rows:
        try:
            completed.append(apply_result(row, await transcribe_one(row, cfg, key, root)))
            consecutive_failures = 0
        except Exception as exc:
            failed.append(dict(id=row['id'], error_kind=type(exc).__name__))
            consecutive_failures += 1
        stopped_early = consecutive_failures >= 2
        write_jsonl(root/'manifest.jsonl', completed)
        atomic(root/'summary.json', dict(selected=len(rows), completed=len(completed), failed=failed,
               transcribed_hours=sum(r['duration'] for r in completed if r['asr_status']=='transcribed')/3600,
               empty=sum(r['asr_status']=='empty' for r in completed), review_status='draft',
               stopped_early=stopped_early))
        if stopped_early:
            break
    if failed:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--key-file', required=True)
    p.add_argument('--endpoint', default=Config.endpoint); p.add_argument('--model', default=Config.model)
    p.add_argument('--language', choices=['hil', 'fil'], default='hil')
    p.add_argument('--limit', type=int, default=2); p.add_argument('--retries', type=int, default=1)
    p.add_argument('--timeout', type=float, default=180); p.add_argument('--pace', type=float, default=1)
    a = p.parse_args()
    if not 1 <= a.limit <= 10000:
        p.error('limit must be 1..10000')
    asyncio.run(run(a))


if __name__ == '__main__':
    main()
