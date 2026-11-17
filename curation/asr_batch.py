"""Offline batched Whisper transcription on a local GPU: no server, no network requests, no finalize deadline.

The streaming route (``curation.asr_session``) sends one clip at a time to a shared live service and must never replay
a request that may have been processed. This module instead loads the same fine-tuned checkpoint locally from a
pinned Hugging Face revision and decodes many clips per forward pass. Because nothing leaves the machine, a stopped
run is safely resumable: finished chunks are kept and the rest are recomputed.

Output rows keep the streaming contract: each is the input row with ``sentence``, ``num_words``, ``asr_status``
(``transcribed`` or ``empty``), ``review_status='draft'``, ``llm_status='pending'`` and the provenance fields
``asr_config``, ``asr_config_sha256``, ``asr_cache_key`` and ``asr_audio_sha256``. ``asr_config['engine']`` is
``local-batch`` so batch and streaming transcripts are never confused.

Layout of OUTPUT: ``plan.json`` (frozen input hash, config and chunk boundaries), ``chunks/chunk-NNNNN.jsonl``,
``worker.json``, and after the last chunk ``manifest.jsonl`` (input order) plus ``summary.json``.
Rows longer than one 30 s Whisper window are not transcribed; they are listed in ``plan.json`` and the summary.

Usage (dwnd-01, cwd /home/iotim/chika-curation):
  PYTHONPATH=. .venv-batchasr/bin/python -m curation.asr_batch --manifest IN.jsonl --output OUT \
      --model sapinsapin/whisper-large-v3-pld-fil-norm --revision <sha> --language fil
"""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import time
import zlib

from .review import atomic, sha, utc
from .schema import digest, load_manifest, validate

SAMPLE_RATE = 16000
WINDOW_SECONDS = 30.0
GPU_LOCK = '/tmp/sapinsapin-gpu.lock'
# Both deployed fine-tunes were trained with Whisper's Tagalog token (generation_config.json language=tagalog).
LANGUAGE_TOKENS = {'fil': 'tagalog', 'hil': 'tagalog', 'ceb': 'tagalog', 'ilo': 'tagalog', 'bik': 'tagalog'}


def make_config(model, revision, language, dtype='float16', num_beams=1, max_new_tokens=440,
                compression_ratio_threshold=2.4, temperatures=(0.2, 0.4, 0.6, 0.8, 1.0)):
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('revision must be a full 40-character commit sha')
    if language not in LANGUAGE_TOKENS or dtype not in ('bfloat16', 'float16', 'float32'):
        raise ValueError('unsupported language or dtype')
    if not 1 <= num_beams <= 8 or not 1 <= max_new_tokens <= 440:
        raise ValueError('invalid decoding bounds')
    if compression_ratio_threshold is not None and compression_ratio_threshold <= 1:
        raise ValueError('invalid compression ratio threshold')
    if any(not 0 < t <= 1.5 for t in temperatures) or list(temperatures) != sorted(temperatures):
        raise ValueError('fallback temperatures must be ascending in (0, 1.5]')
    return {'engine': 'local-batch', 'version': 2, 'model': model, 'revision': revision, 'language': language,
            'whisper_language': LANGUAGE_TOKENS[language], 'task': 'transcribe', 'dtype': dtype,
            'num_beams': num_beams, 'max_new_tokens': max_new_tokens, 'sample_rate': SAMPLE_RATE,
            'compression_ratio_threshold': compression_ratio_threshold, 'fallback_temperatures': list(temperatures)}


def compression_ratio(text):
    """gzip-style repetition measure used by Whisper and faster-whisper (looping output compresses well)."""
    data = text.encode()
    return len(data) / len(zlib.compress(data)) if data else 0.0


def cache_key(row, config):
    return sha({'audio_sha256': row['audio_sha256'], 'config': config})


def local_path(path, path_map=()):
    """Where to read a clip on this host. Rows keep their original audio_path; only reads are redirected."""
    for old, new in path_map:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def load_audio(row, path_map=()):
    """Verified 16 kHz mono PCM16 samples as float32 in [-1, 1). Never resamples or downmixes."""
    import numpy as np
    import soundfile as sf
    path = local_path(row['audio_path'], path_map)
    if digest(path) != row['audio_sha256']:
        raise ValueError(f"audio checksum mismatch for {row['id']}")
    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.subtype != 'PCM_16':
        raise ValueError(f"prepared audio must be mono PCM16 at 16000 Hz: {row['id']}")
    if not 0 < info.frames <= WINDOW_SECONDS * SAMPLE_RATE or abs(info.frames / SAMPLE_RATE - row['duration']) > 1 / SAMPLE_RATE:
        raise ValueError(f"invalid prepared window duration: {row['id']}")
    data, _ = sf.read(path, dtype='int16')
    return data.astype(np.float32) / 32768.0


class WhisperTranscriber:
    """Batched greedy/beam Whisper decoding with transformers on CUDA."""

    def __init__(self, config):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        self.torch, self.config = torch, config
        self.processor = WhisperProcessor.from_pretrained(config['model'], revision=config['revision'])
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config['model'], revision=config['revision'], dtype=getattr(torch, config['dtype'])).to('cuda').eval()

    def _generate(self, inputs, mask, **sampling):
        c = self.config
        with self.torch.inference_mode():
            ids = self.model.generate(inputs, attention_mask=mask, language=c['whisper_language'], task=c['task'],
                                      num_beams=c['num_beams'], max_new_tokens=c['max_new_tokens'],
                                      return_timestamps=False, **sampling)
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]

    def __call__(self, audios):
        """Greedy decode; clips whose text loops (compression ratio above the threshold) are re-decoded with
        seeded sampling at rising temperatures, as faster-whisper's fallback does. Returns (text, meta) pairs."""
        c = self.config
        features = self.processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors='pt',
                                                    return_attention_mask=True)
        inputs = features.input_features.to('cuda', dtype=self.model.dtype)
        mask = features.attention_mask.to('cuda')

        def regenerate(indices, temperature):
            self.torch.manual_seed(int(temperature * 1000))  # reproducible for the same batch on the same GPU
            idx = self.torch.tensor(indices, device='cuda')
            return self._generate(inputs[idx], mask[idx], do_sample=True, temperature=temperature)
        return with_fallback(self._generate(inputs, mask), regenerate, c)


def with_fallback(texts, regenerate, config):
    """Replace looping greedy outputs with the first sampled retry that no longer loops; flag the rest."""
    texts = list(texts)
    meta = [{'temperature': 0.0, 'compression_ratio': round(compression_ratio(t), 3)} for t in texts]
    threshold = config.get('compression_ratio_threshold')
    pending = [i for i, t in enumerate(texts) if threshold and compression_ratio(t) > threshold]
    for temperature in config.get('fallback_temperatures', []):
        if not pending:
            break
        retry = regenerate(pending, temperature)
        if len(retry) != len(pending):
            raise RuntimeError('fallback returned the wrong number of results')
        still = []
        for i, t in zip(pending, retry):
            if compression_ratio(t) <= threshold:
                meta[i] = {'temperature': temperature, 'compression_ratio': round(compression_ratio(t), 3),
                           'fallback_from_ratio': meta[i]['compression_ratio']}
                texts[i] = t
            else:
                still.append(i)
        pending = still
    for i in pending:  # no temperature removed the loop: keep the greedy text and flag it for review
        meta[i]['repetition_unresolved'] = True
    return list(zip(texts, meta))


def mem_available_gib():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return math.inf  # non-Linux test hosts


def freeze_plan(manifest, output, config, chunk_rows):
    rows = load_manifest(manifest)
    ids = [r['id'] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError('input must be non-empty with unique identifiers')
    long_rows = [r['id'] for r in rows if float(r['duration']) > WINDOW_SECONDS]
    keep = [r['id'] for r in rows if float(r['duration']) <= WINDOW_SECONDS]
    chunks = [keep[i:i + chunk_rows] for i in range(0, len(keep), chunk_rows)]
    plan = {'version': 1, 'input': str(Path(manifest).resolve()), 'input_sha256': digest(manifest), 'config': config,
            'config_sha256': sha(config), 'chunk_rows': chunk_rows, 'chunks': chunks, 'not_attempted_over_30s': long_rows}
    plan_path = Path(output) / 'plan.json'
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old != plan:
            raise ValueError('existing plan differs (input, config or chunking changed); use a new output directory')
    else:
        atomic(plan_path, plan)
    return plan, {r['id']: r for r in rows}


def result_row(row, result, config):
    text, meta = result if isinstance(result, tuple) else (result, None)
    out = copy.deepcopy(row)
    out.update(sentence=text, num_words=len(text.split()), asr_status='transcribed' if text else 'empty',
               review_status='draft', llm_status='pending')
    out.pop('audio_review', None)
    out['provenance'] = dict(out.get('provenance') or {}, asr_config=config, asr_config_sha256=sha(config),
                             asr_cache_key=cache_key(row, config), asr_audio_sha256=row['audio_sha256'])
    if meta is not None:
        out['provenance']['asr_decode'] = meta
    return validate(out)


def run(manifest, output, config, batch_size=16, chunk_rows=256, min_mem_gib=25.0, transcriber=None,
        audio_loader=load_audio, gpu_lock=GPU_LOCK, mem=mem_available_gib, path_map=()):
    if not 1 <= batch_size <= 128 or not 1 <= chunk_rows <= 10000 or min_mem_gib < 0:
        raise ValueError('invalid limits')
    output = Path(output).resolve()
    (output / 'chunks').mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, by_id = freeze_plan(manifest, output, config, chunk_rows)
        state = {'pid': os.getpid(), 'status': 'running', 'started_at': utc(), 'chunks': len(plan['chunks']),
                 'chunks_done': 0, 'rows_done': 0, 'audio_seconds': 0.0, 'compute_seconds': 0.0}
        atomic(output / 'worker.json', state)
        try:
            model = None
            for n, chunk in enumerate(plan['chunks']):
                path = output / 'chunks' / f'chunk-{n:05d}.jsonl'
                if path.exists():
                    done = load_manifest(path)
                    if [r['id'] for r in done] != chunk:
                        raise ValueError(f'{path.name} does not match the frozen plan')
                    state['chunks_done'] += 1
                    state['rows_done'] += len(done)
                    continue
                if digest(manifest) != plan['input_sha256']:
                    raise ValueError('input manifest changed during the run')
                if mem() < min_mem_gib:
                    raise RuntimeError('MemAvailable below floor')
                rows = [by_id[i] for i in chunk]
                audios = [audio_loader(r, path_map) for r in rows]  # verify every clip before any GPU work
                order = sorted(range(len(rows)), key=lambda i: len(audios[i]))  # similar lengths per batch
                texts = [None] * len(rows)
                with open(gpu_lock, 'a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if model is None:
                        model = transcriber or WhisperTranscriber(config)
                    start = time.monotonic()
                    for b in range(0, len(order), batch_size):
                        idx = order[b:b + batch_size]
                        decoded = model([audios[i] for i in idx])
                        if len(decoded) != len(idx):
                            raise RuntimeError('transcriber returned the wrong number of results')
                        for i, t in zip(idx, decoded):
                            texts[i] = t
                    elapsed = time.monotonic() - start
                out = [result_row(r, t, config) for r, t in zip(rows, texts)]
                tmp = path.with_suffix('.tmp')
                tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
                tmp.replace(path)
                seconds = sum(float(r['duration']) for r in rows)
                state['chunks_done'] += 1
                state['rows_done'] += len(rows)
                state['audio_seconds'] = round(state['audio_seconds'] + seconds, 3)
                state['compute_seconds'] = round(state['compute_seconds'] + elapsed, 3)
                atomic(output / 'worker.json', state)
            rows = [r for n in range(len(plan['chunks'])) for r in load_manifest(output / 'chunks' / f'chunk-{n:05d}.jsonl')]
            tmp = output / 'manifest.jsonl.tmp'
            tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            tmp.replace(output / 'manifest.jsonl')
            summary = {'rows': len(rows), 'transcribed': sum(r['asr_status'] == 'transcribed' for r in rows),
                       'empty': sum(r['asr_status'] == 'empty' for r in rows),
                       'temperature_fallbacks': sum(bool(r['provenance'].get('asr_decode', {}).get('temperature'))
                                                    for r in rows),
                       'repetition_unresolved': [r['id'] for r in rows
                                                 if r['provenance'].get('asr_decode', {}).get('repetition_unresolved')],
                       'not_attempted_over_30s': plan['not_attempted_over_30s'],
                       'audio_hours': sum(float(r['duration']) for r in rows) / 3600,
                       'manifest_sha256': digest(output / 'manifest.jsonl'), 'input_sha256': plan['input_sha256'],
                       'config': config, 'review_status': 'draft', 'network_requests': 0}
            atomic(output / 'summary.json', summary)
            state.update(status='complete', finished_at=utc())
            return summary
        except BaseException as exc:
            state.update(status='stopped_for_inspection', error=f'{type(exc).__name__}: {exc}'[:400], finished_at=utc())
            raise
        finally:
            atomic(output / 'worker.json', state)


def audit(manifest, output, path_map=()):
    """Read-only check that a finished batch output covers the frozen input and matches its config and audio."""
    output = Path(output)
    plan = json.loads((output / 'plan.json').read_text())
    if digest(manifest) != plan['input_sha256']:
        raise ValueError('input manifest differs from the frozen plan')
    source = {r['id']: r for r in load_manifest(manifest)}
    rows = load_manifest(output / 'manifest.jsonl')
    expected = [i for chunk in plan['chunks'] for i in chunk]
    if [r['id'] for r in rows] != expected or set(expected) | set(plan['not_attempted_over_30s']) != set(source):
        raise ValueError('output does not cover the frozen input exactly')
    config = plan['config']
    asr_fields = {'sentence', 'num_words', 'asr_status', 'review_status', 'llm_status', 'provenance', 'audio_review'}
    for r in rows:
        s = source[r['id']]
        p = r['provenance']
        if p['asr_config'] != config or p['asr_config_sha256'] != sha(config) or p['asr_cache_key'] != cache_key(s, config):
            raise ValueError(f"config or cache identity mismatch for {r['id']}")
        if p['asr_audio_sha256'] != s['audio_sha256'] or digest(local_path(r['audio_path'], path_map)) != s['audio_sha256']:
            raise ValueError(f"audio mismatch for {r['id']}")
        if {k: v for k, v in r.items() if k not in asr_fields} != {k: v for k, v in s.items() if k not in asr_fields}:
            raise ValueError(f"non-ASR fields changed for {r['id']}")
        if {k: v for k, v in p.items() if not k.startswith('asr_')} != {k: v for k, v in (s.get('provenance') or {}).items() if not k.startswith('asr_')}:
            raise ValueError(f"provenance changed for {r['id']}")
        if r['num_words'] != len(r['sentence'].split()) or r['asr_status'] != ('transcribed' if r['sentence'] else 'empty'):
            raise ValueError(f"inconsistent ASR fields for {r['id']}")
        if r['review_status'] != 'draft':
            raise ValueError('batch ASR output must stay draft')
    return {'rows': len(rows), 'not_attempted_over_30s': len(plan['not_attempted_over_30s']),
            'manifest_sha256': digest(output / 'manifest.jsonl'), 'input_sha256': plan['input_sha256'],
            'config_sha256': plan['config_sha256'], 'scope': 'complete frozen input', 'network_requests': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--language', required=True, choices=sorted(LANGUAGE_TOKENS))
    p.add_argument('--dtype', default='float16')
    p.add_argument('--num-beams', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--chunk-rows', type=int, default=256)
    p.add_argument('--min-mem-gib', type=float, default=25.0)
    p.add_argument('--audio-path-map', action='append', default=[], metavar='OLD=NEW',
                   help='read clips whose audio_path starts with OLD from NEW instead (rows keep OLD)')
    p.add_argument('--audit', action='store_true', help='only audit a finished output')
    a = p.parse_args()
    path_map = tuple(tuple(m.split('=', 1)) for m in a.audio_path_map)
    if any(len(m) != 2 or not m[0] or not m[1] for m in path_map):
        raise SystemExit('--audio-path-map must be OLD=NEW')
    config = make_config(a.model, a.revision, a.language, a.dtype, a.num_beams)
    if a.audit:
        print(json.dumps(audit(a.manifest, a.output, path_map), indent=2))
        return
    print(json.dumps(run(a.manifest, a.output, config, a.batch_size, a.chunk_rows, a.min_mem_gib,
                         path_map=path_map), indent=2))


if __name__ == '__main__':
    main()
