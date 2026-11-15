"""Read-only audit of a completed persistent-ASR batch; never contacts the server."""
import argparse
from dataclasses import fields
import json
from pathlib import Path

from .asr import apply_result, decode
from .asr_session import Config, transcript
from .review import sha
from .schema import digest, load_manifest


def audit(inputs, manifest, cache):
    inputs, manifest, cache = Path(inputs), Path(manifest), Path(cache)
    input_hash, output_hash = digest(inputs), digest(manifest)
    source = load_manifest(inputs)
    rows = load_manifest(manifest)
    by_id = {r['id']: r for r in source}
    if (len(by_id) != len(source) or len({r['id'] for r in rows}) != len(rows)
            or set(by_id) != {r['id'] for r in rows}):
        raise ValueError('input/output IDs must be unique and cover the complete batch')
    configs = set()
    for row in rows:
        original = by_id[row['id']]
        key = row['provenance']['asr_cache_key']
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('invalid cache key')
        result = json.loads((cache / (key + '.json')).read_text())
        raw = json.loads((cache / (key + '.response.json')).read_text())
        cfg = Config(**{k: v for k, v in result['config'].items()
                        if k in {f.name for f in fields(Config)}})
        cfg.check()
        if cfg.identity() != result['config']:
            raise ValueError('unsupported cache configuration')
        expected = sha({'audio_sha256': original['audio_sha256'], 'config': cfg.identity()})
        if key != expected or result['cache_key'] != key or raw['cache_key'] != key:
            raise ValueError('cache identity mismatch')
        if sha(raw) != result['response_sha256']:
            raise ValueError('raw response checksum mismatch')
        text, _ = transcript(raw, cfg)
        if (result['audio_sha256'] != original['audio_sha256'] or result['sentence'] != text
                or result['status'] != ('transcribed' if text else 'empty')
                or apply_result(original, result) != row):
            raise ValueError('result differs from original input or validated response')
        # decode verifies the file checksum and format, and checks wire sample count.
        pcm = decode(original)
        if raw['pcm_samples'] != len(pcm) // 2:
            raise ValueError('wire sample count differs from source audio')
        configs.add(sha(cfg.identity()))
    if digest(inputs) != input_hash or digest(manifest) != output_hash:
        raise ValueError('manifest changed during audit')
    return {'rows': len(rows), 'nonempty': sum(bool(r['sentence'].strip()) for r in rows),
            'input_sha256': input_hash, 'manifest_sha256': output_hash,
            'config_sha256': sorted(configs), 'server_requests': 0,
            'scope': 'Protocol, content binding and exact decoded sample count; not transcription quality approval.'}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--inputs', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    report = audit(a.inputs, a.manifest, a.cache)
    target = Path(a.report)
    with target.open('x') as f:
        f.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
