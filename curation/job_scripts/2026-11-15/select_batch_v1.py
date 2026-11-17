"""Freeze the remaining Filipino or Hiligaynon pool for offline batch ASR (curation.asr_batch). Sends nothing.

Same exclusions as the streaming selectors (select_v6.py, select_hil_q3.py): every audio hash in any JSONL under any
remote-<lang>-asr-* root (queue inputs and outputs, merges, quarantined requests, benchmarks) and, for Filipino,
every VAD v1 candidate. Also excludes rows over 30 s (one Whisper window) and zero-length rows, and lists them.
Writes OUTPUT/input.jsonl (pool order) and OUTPUT/selection.json.

Usage (dwnd-01, cwd /home/iotim/chika-curation, PYTHONPATH=.): python select_batch_v1.py fil|hil OUTPUT [--dry-run]
"""
import json, sys
from pathlib import Path
from curation.schema import digest

out = Path('/home/iotim/chika-curation/curation/outputs')
lang, target = sys.argv[1], Path(sys.argv[2]).resolve()
dry = '--dry-run' in sys.argv
if lang == 'fil':
    pool = out / 'remote-fil-vad-dedup-v1/deduplicated/manifest.jsonl'
    if digest(pool) != '99df3ba812e981e8dfafff1639465dfc8ade3ce6b4755c82548d84d7467cddfd':
        raise SystemExit('pool manifest changed')
elif lang == 'hil':
    pool = out / 'remote-hil-vad-v1/vad/manifest.jsonl'
    if digest(pool) != json.loads((out / 'remote-hil-vad-v1/pcm-audit.json').read_text())['candidate_manifest_sha256']:
        raise SystemExit('candidate manifest does not match its PCM audit')
else:
    raise SystemExit('language must be fil or hil')

excluded, files = set(), 0
roots = sorted(out.glob(f'remote-{lang}-asr-*')) + [out / 'batch-asr-bench-v2']
for d in roots:
    for f in d.rglob('*.jsonl'):
        files += 1
        for line in f.open():
            if '"audio_sha256"' in line:
                try:
                    excluded.add(json.loads(line)['audio_sha256'])
                except (ValueError, KeyError):
                    pass
if lang == 'fil':
    for line in (out / 'remote-fil-vad-v1/vad/manifest.jsonl').open():
        excluded.add(json.loads(line)['audio_sha256'])
rows = [json.loads(l) for l in pool.open()]
if len({r['id'] for r in rows}) != len(rows):
    raise SystemExit('duplicate ids in pool')
fresh = [r for r in rows if r['audio_sha256'] not in excluded]
long_rows = [r['id'] for r in fresh if float(r['duration']) > 30.0]
zero = [r['id'] for r in fresh if float(r['duration']) <= 0]
keep, seen = [], set()
for r in fresh:
    if 0 < float(r['duration']) <= 30.0 and r['audio_sha256'] not in seen:
        seen.add(r['audio_sha256'])
        keep.append(r)
info = {'language': lang, 'pool': str(pool), 'pool_sha256': digest(pool), 'pool_rows': len(rows),
        'pool_hours': sum(float(r['duration']) for r in rows) / 3600, 'excluded_hashes': len(excluded),
        'scanned_files': files, 'scanned_roots': [d.name for d in roots], 'fresh_rows': len(fresh),
        'over_30s': len(long_rows), 'zero_length': len(zero), 'duplicate_audio_dropped': len(fresh) - len(long_rows) - len(zero) - len(keep),
        'selected_rows': len(keep), 'selected_hours': sum(float(r['duration']) for r in keep) / 3600,
        'recordings': len({r['recording_id'] for r in keep})}
print(json.dumps(info))
if dry:
    sys.exit()
if target.exists():
    raise SystemExit('output exists')
target.mkdir()
(target / 'input.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in keep))
info.update(input_sha256=digest(target / 'input.jsonl'), over_30s_ids=long_rows, zero_length_ids=zero)
(target / 'selection.json').write_text(json.dumps(info, indent=2))
