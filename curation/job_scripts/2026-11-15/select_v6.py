"""Reserve the next Filipino ASR queue (v6) from the verified deduplicated pool.

Excludes every audio hash that appears in any JSONL under any remote-fil-asr-* root (all queue inputs/outputs,
smoke/batch/continuation attempts including the quarantined uncertain request) and every VAD v1 candidate.
Writes per-batch input.jsonl files and plan.json under remote-fil-asr-queue-v6. Sends nothing.
Usage: python select_v6.py <rows> [--dry-run]
"""
import json, sys
from pathlib import Path
from curation.schema import digest

out = Path('/home/iotim/chika-curation/curation/outputs')
pool = out / 'remote-fil-vad-dedup-v1/deduplicated/manifest.jsonl'
POOL_SHA = '99df3ba812e981e8dfafff1639465dfc8ade3ce6b4755c82548d84d7467cddfd'
root = out / 'remote-fil-asr-queue-v6'
n = int(sys.argv[1])
dry = '--dry-run' in sys.argv
if digest(pool) != POOL_SHA:
    raise SystemExit('pool manifest changed')

excluded, files = set(), 0
for d in sorted(out.glob('remote-fil-asr-*')):
    for f in d.rglob('*.jsonl'):
        files += 1
        for line in f.open():
            if '"audio_sha256"' in line:
                try:
                    excluded.add(json.loads(line)['audio_sha256'])
                except (ValueError, KeyError):
                    pass
for line in (out / 'remote-fil-vad-v1/vad/manifest.jsonl').open():
    excluded.add(json.loads(line)['audio_sha256'])
# the quarantined connection-closed / finalize-timeout requests are recorded by id in attempt files; ids too
rows = [json.loads(l) for l in pool.open()]
remaining = [r for r in rows if r['audio_sha256'] not in excluded and r['duration'] > 0]
print(json.dumps({'pool_rows': len(rows), 'pool_hours': sum(r['duration'] for r in rows) / 3600,
                  'excluded_hashes': len(excluded), 'scanned_files': files, 'remaining_rows': len(remaining),
                  'remaining_hours': sum(r['duration'] for r in remaining) / 3600}))
if dry:
    sys.exit()
if root.exists():
    raise SystemExit('queue v6 root exists')
sel = remaining[:n]
for r in sel:
    if digest(r['audio_path']) != r['audio_sha256']:
        raise SystemExit('audio hash mismatch ' + r['id'])
root.mkdir()
batches = []
for i in range(0, len(sel), 100):
    b = root / f'batch-{i // 100 + 1:03d}'
    b.mkdir()
    p = b / 'input.jsonl'
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sel[i:i + 100]))
    batches.append({'root': str(b), 'rows': len(sel[i:i + 100]), 'input_sha256': digest(p)})
(root / 'plan.json').write_text(json.dumps({'version': 1, 'batches': batches}, indent=2))
(root / 'selection.json').write_text(json.dumps({
    'source_manifest_sha256': POOL_SHA, 'rows': len(sel), 'hours': sum(r['duration'] for r in sel) / 3600,
    'excluded_hashes': len(excluded), 'plan_sha256': digest(root / 'plan.json'),
    'selected_audio_sha256': [r['audio_sha256'] for r in sel],
    'scope': 'Next unattempted rows from the deduplicated pool in pool order; excludes every audio hash in any '
             'remote-fil-asr-* JSONL and VAD v1. Nothing sent by selection.'}, indent=1))
print('reserved', len(sel), 'rows in', len(batches), 'batches')
