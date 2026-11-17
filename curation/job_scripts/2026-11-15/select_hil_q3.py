"""Reserve a bounded Hiligaynon ASR queue (remote-hil-asr-queue-v3) from the audited remote-hil-vad-v1 candidates.

Round-robin across parent recordings so 2,000 rows span many broadcasts. Excludes any audio hash already present in
any remote-hil-asr-* JSONL. Writes 20 x 100-row batches, worker.json=queued, plan.json. Sends nothing.
Usage: python select_hil_q1.py [--dry-run]
"""
import json, sys
from collections import OrderedDict
from pathlib import Path
from curation.schema import digest

out = Path('/home/iotim/chika-curation/curation/outputs')
src = out / 'remote-hil-vad-v1/vad/manifest.jsonl'
audit = json.loads((out / 'remote-hil-vad-v1/pcm-audit.json').read_text())
root = out / 'remote-hil-asr-queue-v3'
N, B = 2000, 25
MAXDUR = 12.0
if digest(src) != audit['candidate_manifest_sha256']:
    raise SystemExit('candidate manifest does not match its PCM audit')
excluded = set()
for d in out.glob('remote-hil-asr-*'):
    for f in d.rglob('*.jsonl'):
        for line in f.open():
            if '"audio_sha256"' in line:
                excluded.add(json.loads(line)['audio_sha256'])
rows = [json.loads(l) for l in src.open()]
if len(rows) != audit['rows'] or len({r['id'] for r in rows}) != len(rows):
    raise SystemExit('row count or duplicate ids')
seen, groups = set(), OrderedDict()
for r in rows:
    if r['audio_sha256'] in excluded or r['audio_sha256'] in seen or r['duration'] <= 0 or r['duration'] > MAXDUR or r['language'] != 'hil':
        continue
    seen.add(r['audio_sha256'])
    groups.setdefault(r['recording_id'], []).append(r)
sel, i = [], 0
while len(sel) < N and any(groups.values()):
    for k in list(groups):
        if groups[k]:
            sel.append(groups[k].pop(0))
            if len(sel) == N:
                break
info = {'source_manifest_sha256': audit['candidate_manifest_sha256'], 'candidates': len(rows),
        'recordings': len(groups), 'rows': len(sel), 'hours': sum(r['duration'] for r in sel) / 3600,
        'recordings_selected': len({r['recording_id'] for r in sel}), 'excluded_hashes': len(excluded)}
print(json.dumps(info))
if '--dry-run' in sys.argv:
    sys.exit()
if root.exists():
    raise SystemExit('queue root exists')
for r in sel:
    if digest(r['audio_path']) != r['audio_sha256']:
        raise SystemExit('audio hash mismatch ' + r['id'])
root.mkdir()
batches = []
for j in range(0, len(sel), B):
    b = root / f'batch-{j // B + 1:02d}'
    b.mkdir()
    p = b / 'input.jsonl'
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sel[j:j + B]))
    item = {'root': str(b), 'rows': len(sel[j:j + B]), 'input_sha256': digest(p)}
    (b / 'worker.json').write_text(json.dumps({'status': 'queued', 'rows': item['rows'], 'input_sha256': item['input_sha256']}))
    batches.append(item)
(root / 'plan.json').write_text(json.dumps({'version': 1, 'batches': batches}, indent=2))
info.update(plan_sha256=digest(root / 'plan.json'), selected_audio_sha256=[r['audio_sha256'] for r in sel],
            scope='Round-robin across parent recordings of audited remote-hil-vad-v1; never-attempted hashes only. '
                  'Clips <= 12 s only (the 3,000 ms server finalize deadline failed a 27 s clip in v1). Bounded run of the Hiligaynon route in 25-row fresh sessions via curation.asr_quarantine_queue: a finalize_timeout row is quarantined (never replayed) and the queue moves on; any other error stops.')
(root / 'selection.json').write_text(json.dumps(info, indent=1))
print('reserved', len(sel), 'rows in', len(batches), 'batches')
