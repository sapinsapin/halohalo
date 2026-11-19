"""Sample a Chika comparison set: N clips per language from Chika working manifests, spread across sources/recordings.

References are filled in later (Gemini 2.5 Pro, provisional); refs.jsonl starts with reference=null. Clips are 3-27 s,
at most one clip per recording, round-robin across source_id, seeded. Audio is re-hashed and copied as-is.

Usage (dwnd-01, cwd /home/iotim/chika-curation): python3 build_chika_gold.py OUT_DIR [--n 50]
"""
import argparse, hashlib, json, random, shutil
from collections import defaultdict
from pathlib import Path

O = '/home/iotim/chika-curation/curation/outputs/'
POOLS = {'fil': O + 'remote-fil-asr-v6-merge-v1/manifest.jsonl', 'hil': O + 'remote-hil-asr-partial-merge-v1/manifest.jsonl',
         'ceb': O + 'remote-ceb-processing-v4/vad/manifest.jsonl', 'bcl': O + 'remote-bik-processing-v4/vad/manifest.jsonl',
         'ilo': O + 'remote-ilo-processing-v4/vad/manifest.jsonl'}
p = argparse.ArgumentParser()
p.add_argument('out')
p.add_argument('--n', type=int, default=50)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
report = {}
for lang, pool in POOLS.items():
    rows = [json.loads(l) for l in open(pool)]
    rows = [r for r in rows if 3 <= float(r['duration']) <= 27]
    rng = random.Random(f'chika-gold:{lang}')
    by_src = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_src[r['source_id']][r['recording_id']].append(r)
    queues = {}
    for s, recs in by_src.items():
        keys = sorted(recs)
        rng.shuffle(keys)
        queues[s] = [rng.choice(sorted(recs[k], key=lambda x: x['id'])) for k in keys]
    picks, srcs = [], sorted(queues)
    rng.shuffle(srcs)
    while len(picks) < a.n and any(queues.values()):
        for s in srcs:
            if queues[s] and len(picks) < a.n:
                picks.append(queues[s].pop())
    d = out / lang
    d.mkdir()
    items = []
    for r in picks:
        src = Path(r['audio_path'])
        assert hashlib.sha256(src.read_bytes()).hexdigest() == r['audio_sha256']
        name = r['id'][:16] + src.suffix
        shutil.copyfile(src, d / name)
        items.append({'id': r['id'][:16], 'chika_id': r['id'], 'file': f'{lang}/{name}', 'source_id': r['source_id'],
                      'recording_id': r['recording_id'], 'seconds': float(r['duration']), 'pool': pool,
                      'audio_sha256': r['audio_sha256'], 'reference': None,
                      'existing_draft': r.get('sentence') or None})
    (d / 'refs.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items))
    report[lang] = {'pool_rows_3_27s': len(rows), 'sources': len(by_src), 'sampled': len(items),
                    'sources_sampled': len({x['source_id'] for x in items}),
                    'audio_seconds': round(sum(x['seconds'] for x in items), 1)}
    print(lang, report[lang], flush=True)
(out / 'gold.json').write_text(json.dumps(report, indent=2))
