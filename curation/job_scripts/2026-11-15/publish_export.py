"""Publish an export privately, register its card config, then stream-verify every row.

Usage (Mac, repo root, PYTHONPATH=.): python publish_export.py WORKDIR CONFIG ROWS PARAGRAPH_FILE [MAX_FILES=100]
WORKDIR holds export/ (copied and re-hashed beforehand); receipts are written beside it.
Same three stages as dwnd-fil-2000-publish-v1 (publish -> card -> stream check), in one worker.
"""
import hashlib, json, os, sys, time
from pathlib import Path
import pyarrow.parquet as pq
from datasets import Audio, load_dataset
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from curation.publish import publish

root = Path(sys.argv[1]).resolve()
repo, config, rows_expected = 'sapinsapin/chika', sys.argv[2], int(sys.argv[3])
paragraph = Path(sys.argv[4]).read_text().strip() + '\n\n'
max_files = int(sys.argv[5]) if len(sys.argv) > 5 else 100  # 100 shards + README, summary, manifest needs 110
assert config in paragraph
s = {'pid': os.getpid(), 'status': 'running', 'stage': 'publish', 'started_at': time.time()}


def save():
    p = root / 'worker.tmp'
    p.write_text(json.dumps(s))
    p.replace(root / 'worker.json')


save()
try:
    r = publish(root / 'export', 'data/' + config, root / 'hub-receipt.json', max_files=max_files)
    receipt = json.loads((root / 'hub-receipt.json').read_text())
    assert receipt['verified'] and receipt['private'] and receipt['rows'] == rows_expected and receipt['prefix'] == 'data/' + config
    s.update(stage='registering_card', data_revision=r['revision'])
    save()
    api = HfApi()
    info = api.dataset_info(repo)
    assert info.private
    parent = info.sha
    card = Path(hf_hub_download(repo, 'README.md', repo_type='dataset', revision=parent)).read_text()
    assert config not in card and card.count('configs:\n') == 1 and card.count('## Available configuration\n\n') == 1
    entry = f'- config_name: {config}\n  data_files:\n  - split: unassigned\n    path: data/{config}/curation-unassigned-*.parquet\n'
    card = card.replace('configs:\n', 'configs:\n' + entry, 1)
    card = card.replace('## Available configuration\n\n', '## Available configuration\n\n' + paragraph, 1)
    commit = api.create_commit(repo_id=repo, repo_type='dataset', parent_commit=parent,
                               operations=[CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card.encode())],
                               commit_message=f'Register private curation snapshot {config}')
    (root / 'card-receipt.json').write_text(json.dumps({'revision': commit.oid, 'parent': parent, 'config': config}, indent=2))
    assert Path(hf_hub_download(repo, 'README.md', repo_type='dataset', revision=commit.oid, force_download=True)).read_text() == card
    s['stage'] = 'stream_verification'
    save()
    expected = {}
    for p in sorted((root / 'export').glob('*.parquet')):
        for row in pq.read_table(p).to_pylist():
            assert row['id'] not in expected
            expected[row['id']] = {'sentence': row['sentence'], 'review_status': row['review_status'], 'split': row['split'],
                                   'audio_sha256': hashlib.sha256(row['audio']['bytes']).hexdigest(),
                                   'provenance_json': row['provenance_json']}
    seen = set()
    for row in load_dataset(repo, config, revision=commit.oid, split='unassigned', streaming=True).cast_column('audio', Audio(decode=False)):
        assert row['id'] not in seen and row['id'] in expected
        e = expected[row['id']]
        for k in ('sentence', 'review_status', 'split', 'provenance_json'):
            assert row[k] == e[k]
        assert hashlib.sha256(row['audio']['bytes']).hexdigest() == e['audio_sha256']
        seen.add(row['id'])
    assert seen == set(expected) and len(seen) == rows_expected
    (root / 'stream-check.json').write_text(json.dumps({'revision': commit.oid, 'config': config, 'rows': len(seen),
                                                         'exact_ids_text_provenance_audio_bytes': True,
                                                         'training_approved': False}, indent=2))
    s.update(status='complete', stage='complete', card_revision=commit.oid, finished_at=time.time())
    save()
except BaseException as e:
    s.update(status='stopped_for_inspection', error=f'{type(e).__name__}: {e}'[:500], finished_at=time.time())
    save()
    raise
