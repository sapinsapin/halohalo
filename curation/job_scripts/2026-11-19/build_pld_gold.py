"""Build a 50-clip-per-language ASR gold set from PLD's frozen speaker- and prompt-disjoint test split.

Rebuilds each split spec with halolib.splits.build_spec over the same rows halolib.splits._load_rows uses on the Hub
path (sapinsapin/pld train split, text_is_prompt false), reading only metadata columns from a local snapshot of the train shards. Then samples N test rows per
language with a fixed seed and writes their audio (16 kHz mono FLAC) and references. Read-only on the Hub.
PLD is CC-BY-NC, research-only: outputs stay on private hosts and are not committed.

Usage (dwnd-01, cwd /home/iotim/chika-curation):
  PYTHONPATH=/home/iotim/halohalo python build_pld_gold.py OUT_DIR [--n 50] [--languages bcl ceb fil hil ilo]
"""
import argparse, hashlib, io, json, random
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from halolib.splits import assign, build_spec

p = argparse.ArgumentParser()
p.add_argument('out')
p.add_argument('--n', type=int, default=50)
p.add_argument('--languages', nargs='*', default=['bcl', 'ceb', 'fil', 'hil', 'ilo'])
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
repo = 'sapinsapin/pld'
rev = HfApi().dataset_info(repo).sha
files = sorted(f for f in HfApi().list_repo_files(repo, repo_type='dataset', revision=rev)
               if f.startswith('data/train-') and f.endswith('.parquet'))
snapshot_download(repo, repo_type='dataset', revision=rev, allow_patterns=['data/train-*.parquet'], max_workers=16)
local = lambda f: hf_hub_download(repo, f, repo_type='dataset', revision=rev)
meta_cols = ['speaker_id', 'sentence', 'language', 'text_is_prompt']
rows = []  # (file, row_index_in_file, meta)
for f in files:
    t = pq.read_table(local(f), columns=meta_cols).to_pylist()
    rows += [(f, i, r) for i, r in enumerate(t) if not r['text_is_prompt'] and r['language'] in a.languages]
report = {'repo': repo, 'revision': rev, 'files': len(files), 'n_per_language': a.n, 'languages': {}}
for lang in a.languages:
    lr = [x for x in rows if x[2]['language'] == lang]
    spec = build_spec([x[2] for x in lr], lang)
    test = [x for x in lr if assign(x[2], spec) == 'test']
    # one clip per speaker+prompt pair at most; sample deterministically
    rng = random.Random(f'gold:{lang}')
    pick = rng.sample(test, min(a.n, len(test)))
    d = out / lang
    d.mkdir()
    items = []
    by_file = {}
    for f, i, m in pick:
        by_file.setdefault(f, []).append((i, m))
    for f, sel in sorted(by_file.items()):
        pf = pq.ParquetFile(local(f))
        tab = pf.read(columns=['audio', 'sentence', 'speaker_id'])
        for i, m in sel:
            row = tab.slice(i, 1).to_pylist()[0]
            assert row['sentence'] == m['sentence'] and row['speaker_id'] == m['speaker_id']
            audio, sr = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert sr == 16000, sr
            cid = hashlib.sha256(f'{f}:{i}'.encode()).hexdigest()[:16]
            path = d / f'{cid}.flac'
            sf.write(path, audio, 16000, format='FLAC')
            items.append({'id': cid, 'file': f'{lang}/{cid}.flac', 'source': f'{f}#{i}', 'speaker_id': m['speaker_id'],
                          'reference': m['sentence'], 'seconds': round(len(audio) / 16000, 3),
                          'flac_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    items.sort(key=lambda x: x['id'])
    (d / 'refs.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items))
    report['languages'][lang] = {'split_stats': spec['stats'], 'test_rows': len(test), 'sampled': len(items),
                                 'audio_seconds': round(sum(x['seconds'] for x in items), 2)}
    print(lang, report['languages'][lang], flush=True)
(out / 'gold.json').write_text(json.dumps(report, indent=2))
