"""Run with python -m curation.cli. All outputs local; no automatic Hub upload."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

from .schema import VERSION, check_splits, digest, load_manifest, validate


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def import_pld(args):
    rows = []
    for path in sorted(Path(args.drafts).glob('*.json')):
        d = json.loads(path.read_text())
        if not isinstance(d, dict) or 'transcript' not in d:
            continue
        row = dict(schema_version=VERSION, id='pld:' + d['id'], source_id='sapinsapin/pld',
                   recording_id=d['key'], audio_path=d['audio'], audio_sha256=d['audio_sha256'],
                   sentence=d['transcript'], prompt=d['question'], text_is_prompt=False,
                   language=d['language'], languages=[d['language']],
                   language_annotation='source label; code-switching not exhaustively annotated',
                   speech_type='spontaneous', speaker_id=d['speaker_id'], duration=d['duration'],
                   num_words=len(d['transcript'].split()), split=d['split'], review_status='draft',
                   rights=dict(training='unknown', commercial='unknown', redistribution='unknown',
                               license='source restrictions apply', evidence=d.get('license_scope', '')),
                   provenance=dict(source_draft=str(path.resolve()), source_draft_sha256=digest(path),
                                   source_shard=d['source_shard'], row_group=d['row_group'], row_index=d['row_index'],
                                   split_spec_sha256=d['split_spec_sha256'], asr_config_sha256=d['config_sha256']),
                   segments=d['segments'], timestamp_kind=d['timestamp_kind'],
                   flags=d.get('flags', []), synthetic=False)
        if args.reviews:
            rp = Path(args.reviews) / (d['id'] + '.json')
            if rp.exists():
                review = json.loads(rp.read_text())
                if review['source_draft_sha256'] != hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest():
                    raise ValueError(f'stale text review: {d["id"]}')
                row['text_review'] = review
                row['review_status'] = 'text_reviewed'
        rows.append(validate(row))
    if not rows:
        raise ValueError('no PLD drafts found')
    check_splits(rows)
    write_jsonl(args.output, rows)
    print(json.dumps({'rows': len(rows), 'hours': sum(r['duration'] for r in rows) / 3600,
                      'reserved_test': sum(r['split'] == 'reserved_test' for r in rows), 'output': args.output}))


def export(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    rows = load_manifest(args.manifest)
    check_splits(rows)
    if args.profile != 'curation':
        for row in rows:
            if not row['sentence'].strip():
                raise ValueError(f'{row["id"]}: training export needs a nonempty verified transcript')
            if row['review_status'] != 'audio_verified' or row['rights']['training'] != 'allowed':
                raise ValueError(f'{row["id"]}: training export needs audio verification and training permission')
            if row['split'] == 'unassigned':
                raise ValueError('training export requires frozen splits')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('export directory must be empty (immutable exports)')
    # Validate all inputs before publishing any shards.
    for row in rows:
        path = Path(args.audio_root) / Path(row['audio_path']).name if args.audio_root else Path(row['audio_path'])
        if digest(path) != row['audio_sha256']:
            raise ValueError(f'audio hash mismatch: {row["id"]}')
    out.mkdir(parents=True, exist_ok=True)
    rate = 24000 if args.profile == 'tts' else 16000
    schema = pa.schema([
        ('audio', pa.struct([('bytes', pa.binary()), ('path', pa.string())])),
        ('sentence', pa.string()), ('duration', pa.float64()), ('speaker_id', pa.string()),
        ('speech_type', pa.string()), ('language', pa.string()), ('num_words', pa.int64()),
        ('id', pa.string()), ('split', pa.string()), ('review_status', pa.string()),
        ('schema_version', pa.string()), ('provenance_json', pa.string()),
        ('text_is_prompt', pa.bool_()), ('prompt', pa.string()),
        ('source_id', pa.string()), ('recording_id', pa.string()),
    ])
    hf = {'info': {'features': {k: {'dtype': str(t), '_type': 'Value'} for k, t in [
        ('sentence', 'string'), ('duration', 'float64'), ('speaker_id', 'string'),
        ('speech_type', 'string'), ('language', 'string'), ('num_words', 'int64'),
        ('id', 'string'), ('split', 'string'), ('review_status', 'string'),
        ('schema_version', 'string'), ('provenance_json', 'string'),
        ('text_is_prompt', 'bool'), ('prompt', 'string'), ('source_id', 'string'), ('recording_id', 'string')]}}}
    hf['info']['features']['audio'] = {'sampling_rate': rate, '_type': 'Audio'}
    schema = schema.with_metadata({b'huggingface': json.dumps(hf).encode()})
    artifacts = []
    for split in sorted({r['split'] for r in rows}):
        group = [r for r in rows if r['split'] == split]
        n = math.ceil(len(group) / args.shard_size)
        for i in range(n):
            batch = []
            for row in group[i * args.shard_size:(i + 1) * args.shard_size]:
                path = Path(args.audio_root) / Path(row['audio_path']).name if args.audio_root else Path(row['audio_path'])
                audio, sr = sf.read(path, dtype='float32', always_2d=True)
                if abs(len(audio) / sr - row['duration']) > .1:
                    raise ValueError(f'duration mismatch: {row["id"]}')
                mono = audio.mean(axis=1)
                if sr != rate:
                    g = math.gcd(sr, rate)
                    mono = resample_poly(mono, rate // g, sr // g)
                buffer = io.BytesIO()
                sf.write(buffer, mono, rate, format='FLAC', subtype='PCM_16')
                r = {k: row[k] for k in schema.names if k not in {'audio', 'provenance_json'}}
                r['audio'] = {'bytes': buffer.getvalue(), 'path': None}
                r['provenance_json'] = json.dumps({**row, 'export_sample_rate': rate, 'original_sample_rate': sr}, ensure_ascii=False)
                batch.append(r)
            name = f'{args.profile}-{split}-{i:05d}-of-{n:05d}.parquet'
            target = out / name
            tmp = out / (name + '.tmp')
            pq.write_table(pa.Table.from_pylist(batch, schema=schema), tmp)
            tmp.replace(target)
            artifacts.append({'path': name, 'sha256': digest(target), 'rows': len(batch)})
    write_jsonl(out / 'manifest.jsonl', rows)
    summary = dict(schema_version=VERSION, profile=args.profile, rows=len(rows),
                   hours=sum(r['duration'] for r in rows) / 3600, sample_rate=rate,
                   input_manifest_sha256=digest(args.manifest), artifacts=artifacts)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'README.md').write_text('# SapinSapin speech export\n\n'
        + f'Profile: {args.profile}. {len(rows)} rows. Sample rate: {rate} Hz.\n\n'
        + 'Curation rows are unverified drafts, not training-approved labels. Source questions remain in provenance, never in sentence. '
        + 'Hiligaynon–English code-switching is valid. Rights are recorded per row; this export grants no new rights. '
        + 'Reserved test rows must never be used for training. See manifest.jsonl for original metadata and source hashes.\n')
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('import-pld')
    p.add_argument('--drafts', required=True)
    p.add_argument('--reviews')
    p.add_argument('--output', required=True)
    p.set_defaults(fn=import_pld)
    p = commands.add_parser('validate')
    p.add_argument('manifest')
    def validate_manifest(a):
        rows = load_manifest(a.manifest)
        check_splits(rows)
        print(json.dumps({'valid_rows': len(rows)}))
    p.set_defaults(fn=validate_manifest)
    p = commands.add_parser('export')
    p.add_argument('--manifest', required=True)
    p.add_argument('--audio-root')
    p.add_argument('--output', required=True)
    p.add_argument('--profile', choices=['curation', 'asr', 'tts'], default='curation')
    p.add_argument('--shard-size', type=int, default=100)
    p.set_defaults(fn=export)
    args = parser.parse_args()
    if hasattr(args, 'shard_size') and args.shard_size < 1:
        parser.error('--shard-size must be positive')
    args.fn(args)


if __name__ == '__main__':
    main()
