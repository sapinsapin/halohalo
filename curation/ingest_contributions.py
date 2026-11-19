"""Ingest website/MCP contributor deliveries (contributor-staging/v1/<uuid>/{audio.wav,manifest.jsonl}) into a manifest.

Read-only toward the source. Each staged contribution must: hold exactly one manifest row whose id is the folder uuid;
carry `chika-private-v1` private-deposit consent; match its audio bytes to `audio_sha256`; be a canonical 16 kHz mono
16-bit WAV whose measured duration matches; keep `sentence_sha256` consistent; and stay `draft`/`unassigned`. Rows that
fail are listed in the report and left out; nothing is repaired or promoted. Contributor rights are kept as recorded.

    # from a local copy (e.g. `hf download <repo> --include 'contributor-staging/*' --local-dir STAGE`)
    python -m curation.ingest_contributions --staging STAGE/contributor-staging/v1 --output OUT
    # or straight from the Hub at a pinned revision
    python -m curation.ingest_contributions --repo sapinsapin/chika-contributions --revision <sha> --output OUT
"""
import argparse
import hashlib
import json
import shutil
import wave
from pathlib import Path

from .cli import write_jsonl
from .schema import digest, validate

PREFIX = 'contributor-staging/v1'
CONSENT = 'chika-private-v1'


def check(folder):
    """Return (row with local audio_path) for one staged contribution, or raise ValueError with the reason."""
    folder = Path(folder)
    lines = [l for l in (folder / 'manifest.jsonl').read_text().splitlines() if l.strip()]
    if len(lines) != 1:
        raise ValueError('manifest must hold exactly one row')
    row = json.loads(lines[0])
    if row.get('id') != folder.name:
        raise ValueError('manifest id differs from staging folder')
    if row.get('source_id') == 'synthetic-wiring-test':
        raise ValueError('synthetic wiring-test record, not a contribution')
    consent = row.get('consent') or {}
    if consent.get('version') != CONSENT or consent.get('private_deposit') is not True or not consent.get('evidence'):
        raise ValueError('missing private-deposit consent')
    if row.get('review_status') != 'draft' or row.get('split') != 'unassigned':
        raise ValueError('contributions must arrive as unassigned drafts')
    audio = folder / 'audio.wav'
    if row.get('audio_path') != 'audio.wav' or digest(audio) != row.get('audio_sha256'):
        raise ValueError('audio bytes do not match audio_sha256')
    with wave.open(str(audio)) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 16000):
            raise ValueError('audio is not 16 kHz mono 16-bit PCM')
        seconds = w.getnframes() / 16000
    if abs(seconds - float(row['duration'])) > 0.01:
        raise ValueError('recorded duration differs from audio')
    prov = row.get('provenance') or {}
    if prov.get('sentence_sha256') != hashlib.sha256(row['sentence'].encode()).hexdigest():
        raise ValueError('sentence changed after submission')
    row['audio_path'] = str(audio.resolve())
    row['provenance'] = dict(prov, staging_prefix=f'{PREFIX}/{folder.name}', ingest='curation.ingest_contributions')
    return validate(row)


def ingest(staging, output):
    staging, output = Path(staging), Path(output)
    if output.exists():
        raise SystemExit('output exists; outputs are immutable')
    folders = sorted(p for p in staging.iterdir() if p.is_dir()) if staging.exists() else []
    rows, rejected = [], []
    audio_dir = output / 'audio'
    audio_dir.mkdir(parents=True)
    for f in folders:
        try:
            row = check(f)
        except (ValueError, KeyError, OSError, wave.Error, json.JSONDecodeError) as e:
            rejected.append({'id': f.name, 'reason': str(e)})
            continue
        target = audio_dir / f'{row["id"]}.wav'
        shutil.copyfile(row['audio_path'], target)
        if digest(target) != row['audio_sha256']:
            raise RuntimeError(f'copy of {row["id"]} changed bytes')
        row['audio_path'] = str(target.resolve())
        rows.append(validate(row))
    if len({r['audio_sha256'] for r in rows}) != len(rows):
        raise ValueError('the same audio was contributed more than once; resolve before ingesting')
    write_jsonl(output / 'manifest.jsonl', rows)
    report = {'staged': len(folders), 'ingested': len(rows), 'rejected': rejected,
              'hours': sum(r['duration'] for r in rows) / 3600,
              'languages': sorted({r['language'] for r in rows}), 'review_status': 'draft', 'split': 'unassigned'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--staging', help='local contributor-staging/v1 directory')
    p.add_argument('--repo', help='Hub dataset holding contributor-staging/')
    p.add_argument('--revision', help='pinned Hub revision (required with --repo)')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if bool(a.staging) == bool(a.repo):
        p.error('give exactly one of --staging or --repo')
    staging = a.staging
    if a.repo:
        if not a.revision or len(a.revision) != 40:
            p.error('--repo needs a full 40-character --revision')
        from huggingface_hub import snapshot_download
        local = snapshot_download(a.repo, repo_type='dataset', revision=a.revision,
                                  allow_patterns=[f'{PREFIX}/*/audio.wav', f'{PREFIX}/*/manifest.jsonl'])
        staging = Path(local) / PREFIX
    print(json.dumps(ingest(staging, a.output)))


if __name__ == '__main__':
    main()
