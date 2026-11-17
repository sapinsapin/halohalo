"""Freeze an entirely eligible Filipino manifest into MOSS waveform conversations."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

from .schema import digest, load_manifest
from .tts_readiness import audit


def export(manifest, recipe_path, output):
    manifest, recipe_path, output = map(Path, (manifest, recipe_path, output))
    output = output.resolve()
    if output.exists():
        raise ValueError('refusing to overwrite an existing export')
    recipe = json.loads(recipe_path.read_text())
    report = audit(manifest, recipe)
    if not report['data_gate_passed'] or report['excluded_rows']:
        raise ValueError('export requires a frozen manifest containing only eligible rows and sufficient hours')
    rows = load_manifest(manifest)
    if digest(manifest) != report['manifest_sha256']:
        raise ValueError('manifest changed during audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.tts-export-', dir=output.parent))
    try:
        (stage / 'audio').mkdir()
        splits = {name: [] for name in ('train', 'validation', 'test')}
        seen = set()
        for index, row in enumerate(rows):
            source = Path(row['audio_path'])
            if digest(source) != row['audio_sha256']:
                raise ValueError('source audio changed before export')
            relative = Path('audio') / f'{index:08d}.wav'
            target = stage / relative
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
                            '-map', '0:a:0', '-ac', '1', '-ar', '24000',
                            '-c:a', 'pcm_s16le', str(target)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if digest(source) != row['audio_sha256']:
                raise ValueError('source audio changed during export')
            with wave.open(str(target)) as wav:
                if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1, 24000, 2):
                    raise ValueError('unexpected exported waveform format')
                duration = wav.getnframes() / 24000
            if abs(duration - row['duration']) > .05:
                raise ValueError('decoded duration differs from approved metadata')
            sha = digest(target)
            if sha in seen:
                raise ValueError('duplicate audio after waveform normalization')
            seen.add(sha)
            split = 'test' if row['split'] == 'reserved_test' else row['split']
            splits[split].append({'id': row['id'], 'conversations': [
                {'role': 'assistant', 'text': row['sentence'], 'wav': str(output / relative)}],
                'provenance': {'canonical_row': row, 'audio_sha256': sha,
                               'duration_seconds': duration, 'split': split,
                               'derivation': 'ffmpeg mono 24000 Hz PCM16; text unchanged'}})
        for split, records in splits.items():
            (stage / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
        shutil.copyfile(manifest, stage / 'canonical.jsonl')
        shutil.copyfile(recipe_path, stage / 'recipe.json')
        if digest(stage / 'canonical.jsonl') != report['manifest_sha256']:
            raise ValueError('manifest changed before freeze')
        report.update(counts={k: len(v) for k, v in splits.items()},
                      recipe_sha256=digest(stage / 'recipe.json'),
                      manifests_sha256={k: digest(stage / f'{k}.jsonl') for k in splits},
                      ffmpeg_version=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
                      codec_encoded=False, training_launched=False)
        (stage / 'export_report.json').write_text(json.dumps(report, indent=2)+'\n')
        stage.rename(output)
        return report
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--recipe', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(export(args.manifest, args.recipe, args.output)))


if __name__ == '__main__':
    main()
