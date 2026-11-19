"""Transcribe the disjoint PLD gold set with Gemini CLI (signed in with the user's Google account), one clip per call.

Each call runs `gemini -m MODEL -p PROMPT` from inside the language directory with the clip referenced as @file, so the
CLI attaches the audio. Responses are cached per clip in OUT/<model>/<lang>.jsonl; reruns skip finished clips.
Stops on quota errors instead of retrying in a loop.

Usage (Mac): python3 eval_gold_gemini.py GOLD_DIR [--model gemini-2.5-flash] [--languages fil hil ceb bcl ilo]
"""
import argparse, json, subprocess, time
from pathlib import Path

NAMES = {'fil': 'Filipino (Tagalog)', 'hil': 'Hiligaynon (Ilonggo)', 'ceb': 'Cebuano (Bisaya)',
         'bcl': 'Central Bikol', 'ilo': 'Ilocano'}
PROMPT = ('Transcribe the speech in @{file} verbatim. The speaker reads one sentence in {name}. '
          'Write exactly the words spoken, in the original language; do not translate, explain or add anything. '
          'Output only the transcript on one line.')
p = argparse.ArgumentParser()
p.add_argument('gold')
p.add_argument('--model', default='gemini-2.5-flash')
p.add_argument('--languages', nargs='*', default=['fil', 'hil', 'ceb', 'bcl', 'ilo'])
a = p.parse_args()
gold = Path(a.gold).resolve()
outd = gold / 'gemini' / a.model
outd.mkdir(parents=True, exist_ok=True)
for lang in a.languages:
    refs = [json.loads(l) for l in (gold / lang / 'refs.jsonl').read_text().splitlines()]
    path = outd / f'{lang}.jsonl'
    done = {json.loads(l)['id'] for l in path.read_text().splitlines()} if path.exists() else set()
    for r in refs:
        if r['id'] in done:
            continue
        name = Path(r['file']).name
        t = time.monotonic()
        res = subprocess.run(['gemini', '-m', a.model, '-p', PROMPT.format(file=name, name=NAMES[lang])],
                             cwd=gold / lang, capture_output=True, text=True, timeout=180)
        err = (res.stderr or '')[-400:]
        if res.returncode != 0:
            if any(k in err.lower() for k in ('quota', '429', 'rate limit', 'resource_exhausted')):
                raise SystemExit(f'quota reached at {lang}/{r["id"]}: {err}')
            raise SystemExit(f'gemini failed on {lang}/{r["id"]}: {err}')
        hyp = ' '.join(res.stdout.strip().splitlines()).strip()
        with path.open('a') as f:
            f.write(json.dumps({'id': r['id'], 'hypothesis': hyp, 'seconds': round(time.monotonic() - t, 2)},
                               ensure_ascii=False) + '\n')
    print(lang, 'done', flush=True)
