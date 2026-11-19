"""Transcribe a comparison set with the Gemini API (key read from ~/gemini-key, never printed), one clip per request.

Results cache per clip in GOLD/gemini/<model>/<lang>.jsonl with the model version Google reports; reruns skip done
clips. Stops on quota (429) after a short bounded retry instead of looping. Standard library only.

Usage (Mac): python3 eval_gold_gemini_api.py GOLD_DIR --model gemini-pro-latest [--languages fil hil ceb bcl ilo]
"""
import argparse, base64, json, time, urllib.error, urllib.request
from pathlib import Path

NAMES = {'fil': 'Filipino/Tagalog (often mixed with English)', 'hil': 'Hiligaynon (Ilonggo)', 'ceb': 'Cebuano (Bisaya)',
         'bcl': 'Bikol', 'ilo': 'Ilocano'}
MIME = {'.flac': 'audio/flac', '.wav': 'audio/wav', '.mp3': 'audio/mp3'}
PROMPT = ('Transcribe this audio clip verbatim. It is from a Philippine broadcast or podcast, expected language: {name}. '
          'Write exactly the words spoken in the language spoken, including any English code-switching. '
          'Do not translate, summarise, explain, add speaker labels or timestamps. '
          'If there is no intelligible speech, output an empty line. Output only the transcript.')
p = argparse.ArgumentParser()
p.add_argument('gold')
p.add_argument('--model', required=True)
p.add_argument('--languages', nargs='*', default=['fil', 'hil', 'ceb', 'bcl', 'ilo'])
p.add_argument('--min-interval', type=float, default=4.0, help='seconds between requests (free-tier pacing)')
a = p.parse_args()
key = (Path.home() / 'gemini-key').read_text().strip()
gold = Path(a.gold).resolve()
outd = gold / 'gemini' / a.model
outd.mkdir(parents=True, exist_ok=True)
url = f'https://generativelanguage.googleapis.com/v1beta/models/{a.model}:generateContent'
last = 0.0
for lang in a.languages:
    refs = [json.loads(l) for l in (gold / lang / 'refs.jsonl').read_text().splitlines()]
    path = outd / f'{lang}.jsonl'
    done = {json.loads(l)['id'] for l in path.read_text().splitlines()} if path.exists() else set()
    for r in refs:
        if r['id'] in done:
            continue
        f = gold / r['file']
        body = {'contents': [{'parts': [
                    {'inline_data': {'mime_type': MIME[f.suffix], 'data': base64.b64encode(f.read_bytes()).decode()}},
                    {'text': PROMPT.format(name=NAMES[lang])}]}],
                'generationConfig': {'temperature': 0}}
        for attempt in range(4):
            time.sleep(max(0, a.min_interval - (time.monotonic() - last)))
            last = time.monotonic()
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST',
                                         headers={'x-goog-api-key': key, 'content-type': 'application/json'})
            try:
                resp = json.load(urllib.request.urlopen(req, timeout=180))
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors='replace')[:300]
                if e.code in (429, 500, 503) and attempt < 3:
                    time.sleep(30 * (attempt + 1))
                    continue
                raise SystemExit(f'{a.model} HTTP {e.code} at {lang}/{r["id"]}: {msg}')
        cands = resp.get('candidates') or [{}]
        parts = (cands[0].get('content') or {}).get('parts') or []
        hyp = ' '.join(' '.join(x.get('text', '') for x in parts if not x.get('thought')).split())
        with path.open('a') as out:
            out.write(json.dumps({'id': r['id'], 'hypothesis': hyp, 'model_version': resp.get('modelVersion'),
                                  'finish_reason': cands[0].get('finishReason'),
                                  'usage': resp.get('usageMetadata', {})}, ensure_ascii=False) + '\n')
    print(a.model, lang, 'done', flush=True)
