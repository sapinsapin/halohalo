"""Score every system's hypotheses on the disjoint PLD gold set with one normaliser (nyo-serve textnorm, as in
eval_batch_asr_gold.py). Systems are directories of <lang>.jsonl files: GOLD/ours, GOLD/gemini/<model>.

Usage: python3 score_gold.py GOLD_DIR  (needs jiwer)
"""
import json, re, sys, unicodedata
from pathlib import Path
import jiwer


def normalise_text(text):  # nyo-serve modules/asr_fw/textnorm.py
    t = unicodedata.normalize("NFD", (text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = unicodedata.normalize("NFC", t).replace("’", "'").replace("‘", "'")
    return " ".join(re.sub(r"[^\w\s']", " ", t).split())


gold = Path(sys.argv[1])
systems = [gold / 'ours'] + sorted((gold / 'gemini').glob('*')) if (gold / 'gemini').exists() else [gold / 'ours']
report = {}
for sysdir in systems:
    name = str(sysdir.relative_to(gold))
    for f in sorted(sysdir.glob('*.jsonl')):
        lang = f.stem
        refs = {r['id']: normalise_text(r['reference']) for r in map(json.loads, (gold / lang / 'refs.jsonl').read_text().splitlines())}
        hyps = {h['id']: normalise_text(h['hypothesis']) for h in map(json.loads, f.read_text().splitlines())}
        ids = [i for i in refs if i in hyps]
        if not ids:
            continue
        r, h = [refs[i] for i in ids], [hyps[i] for i in ids]
        report.setdefault(lang, {})[name] = {'n': len(ids), 'of': len(refs),
                                             'wer_pct': round(100 * jiwer.wer(r, h), 2),
                                             'cer_pct': round(100 * jiwer.cer(r, h), 2)}
(gold / 'scores.json').write_text(json.dumps(report, indent=2))
for lang in sorted(report):
    for name, s in report[lang].items():
        print(f"{lang:4} {name:32} n={s['n']}/{s['of']:<3} WER {s['wer_pct']:6.2f}  CER {s['cer_pct']:6.2f}")
