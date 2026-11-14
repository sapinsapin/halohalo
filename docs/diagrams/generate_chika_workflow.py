"""Regenerate the editable Chika workflow SVG using only the Python standard library."""
import argparse
import shutil
import subprocess
import tempfile
from html import escape
from pathlib import Path

OUT = Path(__file__).with_name('chika-workflow.svg')
parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1200" viewBox="0 0 1500 1200" role="img" aria-labelledby="title description">
<title id="title">Chika: from everyday speech to reusable Philippine-language datasets</title>
<desc id="description">Six-stage bulk workflow: discover sources, acquire originals, prepare speech candidates, transcribe with Boses, review text with an LLM, and publish private PLD-compatible drafts. Manifests, caches, hashes and verified backups support recovery. A separate contributor website and MCP route using ChatGPT or Claude is in development. Training promotion requires audio verification, permission, frozen splits and quality checks. Target: 250 hours for each of five language groups, not current achieved volume.</desc>
<defs>
 <marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#70849b"/></marker>
 <marker id="gold-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#bf8125"/></marker>
 <filter id="shadow" x="-10%" y="-10%" width="120%" height="125%"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#15304e" flood-opacity="0.07"/></filter>
 <style>text{font-family:Arial,Helvetica,sans-serif} .icon{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}</style>
</defs>
<rect width="1500" height="1200" fill="#f4f7fb"/>
<path d="M0 0H1500V220Q1500 240 1480 240H20Q0 240 0 220Z" fill="#101f3c"/>
''']


def rect(x, y, w, h, fill, stroke='none', radius=14, extra=''):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" {extra}/>')


def text(x, y, value, size=20, color='#182c49', weight=400, extra=''):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" {extra}>{escape(value)}</text>')


def line(x1, y1, x2, y2, color='#70849b', arrow=False, extra=''):
    marker = 'marker-end="url(#arrow)"' if arrow else ''
    parts.append(f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="{color}" stroke-width="2" {marker} {extra}/>')



text(64, 47, 'SAPINSAPIN / HALOHALO', 16, '#90b8d5', 700, 'letter-spacing="2.5"')
text(64, 108, 'CHIKA', 57, '#ffffff', 800, 'letter-spacing="1"')
text(285, 105, 'Everyday speech. Reusable datasets.', 35, '#ffffff', 700)
text(64, 151, 'A traceable path from source audio to Philippine-language speech data.', 23, '#c5d6e5')
rect(1160, 34, 276, 135, '#1a3152', '#345170', 18)
text(1182, 81, '1,250 h', 43, '#67e0cd', 700)
text(1182, 111, 'TARGET · 250 h × 5 groups', 17, '#e3eef8', 700)
text(1182, 141, 'Not current achieved volume', 16, '#b9cce0')
x = 64
for name, width in [('Filipino / Taglish',218),('Hiligaynon',164),('Cebuano',145),('Bikolano',147),('Ilocano',137)]:
    rect(x, 183, width, 34, '#1b3455', '#36516e', 17)
    text(x+width/2, 206, name, 17, '#e0eef8', 600, 'text-anchor="middle"')
    x += width+12

text(64, 292, 'THE BULK CURATION WORKFLOW', 23, '#182c49', 700, 'letter-spacing="1"')
text(1436, 291, 'Implemented stages · bounded live validation', 17, '#60758e', 400, 'text-anchor="end"')

icons = [
 '<circle cx="13" cy="13" r="9"/><path d="M20 20L29 29M7 13h3l2-4 3 9 2-5h3"/>',
 '<path d="M16 2v19m-7-7 7 7 7-7M3 22v7h26v-7"/>',
 '<path d="M3 10h26M3 24h26M10 5v10M23 19v10"/><circle cx="10" cy="10" r="3" fill="#fff"/><circle cx="23" cy="24" r="3" fill="#fff"/>',
 '<path d="M4 3h24v19H14l-8 7v-7H4zM10 9v7m6-10v13m6-10v7"/>',
 '<path d="M4 3h17v7M4 3v26h24V17M10 12h6M10 18h4M17 22l2-7 9-9 5 5-9 9z"/>',
 '<ellipse cx="16" cy="6" rx="13" ry="5"/><path d="M3 6v19c0 7 26 7 26 0V6M3 15c0 7 26 7 26 0"/>',
]
cards = [
 ('Discover', '#2389a2', ['RSS & podcasts','Videos & radio','PLD imports'], 'Keep source evidence'),
 ('Acquire', '#2389a2', ['Resumable downloads','Media checks + SHA256','Source + rights ledger'], 'Retries + disk floors'),
 ('Prepare', '#138f86', ['16 kHz mono audio','VAD candidates ≤ 27 s','Exact source offsets'], 'VAD ≠ verified speech'),
 ('Transcribe', '#6373cc', ['Boses ASR','Hil + Fil routes checked','Raw events + cache'], 'Other routes unverified'),
 ('Review', '#6373cc', ['LLM plausibility check','Keep code-switching','Bind review to hashes'], 'Text review ≠ listening'),
 ('Publish', '#138f86', ['PLD-style Parquet','Embedded FLAC + text','Private HF: Chika'], 'Unverified curation set'),
]
for i, (name, color, body, foot) in enumerate(cards):
    x, y = 64+i*232, 328
    rect(x, y, 212, 286, '#ffffff', '#d8e2ed', 16, 'filter="url(#shadow)"')
    rect(x+1, y+1, 210, 6, color, radius=3)
    text(x+20, y+43, f'{i+1:02}', 21, color, 700)
    parts.append(f'<g class="icon" stroke="{color}" transform="translate({x+159} {y+22})">{icons[i]}</g>')
    text(x+18, y+94, name, 28, '#182c49', 700)
    for j, value in enumerate(body): text(x+18, y+136+31*j, value, 17, '#425974')
    line(x+18, y+222, x+194, y+222, '#e5ebf2')
    text(x+106, y+254, foot, 14, color, 600, 'text-anchor="middle"')
    if i<5: line(x+215, y+145, x+230, y+145, arrow=True)

# A cross-cutting reliability layer; it is not a separate data transformation.
rect(64, 650, 1372, 84, '#e4edf6', '#cfddea', 16)
parts.append('<g class="icon" stroke="#3b668d" transform="translate(90 674)"><path d="M16 0L30 6v10c0 9-14 16-14 16S2 25 2 16V6zM9 15l5 5 10-11"/></g>')
text(140, 684, 'A recoverable record at every step', 22, '#234568', 700)
text(140, 714, 'Manifests + caches  ·  source / model hashes  ·  verified backups  ·  interval-based coverage accounting', 18, '#425f7d')

# Promotion is separate from publishing private drafts.
parts.append('<path d="M1330 614V635H1460V763H1190V784" fill="none" stroke="#bf8125" stroke-width="2" marker-end="url(#gold-arrow)"/>')
text(1360, 756, 'promote only after review', 15, '#96641c', 600, 'text-anchor="end"')
rect(944, 790, 492, 300, '#fffaf0', '#e3c58e', 18)
text(970, 822, 'QUALITY GATES · NOT COMPLETE AT SCALE', 14, '#96641c', 700)
text(970, 861, 'From drafts to training', 27, '#182c49', 700)
for j, label in enumerate(['Audio-verified transcript','Training permission + frozen splits','Language / music / overlap / dedup QC']):
    parts.append(f'<circle cx="978" cy="{893+31*j}" r="4" fill="#c49446"/>')
    text(992, 899+31*j, label, 18, '#5a5260')
rect(970, 997, 207, 47, '#e2f2ec', '#bfded2', 10)
rect(1191, 997, 219, 47, '#e2f2ec', '#bfded2', 10)
text(1073.5, 1027, 'ASR · 16 kHz', 22, '#1a6b59', 700, 'text-anchor="middle"')
text(1300.5, 1027, 'TTS · 24 kHz', 22, '#1a6b59', 700, 'text-anchor="middle"')
text(970, 1072, 'Private draft publication does not grant training rights.', 16, '#75644d')

# A separately owned product path. Dashes deliberately distinguish its status.
rect(64, 790, 850, 300, '#f0f2ff', '#98a4de', 18, 'stroke-width="1.5" stroke-dasharray="8 6"')
text(90, 822, 'CONTRIBUTOR ROUTE · SEPARATE WORKSTREAM · IN DEVELOPMENT', 14, '#5966a4', 700)
text(90, 861, 'Contribute through your own assistant', 27, '#182c49', 700)
for x, title, sub in [(90,'Voice website + MCP','Contributor sign-in'),(352,'ChatGPT / Claude','Client-produced transcript'),(614,'Scoped submission','Same private Chika staging')]:
    rect(x, 890, 242, 103, '#ffffff', '#c7cef0', 12)
    text(x+121, 930, title, 21, '#435292', 700, 'text-anchor="middle"')
    text(x+121, 965, sub, 16, '#68749c', 400, 'text-anchor="middle"')
for x in [333,595]:line(x,942,x+17,942,arrow=True)
text(90, 1032, 'Native-client ASR support must be checked, not assumed.', 18, '#556285')
text(90, 1063, 'Hugging Face credentials stay server-side; submissions remain unverified.', 18, '#556285')
text(64, 1140, 'Preserve natural code-switching. Keep questions separate from transcripts.', 23, '#182c49', 700)
text(64, 1175, 'Raw hours ≠ unique speech ≠ transcribed hours ≠ audio-verified hours.', 20, '#62758c')
text(1436, 1175, 'WORKFLOW OVERVIEW · NOV 2026', 13, '#7d8ea3', 600, 'text-anchor="end"')
parts.append('</svg>\n')
OUT.write_text(''.join(parts))
print(OUT)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--png', action='store_true', help='also render a 2x PNG using Chromium')
parser.add_argument('--browser', help='path to a Chromium/Chrome executable')
args = parser.parse_args()
if args.png:
    browser = args.browser or shutil.which('chromium') or shutil.which('chromium-browser') or shutil.which('google-chrome')
    if not browser:
        mac = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        browser = str(mac) if mac.is_file() else None
    if not browser:
        parser.error('--png requires Chromium/Chrome; supply --browser PATH')
    png = OUT.with_suffix('.png').resolve()
    with tempfile.TemporaryDirectory(prefix='chika-diagram-') as profile:
        result = subprocess.run([browser, '--headless=new', '--disable-gpu', '--hide-scrollbars',
            '--no-first-run', '--disable-background-networking', f'--user-data-dir={profile}',
            '--window-size=1500,1200', '--force-device-scale-factor=2',
            f'--screenshot={png}', OUT.resolve().as_uri()], capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors='replace')[-1500:])
    if not png.is_file():
        raise RuntimeError('browser did not produce the PNG')
    print(png)
