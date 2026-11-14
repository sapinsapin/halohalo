"""Acquire finite public Hiligaynon archive sources; never bypass access controls.

Run: python -m curation.acquire_archives --source deditos
Outputs are private acquisition records, not permission to train or redistribute.
"""
import argparse
import concurrent.futures
import hashlib
import gzip
import re
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent / 'outputs' / 'archive_sources'
PAGE = 'https://deditos.org/datw-hiligaynon-ilonggo'
YTDLP = str(Path.home() / '.local/bin/yt-dlp')
FFPROBE = '/opt/homebrew/bin/ffprobe'
FFMPEG = '/opt/homebrew/bin/ffmpeg'


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def atomic(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


def probe(p):
    r = subprocess.run([FFPROBE, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(p)], capture_output=True, check=True)
    return json.loads(r.stdout)


def free_check():
    if shutil.disk_usage(ROOT).free < 25 * 1024**3:
        raise RuntimeError('disk_floor: fewer than 25 GiB free; upload/verify before resuming')


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'a' and a.get('href'): self.links.append(a['href'])


def acquire_direct(item):
    folder = ROOT / item['source_id']; folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / (item['id'] + '.json')
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get('status') == 'downloaded' and Path(previous['audio_path']).exists() and digest(previous['audio_path']) == previous['audio_sha256']:
            return previous
    item = dict(item, started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    raw = folder / (item['id'] + '.mp4')
    partial = raw.with_suffix('.mp4.part')
    try:
        free_check()
        if not raw.exists():
            req = urllib.request.Request(item['download_url'], headers={'User-Agent': 'SapinSapin dataset acquisition/1.0'})
            with urllib.request.urlopen(req, timeout=60) as response, partial.open('wb') as out:
                expected = response.headers.get('Content-Length')
                for b in iter(lambda: response.read(1024 * 1024), b''):
                    free_check(); out.write(b)
            if expected and partial.stat().st_size != int(expected): raise ValueError('content-length mismatch')
            probe(partial)
            partial.replace(raw)
        metadata = probe(raw)
        audio = folder / (item['id'] + '.flac')
        tmp_audio = folder / (item['id'] + '.tmp.flac')
        # Native-rate mono FLAC retains source bandwidth, unlike 16 kHz training derivatives.
        subprocess.run([FFMPEG, '-nostdin', '-v', 'error', '-y', '-i', str(raw), '-map', '0:a:0', '-vn', '-ac', '1', '-c:a', 'flac', str(tmp_audio)], check=True, timeout=600)
        tmp_audio.replace(audio)
        am = probe(audio)
        item.update(status='downloaded', raw_path=str(raw), raw_sha256=digest(raw), raw_bytes=raw.stat().st_size,
                    audio_path=str(audio), audio_sha256=digest(audio), duration=float(am['format']['duration']),
                    sample_rate=int(am['streams'][0]['sample_rate']), channels=1, bytes=audio.stat().st_size,
                    metadata=metadata)
    except Exception as exc:
        item.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:2000])
    item['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    atomic(manifest, item)
    print(json.dumps({k:item.get(k) for k in ('source_id','id','status','duration','error')}, ensure_ascii=False), flush=True)
    return item


def deditos():
    ROOT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(PAGE, timeout=60) as r: html = r.read()
    (ROOT / 'deditos.html').write_bytes(html)
    parser = Links(); parser.feed(html.decode())
    urls = list(dict.fromkeys(u for u in parser.links if 'progressive_redirect/download/' in u and '/360p/' in u))
    items = []
    for i, url in enumerate(urls, 1):
        items.append(dict(id=f'episode-{i:02d}', source_id='deditos-hiligaynon', page_url=PAGE,
            download_url=url, language='hil', language_status='publisher label, audio not language-verified',
            rights={'training':'unknown','commercial':'unknown','redistribution':'unknown'},
            source_kind='scripted_video', music_expected=True,
            qc_notes='Some episode filenames use other regional-language-looking forms; verify spoken language before counting.',
            asr_status='pending', review_status='pending'))
    atomic(ROOT / 'deditos_inventory.json', {'page_url':PAGE, 'scope':'all published 360p episode download links; not duplicate 1080p encodings or music-only tracks', 'items':items})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire_direct, items))
    atomic(ROOT / 'deditos_summary.json', dict(items=len(results), downloaded=sum(r['status']=='downloaded' for r in results),
        measured_hours=sum(r.get('duration', 0) for r in results if r['status']=='downloaded')/3600,
        language_verified_hours=0, transcribed_hours=0))



def acquire_youtube(item, source_id):
    folder = ROOT / source_id; folder.mkdir(parents=True, exist_ok=True)
    video_id = item['id']
    manifest = folder / (video_id + '.json')
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get('status') == 'downloaded' and Path(old['audio_path']).exists() and digest(old['audio_path']) == old['audio_sha256']:
            return old
    record = dict(id=video_id, source_id=source_id, page_url=item.get('url') or 'https://www.youtube.com/watch?v=' + video_id,
        title=item.get('title'), language='hil', language_status='source/series candidate, audio language unverified',
        rights={'training':'unknown','commercial':'unknown','redistribution':'unknown'},
        asr_status='pending', review_status='pending', source_kind='youtube', discovered=item)
    try:
        free_check()
        if item.get('live_status') in ('is_live', 'is_upcoming'):
            raise ValueError('ongoing or upcoming live video excluded from finite archive acquisition')
        log = folder / (video_id + '.download.log')
        with log.open('w') as out:
            result = subprocess.run([YTDLP, '--no-playlist', '-f', 'bestaudio', '--no-progress', '--socket-timeout', '25',
                '--retries', '2', '--fragment-retries', '2', '--write-info-json', '--match-filter', '!is_live',
                '-o', str(folder / '%(id)s.%(ext)s'), record['page_url']], stdout=out, stderr=subprocess.STDOUT, timeout=3600)
        if result.returncode:
            raise RuntimeError('yt-dlp failed; see ' + str(log))
        info_path = folder / (video_id + '.info.json')
        info = json.loads(info_path.read_text())
        files = [p for p in folder.glob(video_id + '.*') if p.suffix in ('.webm', '.m4a', '.mp3', '.ogg', '.opus', '.mp4')]
        if not files: raise ValueError('no complete audio download')
        audio = files[0]; am = probe(audio)
        record.update(status='downloaded', audio_path=str(audio), audio_sha256=digest(audio),
            bytes=audio.stat().st_size, duration=float(am['format']['duration']),
            sample_rate=int(next(s for s in am['streams'] if s['codec_type']=='audio')['sample_rate']),
            metadata_path=str(info_path), uploader=info.get('uploader'), channel_url=info.get('channel_url'),
            license=info.get('license'), language_reported=info.get('language'))
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:2000])
    record['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    atomic(manifest, record)
    print(json.dumps({k:record.get(k) for k in ('source_id','id','status','duration','error')}), flush=True)
    return record


def youtube(inventory, source_id):
    data = json.loads(Path(inventory).read_text())
    entries = data.get('entries', data.get('items', []))
    # Serial downloads: other source workers may be acquiring concurrently.
    results = []
    for item in entries:
        if item is None: continue
        free_check()
        results.append(acquire_youtube(item, source_id))
        atomic(ROOT / (source_id + '_summary.json'), dict(items=len(entries), processed=len(results),
            downloaded=sum(r['status']=='downloaded' for r in results),
            measured_hours=sum(r.get('duration',0) for r in results if r['status']=='downloaded')/3600,
            language_verified_hours=0, transcribed_hours=0))



def queue_all():
    sources = [('ilonggatalks', 'ilonggatalks_inventory.json'),
        ('bombo-ilonggo-drama', 'bombo_ilonggo_inventory.json'),
        ('gma-ratsada', 'gma_ratsada_inventory.json'),
        ('aksyon-streams', 'aksyon_streams_inventory.json'),
        ('aksyon-videos', 'aksyon_inventory.json'),
        ('rmn-iloilo', 'rmn_iloilo_inventory.json'),
        ('rmn-bacolod', 'rmn_bacolod_inventory.json'),
        ('kuan-joj-jai', 'kuan_joj_jai_inventory.json')]
    queues = []
    for source_id, name in sources:
        path = ROOT / name
        if path.exists():
            data = json.loads(path.read_text())
            entries = [e for e in data.get('entries', []) if e]
            queues.append((source_id, iter(entries)))
    count = 0
    while queues:
        remaining = []
        for source_id, iterator in queues:
            item = next(iterator, None)
            if item is None: continue
            free_check()
            acquire_youtube(item, source_id)
            count += 1
            atomic(ROOT / 'archive_queue_checkpoint.json', {'processed_this_run':count,
                'last_source':source_id, 'last_id':item['id'], 'status':'running',
                'resume':'python -m curation.acquire_archives --source all-youtube; completed content hashes are rechecked and skipped'})
            remaining.append((source_id, iterator))
        queues = remaining
    atomic(ROOT / 'archive_queue_checkpoint.json', {'status':'finished_attempts', 'processed_this_run':count,
        'note':'Per-item failures remain retryable; finished attempts does not mean all downloaded.'})



def discover():
    ROOT.mkdir(parents=True, exist_ok=True)
    # Same public archive endpoint used by the official show's page JavaScript.
    endpoint = 'https://data.igma.tv/regionaltv2023/data/grid/videos/3.gz'
    with urllib.request.urlopen(endpoint, timeout=60) as response:
        data = json.loads(gzip.decompress(response.read()))
    atomic(ROOT / 'gma_grid_inventory.json', data)
    entries, seen = [], set()
    for row in data['data']['details']:
        video_id = row.get('youtube_id')
        if not video_id or video_id in seen: continue
        seen.add(video_id)
        entries.append(dict(id=video_id, url='https://www.youtube.com/watch?v=' + video_id,
            title=row['title'], publisher_record=row,
            overlap_risk='full episodes may contain separately published clips; segment deduplication required'))
    date_title = r'^(?:Ratsada Balita|One Western Visayas):\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d'
    entries.sort(key=lambda e: 0 if re.search(date_title, e['title'], re.I) else 1)
    atomic(ROOT / 'gma_ratsada_inventory.json', dict(source=endpoint,
        scope='all distinct video IDs in official show archive; date-titled full episodes prioritized', entries=entries))
    channels = [('aksyon_inventory', 'https://www.youtube.com/@aksyonradyoiloilo720/videos'),
        ('aksyon_streams_inventory', 'https://www.youtube.com/@aksyonradyoiloilo720/streams'),
        ('saify_inventory', 'https://www.youtube.com/channel/UCswUdogwy9fkqy9vJo4uGHA/videos'),
        ('rmn_iloilo_inventory', 'https://www.youtube.com/channel/UC9JdDk1MSIm6ZtUX4guZ08g/videos'),
        ('rmn_bacolod_inventory', 'https://www.youtube.com/channel/UC01WDmrOkiJkAb1E_1BBqJg/streams')]
    for name, url in channels:
        tmp = ROOT / (name + '.tmp.json')
        with tmp.open('w') as out, (ROOT / (name + '.discovery.log')).open('w') as log:
            result = subprocess.run([YTDLP, '--flat-playlist', '--dump-single-json', '--skip-download',
                '--socket-timeout', '20', '--retries', '1', url], stdout=out, stderr=log)
        if result.returncode:
            atomic(ROOT / (name + '.error.json'), {'source_url':url, 'status':'discovery_failed', 'returncode':result.returncode})
            continue
        data = json.loads(tmp.read_text())
        atomic(ROOT / (name + '.json'), data)
        if name == 'saify_inventory':
            scoped = [e for e in data['entries'] if e and re.search('ilongga.?talk', e.get('title',''), re.I)]
            atomic(ROOT / 'ilonggatalks_inventory.json', dict(scope='only IlonggaTalks title matches on saify TeeVee videos tab', entries=scoped))


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--source', choices=['deditos', 'youtube', 'all-youtube', 'discover'], required=True)
    p.add_argument('--inventory'); p.add_argument('--source-id')
    args = p.parse_args()
    if args.source == 'discover': discover()
    elif args.source == 'deditos': deditos()
    elif args.source == 'all-youtube': queue_all()
    else:
        if not args.inventory or not args.source_id: p.error('youtube needs --inventory and --source-id')
        youtube(args.inventory, args.source_id)

if __name__ == '__main__': main()
