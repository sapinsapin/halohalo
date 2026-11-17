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
import sys
import time
import urllib.request

from .transfer_guard import run_transfer

ROOT = Path(__file__).resolve().parent / 'outputs' / 'archive_sources'
PAGE = 'https://deditos.org/datw-hiligaynon-ilonggo'
def executable(name):
    found = shutil.which(name)
    if found:
        return found
    for directory in (Path(sys.executable).parent, Path.home() / '.local/bin', Path('/opt/homebrew/bin'), Path('/usr/local/bin')):
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    # subprocess raises an actionable FileNotFoundError when that feature is used.
    return name


YTDLP = executable('yt-dlp')
FFPROBE = executable('ffprobe')
FFMPEG = executable('ffmpeg')
BOMBO_PAGE = 'https://www.bomboradyo.com/radio-drama'
YOUTUBE_SOURCES = [('ilonggatalks', 'ilonggatalks_inventory.json'),
    ('bombo-ilonggo-drama', 'bombo_ilonggo_inventory.json'),
    ('gma-ratsada', 'gma_ratsada_inventory.json'),
    ('aksyon-streams', 'aksyon_streams_inventory.json'),
    ('aksyon-videos', 'aksyon_inventory.json'),
    ('rmn-iloilo', 'rmn_iloilo_inventory.json'),
    ('rmn-bacolod', 'rmn_bacolod_inventory.json'),
    ('kuan-joj-jai', 'kuan_joj_jai_inventory.json')]



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


def discover_deditos():
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
    if not items:
        raise ValueError('no Deditos download links found; page structure may have changed')
    return items


def deditos():
    items = discover_deditos()
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
            record.update(status='excluded', exclusion_reason='ongoing_or_upcoming_live',
                retry_policy='rediscover after broadcast ends; no missing live interval can be recovered automatically',
                updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            atomic(manifest, record)
            print(json.dumps({k:record.get(k) for k in ('source_id','id','status','exclusion_reason')}), flush=True)
            return record
        log = folder / (video_id + '.download.log')
        with log.open('w') as out:
            result = run_transfer([YTDLP, '--no-playlist', '-f', 'bestaudio', '--no-progress', '--socket-timeout', '25',
                '--retries', '2', '--fragment-retries', '2', '--write-info-json', '--match-filter', '!is_live',
                '-o', str(folder / '%(id)s.%(ext)s'), record['page_url']], folder, video_id, out, min_free_gib=25)
        if result:
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
    ROOT.mkdir(parents=True, exist_ok=True)
    queues, source_status = [], {}
    for source_id, name in YOUTUBE_SOURCES:
        path = ROOT / name
        try:
            if not path.exists():
                raise FileNotFoundError('inventory missing; run --source discover')
            data = json.loads(path.read_text())
            if not isinstance(data.get('entries'), list):
                raise ValueError('inventory must contain an entries list')
            entries = [e for e in data['entries'] if e]
            if not entries:
                source_status[source_id] = {'status':data.get('status', 'empty_inventory'), 'items':0}
                continue
            error_path = ROOT / (Path(name).stem + '.error.json')
            source_status[source_id] = {'status':'queued', 'items':len(entries), 'processed':0, 'failed':0, 'excluded':0}
            if error_path.exists():
                source_status[source_id]['discovery_warning'] = json.loads(error_path.read_text())
            queues.append((source_id, iter(entries)))
        except Exception as exc:
            source_status[source_id] = {'status':'inventory_failed', 'error_type':type(exc).__name__, 'error':str(exc)}
    count = 0
    checkpoint = ROOT / 'archive_queue_checkpoint.json'
    atomic(checkpoint, {'status':'running', 'processed_this_run':count, 'sources':source_status})
    while queues:
        remaining = []
        for source_id, iterator in queues:
            item = next(iterator, None)
            if item is None:
                source_status[source_id]['status'] = 'finished_attempts'
                continue
            free_check()
            record = acquire_youtube(item, source_id)
            count += 1
            source_status[source_id]['processed'] += 1
            source_status[source_id]['failed'] += int(record['status'] not in {'downloaded', 'excluded'})
            source_status[source_id]['excluded'] += int(record['status'] == 'excluded')
            atomic(checkpoint, {'processed_this_run':count,
                'last_source':source_id, 'last_id':item['id'], 'status':'running', 'sources':source_status,
                'resume':'python -m curation.acquire_archives --source all-youtube; completed content hashes are rechecked and skipped'})
            remaining.append((source_id, iterator))
        queues = remaining
    incomplete = [name for name, status in source_status.items() if
        status['status'] != 'finished_attempts' or status.get('failed') or status.get('discovery_warning')]
    report = {'status':'incomplete' if incomplete else 'finished_attempts', 'processed_this_run':count,
        'sources':source_status, 'incomplete_sources':incomplete,
        'note':'All configured sources are accounted for. Missing/empty/failed discovery and download failures remain incomplete.'}
    atomic(checkpoint, report)
    print(json.dumps(report), flush=True)
    return report


class BomboIlonggoParser(HTMLParser):
    """Restrict extraction to the official ILONGGO DRAMA heading section."""
    def __init__(self):
        super().__init__()
        self.in_heading = False
        self.heading = ''
        self.active = False
        self.found_section = False
        self.current = None
        self.title_depth = 0
        self.entries = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if re.fullmatch('h[1-6]', tag):
            self.in_heading = True
            self.heading = ''
        if self.active and tag == 'a' and attrs.get('data-video-id'):
            self.current = {'id':attrs['data-video-id'], 'title':''}
        if self.current and tag == 'div':
            if self.title_depth:
                self.title_depth += 1
            elif 'td_video_title' in attrs.get('class','').split():
                self.title_depth = 1
    def handle_data(self, data):
        if self.in_heading:
            self.heading += data
        if self.current and self.title_depth:
            self.current['title'] += data
    def handle_endtag(self, tag):
        if re.fullmatch('h[1-6]', tag) and self.in_heading:
            self.in_heading = False
            self.active = self.heading.strip().upper() == 'ILONGGO DRAMA'
            self.found_section = self.found_section or self.active
        if self.current and tag == 'div' and self.title_depth:
            self.title_depth -= 1
        if self.current and tag == 'a':
            self.entries.append(self.current)
            self.current = None
            self.title_depth = 0


def discover_bombo():
    with urllib.request.urlopen(BOMBO_PAGE, timeout=60) as response:
        page = response.read()
    (ROOT / 'bombo-drama.html').write_bytes(page)
    parser = BomboIlonggoParser()
    parser.feed(page.decode())
    if not parser.found_section or not parser.entries:
        raise ValueError('ILONGGO DRAMA section or entries missing; check publisher page structure')
    entries, seen = [], set()
    for row in parser.entries:
        if row['id'] in seen:
            continue
        seen.add(row['id'])
        entries.append(dict(row, url='https://www.youtube.com/watch?v=' + row['id'],
            source_page=BOMBO_PAGE, source_section='ILONGGO DRAMA', candidate_language='hil',
            language_verified=False, rights='unknown'))
    atomic(ROOT / 'bombo_ilonggo_inventory.json', {'source':BOMBO_PAGE,
        'scope':'all IDs in official ILONGGO DRAMA section only', 'entries':entries})


def discover_kuan():
    # IDs verified against the official ABS-CBN Entertainment uploader. Other guests
    # and subscription-only extended scenes are outside the Hiligaynon source scope.
    entries = [
        {'id':'qXRJ3iVFNlE', 'title':'JOJ AND JAI AGPANGAN: Basta Ilongga, gwapa! | KUAN ON ONE Full Episode 11'},
        {'id':'hjK61omcuAQ', 'title':'Joj and Jai open up about their love life | Kuan On One Episode 11 Highlights'}]
    for row in entries:
        row.update(url='https://www.youtube.com/watch?v=' + row['id'],
            channel_id='UCstEtN0pgOmCf02EdXsGChw', overlap_risk='highlight overlaps full episode',
            source_page='https://www.abs-cbn.com/videos/joj-at-jai-may-inamin-tungkol-sa-kanilang-love-life-367588')
    atomic(ROOT / 'kuan_joj_jai_inventory.json', {'scope':'verified official Joj/Jai full episode and highlight seeds; access rechecked on download', 'entries':entries})



def discover_gma():
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


def discover_channels():
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
            error = {'source_url':url, 'status':'discovery_failed', 'returncode':result.returncode}
            atomic(ROOT / (name + '.error.json'), error)
            if name == 'saify_inventory':
                atomic(ROOT / 'ilonggatalks_inventory.error.json', error)
            continue
        data = json.loads(tmp.read_text())
        atomic(ROOT / (name + '.json'), data)
        (ROOT / (name + '.error.json')).unlink(missing_ok=True)
        if name == 'saify_inventory':
            scoped = [e for e in data['entries'] if e and re.search('ilongga.?talk', e.get('title',''), re.I)]
            atomic(ROOT / 'ilonggatalks_inventory.json', dict(scope='only IlonggaTalks title matches on saify TeeVee videos tab', entries=scoped))
            (ROOT / 'ilonggatalks_inventory.error.json').unlink(missing_ok=True)


def discover():
    ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [('deditos_inventory', discover_deditos), ('bombo_ilonggo_inventory', discover_bombo),
        ('kuan_joj_jai_inventory', discover_kuan), ('gma_ratsada_inventory', discover_gma),
        ('channel_inventories', discover_channels)]
    reports = {}
    for name, job in tasks:
        try:
            job()
            (ROOT / (name + '.error.json')).unlink(missing_ok=True)
            reports[name] = {'status':'attempted'}
        except Exception as exc:
            reports[name] = {'status':'discovery_failed', 'error_type':type(exc).__name__, 'error':str(exc)}
            atomic(ROOT / (name + '.error.json'), reports[name])
        atomic(ROOT / 'discovery_status.json', reports)
    return reports


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
