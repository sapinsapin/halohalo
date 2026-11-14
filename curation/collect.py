"""Resumable RSS acquisition. SQLite is authoritative; source media is never put in Git.

python -m curation.collect discover --sources curation/sources.json --root curation/outputs/rss
python -m curation.collect download --root curation/outputs/rss --max-items 10000
python -m curation.collect status --root curation/outputs/rss
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import time
import uuid
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

AGENT = 'SapinSapin-Chika/0.1 (public podcast archive research)'


def now():
    return datetime.now(timezone.utc).isoformat()


def connect(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / 'state.sqlite', timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, url TEXT, language TEXT,
        metadata TEXT, discovered_at TEXT, error TEXT);
      CREATE TABLE IF NOT EXISTS items(id TEXT PRIMARY KEY, source_id TEXT, url TEXT,
        guid TEXT, title TEXT, published TEXT, estimated_seconds REAL, status TEXT,
        attempts INTEGER DEFAULT 0, path TEXT, sha256 TEXT, duration REAL, bytes INTEGER,
        error TEXT, updated_at TEXT, etag TEXT, last_modified TEXT);
      CREATE TABLE IF NOT EXISTS events(at TEXT, item_id TEXT, stage TEXT, detail TEXT);
    ''')
    return db


def event(db, item_id, stage, detail):
    db.execute('INSERT INTO events VALUES(?,?,?,?)', (now(), item_id, stage, detail))
    db.commit()


def request(url, **headers):
    if not url.startswith(('https://', 'http://')):
        raise ValueError('only public HTTP(S) enclosures supported')
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': AGENT, **headers}), timeout=45)


def seconds(value):
    try:
        total = 0
        for part in (value or '').split(':'):
            total = total * 60 + float(part)
        return total
    except ValueError:
        return None


def discover(args):
    root = Path(args.root)
    db = connect(root)
    for source in json.loads(Path(args.sources).read_text()):
        sid = source['id']
        try:
            with request(source['url']) as response:
                body = response.read(20 * 1024 * 1024 + 1)
                final_url = response.url
            if len(body) > 20 * 1024 * 1024:
                raise ValueError('feed exceeds 20 MiB limit')
            tree = ET.fromstring(body)
            raw = root / 'feeds' / (hashlib.sha256(body).hexdigest() + '.xml')
            raw.parent.mkdir(exist_ok=True)
            raw.write_bytes(body)
            meta = {**source, 'resolved_feed_url': final_url, 'feed_snapshot': str(raw),
                    'feed_sha256': hashlib.sha256(body).hexdigest(),
                    'training_rights': 'unknown', 'redistribution_rights': 'unknown',
                    'language_verified': False}
            db.execute('INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?)',
                       (sid, source['url'], source['language'], json.dumps(meta), now(), None))
            count = 0
            for item in tree.findall('.//item'):
                enc = next((e for e in item.findall('enclosure') if e.get('url')), None)
                if enc is None:
                    continue
                url = enc.get('url')
                guid = item.findtext('guid') or url
                ident = hashlib.sha256((sid + '\0' + guid).encode()).hexdigest()[:32]
                duration = seconds(item.findtext('{http://www.itunes.com/dtds/podcast-1.0.dtd}duration'))
                db.execute('''INSERT INTO items(id,source_id,url,guid,title,published,estimated_seconds,status,updated_at)
                              VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                              url=excluded.url,title=excluded.title,estimated_seconds=excluded.estimated_seconds''',
                           (ident, sid, url, guid, item.findtext('title'), item.findtext('pubDate'), duration, 'pending', now()))
                count += 1
            db.commit()
            print(json.dumps({'source': sid, 'discovered': count}), flush=True)
        except Exception as exc:
            db.execute('INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?)',
                       (sid, source['url'], source['language'], json.dumps(source), now(), str(exc)))
            event(db, sid, 'discovery_error', str(exc))
            print(json.dumps({'source': sid, 'error': str(exc)}), flush=True)
    status(args)


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True, timeout=120)
    info = json.loads(result.stdout)
    if not any(s.get('codec_type') == 'audio' for s in info.get('streams', [])):
        raise ValueError('no audio stream')
    duration = float(info['format']['duration'])
    if not 0 < duration < float('inf'):
        raise ValueError('invalid duration')
    return duration


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(args):
    import fcntl
    root = Path(args.root)
    db = connect(root)
    lock = (root / 'download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A previous process may have died mid-file. The OS lock proves no peer owns this ledger.
    db.execute("UPDATE items SET status='pending' WHERE status='downloading'")
    db.commit()
    for _ in range(args.max_items):
        measured = db.execute("SELECT coalesce(sum(duration),0) FROM items WHERE status='downloaded'").fetchone()[0]
        if args.max_hours and measured >= args.max_hours * 3600:
            break
        if shutil.disk_usage(root).free < args.min_free_gib * 1024**3:
            event(db, None, 'storage_pause', 'free space below configured floor')
            print('Storage floor reached; checkpoint preserved.', flush=True)
            break
        row = db.execute("""SELECT i.* FROM items i WHERE status IN ('pending','failed') AND attempts < ?
                            AND (? = 0 OR (SELECT coalesce(sum(duration),0) FROM items d WHERE d.source_id=i.source_id AND d.status='downloaded') < ?)
                            ORDER BY attempts,(SELECT coalesce(sum(duration),0) FROM items d WHERE d.source_id=i.source_id AND d.status='downloaded'),source_id,published,id LIMIT 1""", (args.max_attempts, args.max_source_hours, args.max_source_hours * 3600)).fetchone()
        if row is None:
            break
        folder = root / 'audio' / row['source_id']
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / (row['id'] + '.media')
        part = target.with_suffix('.part')
        db.execute("UPDATE items SET status='downloading',attempts=attempts+1,updated_at=? WHERE id=?", (now(), row['id']))
        db.commit()
        try:
            # Recover a completed rename whose DB transaction was interrupted.
            if not target.exists():
                offset = part.stat().st_size if part.exists() else 0
                headers = {}
                validator = row['etag'] or row['last_modified']
                if offset and validator:
                    headers = {'Range': f'bytes={offset}-', 'If-Range': validator}
                else:
                    offset = 0
                with request(row['url'], **headers) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('invalid server resume offset')
                    if response.status == 206 and not append:
                        raise ValueError('unexpected partial response')
                    db.execute('UPDATE items SET etag=?,last_modified=? WHERE id=?',
                               (response.headers.get('ETag'), response.headers.get('Last-Modified'), row['id']))
                    db.commit()
                    expected = response.headers.get('Content-Length')
                    received = 0
                    with part.open('ab' if append else 'wb') as f:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            if shutil.disk_usage(root).free < args.min_free_gib * 1024**3:
                                raise OSError('storage floor reached')
                            f.write(block)
                            received += len(block)
                    if expected is not None and received != int(expected):
                        raise ValueError('incomplete HTTP body')
                probe(part)
                part.replace(target)
            duration = probe(target)
            sha = file_hash(target)
            db.execute("""UPDATE items SET status='downloaded',path=?,sha256=?,duration=?,bytes=?,error=NULL,updated_at=?
                          WHERE id=?""", (str(target.resolve()), sha, duration, target.stat().st_size, now(), row['id']))
            event(db, row['id'], 'downloaded', json.dumps({'seconds': duration, 'sha256': sha}))
            print(json.dumps({'id': row['id'], 'source': row['source_id'], 'seconds': duration, 'status': 'downloaded'}), flush=True)
        except Exception as exc:
            # Access denials are not bypassed or retried automatically.
            denied = isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404, 410)
            db.execute('UPDATE items SET status=?,error=?,updated_at=? WHERE id=?',
                       ('unavailable' if denied else 'failed', str(exc), now(), row['id']))
            event(db, row['id'], 'download_error', str(exc))
            print(json.dumps({'id': row['id'], 'error': str(exc)}), flush=True)
            time.sleep(2)
        time.sleep(args.delay)
    lock.close()
    status(args)


def status(args):
    db = connect(args.root)
    records = [dict(r) for r in db.execute('''SELECT s.language,i.source_id,i.status,count(*) AS items,
                round(sum(i.estimated_seconds)/3600,3) AS estimated_hours,
                round(sum(i.duration)/3600,3) AS measured_hours,sum(i.bytes) AS bytes
                FROM items i JOIN sources s ON s.id=i.source_id GROUP BY s.language,i.source_id,i.status''')]
    unique = [dict(r) for r in db.execute('''SELECT language, count(*) AS unique_files, sum(duration)/3600 AS unique_file_hours
        FROM (SELECT s.language,i.sha256,max(i.duration) AS duration FROM items i JOIN sources s ON s.id=i.source_id
        WHERE i.status='downloaded' GROUP BY s.language,i.sha256) GROUP BY language''')]
    summary = {'updated_at': now(), 'groups': records, 'exact_file_deduplicated': unique,
               'notes': 'File runtime includes music/ads/silence. Language unverified. No ASR hours implied. PCM/near-duplicate checks pending.'}
    path = Path(args.root) / 'status.json'
    tmp = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(summary, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=['discover', 'download', 'status'])
    p.add_argument('--root', required=True)
    p.add_argument('--sources')
    p.add_argument('--max-items', type=int, default=10000)
    p.add_argument('--max-hours', type=float, default=0, help='stop after this many measured file hours; 0 means all')
    p.add_argument('--max-source-hours', type=float, default=0, help='per-source measured hours cap; 0 means all')
    p.add_argument('--max-attempts', type=int, default=3)
    p.add_argument('--min-free-gib', type=float, default=25)
    p.add_argument('--delay', type=float, default=1)
    a = p.parse_args()
    if a.command == 'discover' and not a.sources:
        p.error('discover requires --sources')
    globals()[a.command](a)


if __name__ == '__main__':
    main()
