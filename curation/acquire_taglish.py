"""Discover public Taglish RSS sources, then use the shared curation collector.

Discovery inventories advertised duration; it does not count downloaded or
language-verified speech. Raw feed snapshots remain outside Git in outputs/.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

ITUNES = '{http://www.itunes.com/dtds/podcast-1.0.dtd}'


def seconds(value):
    try:
        total = 0.0
        for part in value.split(':'):
            total = total * 60 + float(part)
        return total
    except (AttributeError, ValueError):
        return None


def inspect(source, out):
    record = dict(source)
    try:
        request = urllib.request.Request(source['url'], headers={'User-Agent': 'SapinSapin-Chika-Curation/1.0 (public podcast archive)'})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read(30 * 1024 * 1024 + 1)
            if len(raw) > 30 * 1024 * 1024:
                raise ValueError('feed exceeds 30 MiB limit')
            record['resolved_feed_url'] = response.url
        folder = out / source['id']
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'feed.xml').write_bytes(raw)
        channel = ET.fromstring(raw).find('channel')
        if channel is None:
            raise ValueError('not an RSS channel')
        record.update(feed_sha256=hashlib.sha256(raw).hexdigest(), title=channel.findtext('title'),
                      description=channel.findtext('description'), feed_language=channel.findtext('language'),
                      copyright=channel.findtext('copyright'))
        episodes = []
        for item in channel.findall('item'):
            enclosure = item.find('enclosure')
            if enclosure is None or not enclosure.get('url'):
                continue
            episodes.append(dict(guid=item.findtext('guid'), title=item.findtext('title'),
                                 url=enclosure.get('url'), duration_estimate_s=seconds(item.findtext(ITUNES+'duration')),
                                 enclosure_length=enclosure.get('length'), published=item.findtext('pubDate')))
        record['episode_count'] = len(episodes)
        record['duration_known_count'] = sum(e['duration_estimate_s'] is not None for e in episodes)
        record['advertised_hours'] = sum(e['duration_estimate_s'] or 0 for e in episodes) / 3600
        record['language_status'] = 'source-level candidate; audio language and code-switching unverified'
        record['rights'] = {'training': 'unknown', 'commercial': 'unknown', 'redistribution': 'unknown'}
        (folder/'inventory.json').write_text(json.dumps({'source': record, 'episodes': episodes}, indent=2, ensure_ascii=False))
        print(json.dumps({k:record[k] for k in ['id','title','episode_count','advertised_hours','feed_language']}),flush=True)
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        print(json.dumps({'id':source['id'],'error':record['error']}),flush=True)
    return record


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--registry', default='curation/taglish_sources.json')
    p.add_argument('--output', default='curation/outputs/taglish')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path(a.registry).read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda s: inspect(s,out), sources))
    (out/'inventory_summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print(json.dumps({'sources':len(results),'advertised_hours':sum(r.get('advertised_hours',0) for r in results),
                      'downloaded_hours':0,'note':'Use collector state for actual download accounting.'}),flush=True)


if __name__ == '__main__':
    main()
