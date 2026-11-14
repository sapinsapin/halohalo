"""Serial, resumable acquisition of explicit finite YouTube inventories."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import time

from .acquire_archives import YTDLP, atomic, digest, probe
from .transfer_guard import run_transfer


def checked(record):
    return (record.get('status') == 'downloaded' and
            Path(record['audio_path']).is_file() and
            digest(record['audio_path']) == record['audio_sha256'])


def acquire(item, root, language, source_id, provenance, min_free_gib=25, max_attempts=2):
    ident=item['id']
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}',ident):
        raise ValueError('invalid YouTube ID')
    manifest=root/(ident+'.json')
    old=json.loads(manifest.read_text()) if manifest.exists() else {}
    if old and (old['language'] != language or old['source_id'] != source_id):
        raise ValueError('output root belongs to a different language/source')
    if checked(old):return old
    if old.get('attempts',0)>=max_attempts:
        if old.get('status')=='downloaded':
            old.update(status='failed',error='cached audio missing or checksum mismatch; attempts exhausted')
            atomic(manifest,old)
        return old
    record=dict(id=ident,source_id=source_id,language=language,
                language_status='source candidate; audio and regional variety unverified',
                page_url='https://www.youtube.com/watch?v='+ident,title=item.get('title'),
                source_kind='youtube',discovered=item,provenance=provenance,
                rights={'training':'unknown','commercial':'unknown','redistribution':'unknown'},
                asr_status='pending',review_status='pending',attempts=old.get('attempts',0))
    if item.get('live_status') in ('is_live','is_upcoming'):
        record.update(status='excluded',exclusion_reason='ongoing_or_upcoming_live')
        atomic(manifest,record);return record
    if shutil.disk_usage(root).free < min_free_gib*1024**3:
        raise OSError('storage floor reached')
    record.update(status='downloading',attempts=record['attempts']+1)
    atomic(manifest,record)
    try:
        log=root/(ident+'.download.log')
        with log.open('a') as output:
            result=run_transfer([YTDLP,'--no-playlist','-f','bestaudio','--no-progress',
                '--socket-timeout','25','--retries','2','--fragment-retries','2',
                '--concurrent-fragments','1','--write-info-json','--match-filter',
                '!is_live & !is_upcoming','-o',str(root/'%(id)s.%(ext)s'),record['page_url']],
                root,ident,output,min_free_gib=min_free_gib)
        if result:raise RuntimeError('yt-dlp failed; inspect '+str(log))
        info_path=root/(ident+'.info.json');info=json.loads(info_path.read_text())
        if info.get('live_status') in ('is_live','is_upcoming'):
            raise ValueError('live item rejected')
        files=[p for p in root.glob(ident+'.*') if p.suffix in ('.webm','.m4a','.mp3','.ogg','.opus','.mp4')]
        if len(files)!=1:raise ValueError('expected exactly one complete audio file')
        audio=files[0];media=probe(audio);duration=float(media['format']['duration'])
        if not math.isfinite(duration) or duration<=0:raise ValueError('invalid duration')
        stream=next(s for s in media['streams'] if s['codec_type']=='audio')
        record.update(status='downloaded',audio_path=str(audio.resolve()),audio_sha256=digest(audio),
                      bytes=audio.stat().st_size,duration=duration,sample_rate=int(stream['sample_rate']),
                      metadata_path=str(info_path.resolve()),uploader=info.get('uploader'),
                      channel_url=info.get('channel_url'),language_reported=info.get('language'),license=info.get('license'))
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:2000])
    record['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    atomic(manifest,record)
    return record


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('inventory','root','language','source-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--max-hours',type=float,default=300)
    p.add_argument('--min-free-gib',type=float,default=25)
    p.add_argument('--max-items',type=int,default=10000)
    p.add_argument('--max-attempts',type=int,default=2)
    a=p.parse_args()
    if not math.isfinite(a.max_hours) or a.max_hours<=0 or not math.isfinite(a.min_free_gib) or a.min_free_gib<0 or min(a.max_items,a.max_attempts)<1:p.error('invalid bounds')
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    data=json.loads(Path(a.inventory).read_text());items=data.get('items',data.get('entries',[]))
    provenance={k:v for k,v in data.items() if k not in ('items','entries')}
    provenance.update(inventory_path=str(Path(a.inventory).resolve()),inventory_sha256=digest(a.inventory))
    with (root/'download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config_path=root/'config.json'
        config={'language':a.language,'source_id':a.source_id}
        if config_path.exists() and json.loads(config_path.read_text())!=config:
            raise ValueError('output root belongs to a different language/source')
        atomic(config_path,config)
        records=[]
        for item in items:
            path=root/(item['id']+'.json')
            if path.exists():
                r=json.loads(path.read_text())
                if checked(r):records.append(r)
        measured=sum(r['duration'] for r in records);processed=0
        for item in items:
            if measured>=a.max_hours*3600 or processed>=a.max_items:break
            if any(r['id']==item['id'] for r in records):continue
            result=acquire(item,root,a.language,a.source_id,provenance,a.min_free_gib,a.max_attempts)
            processed+=1
            if result['status']=='downloaded':measured+=result['duration'];records.append(result)
            print(json.dumps({k:result.get(k) for k in ('id','status','duration','error')}),flush=True)
            atomic(root/'summary.json',{'measured_hours':measured/3600,'downloaded':len(records),
                'processed_this_run':processed,'language':a.language,'max_hours':a.max_hours,
                'notes':'Raw runtime, not verified speech/language. No ASR. Cap checked before each recording.'})
            time.sleep(2)


if __name__=='__main__':main()
