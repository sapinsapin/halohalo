"""Prepare bounded ASR windows from acquired RSS originals, without inventing transcripts.

This is transport preparation, not VAD, diarization or final TTS segmentation.
Original media stays intact. Whole-recording canonical PCM hashes expose exact decoded
repeats; acoustic near-duplicates, embedded clips, simulcasts and speech QC remain separate.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import wave

from .schema import VERSION, digest, validate
from .cli import write_jsonl

RATE = 16000


def config(window_seconds):
    if not math.isfinite(window_seconds) or not 1 <= window_seconds <= 60:
        raise ValueError('window seconds must be between 1 and 60')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('ffmpeg required')
    version = subprocess.run([ffmpeg, '-version'], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    return {'version': 1, 'sample_rate': RATE, 'channels': 1, 'sample_format': 's16le',
            'window_samples': int(window_seconds * RATE), 'ffmpeg_version': version,
            'timestamp_kind': 'fixed nonoverlapping ASR transport windows; not speech or word alignment'}


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def connect(root):
    root.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(root/'prepare.sqlite',timeout=30)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE IF NOT EXISTS recordings(
        stage_id TEXT PRIMARY KEY, source_id TEXT, recording_id TEXT, source_sha256 TEXT,
        config_sha256 TEXT, pcm_sha256 TEXT, duration REAL, status TEXT, error TEXT, metadata TEXT);
        CREATE TABLE IF NOT EXISTS windows(id TEXT PRIMARY KEY,stage_id TEXT,row_json TEXT);
    ''')
    return db


def prepare_one(record, root, db, cfg, min_free_gib=25):
    path=Path(record['path'])
    if digest(path)!=record['sha256']:
        raise ValueError('original audio checksum mismatch')
    cfg_hash=identity(cfg)
    stage=identity({'source_id':record['source_id'],'id':record['id'],'bytes':record['sha256'],'config':cfg_hash})
    old=db.execute('SELECT * FROM recordings WHERE stage_id=?',(stage,)).fetchone()
    if old and old['status']=='prepared':
        rows=[json.loads(r[0]) for r in db.execute('SELECT row_json FROM windows WHERE stage_id=?',(stage,))]
        if rows and all(Path(r['audio_path']).is_file() and digest(r['audio_path'])==r['audio_sha256'] for r in rows):
            return {'stage_id':stage,'status':'cached','windows':len(rows),'duration':old['duration']}
    # Budget for uncompressed decode + window output. Never decode hours into RAM.
    expected_bytes=max(0,record.get('duration',0))*RATE*2
    if shutil.disk_usage(root).free < min_free_gib*1024**3 + expected_bytes*2:
        raise OSError('insufficient space for bounded decode and windows')
    db.execute('INSERT OR REPLACE INTO recordings VALUES(?,?,?,?,?,?,?,?,?,?)',
               (stage,record['source_id'],record['id'],record['sha256'],cfg_hash,None,None,'preparing',None,json.dumps(record)))
    db.commit()
    folder=root/'windows'/stage
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    try:
        with tempfile.TemporaryDirectory(prefix='decode-',dir=root) as tmp:
            decoded=Path(tmp)/'canonical.wav'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:a:0','-vn',
                            '-ac','1','-ar',str(RATE),'-c:a','pcm_s16le',str(decoded)],
                           check=True,capture_output=True,timeout=max(120,record.get('duration',0)*2))
            pcm_hash=hashlib.sha256()
            with wave.open(str(decoded),'rb') as wav:
                if wav.getnchannels()!=1 or wav.getsampwidth()!=2 or wav.getframerate()!=RATE:
                    raise ValueError('canonical decode contract mismatch')
                frames=wav.getnframes();duration=frames/RATE
                if record.get('duration') and abs(duration-record['duration'])>max(1,record['duration']*.01):
                    raise ValueError('decoded duration differs materially from acquisition duration')
                start=0
                while start<frames:
                    pcm=wav.readframes(cfg['window_samples'])
                    if not pcm:break
                    count=len(pcm)//2
                    pcm_hash.update(pcm)
                    ident=identity({'stage_id':stage,'start_sample':start,'end_sample':start+count})
                    dest=folder/(ident+'.flac');part=dest.with_suffix('.tmp.flac')
                    if shutil.disk_usage(root).free < min_free_gib*1024**3:
                        raise OSError('storage floor reached')
                    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','s16le','-ar',str(RATE),'-ac','1',
                                    '-i','pipe:0','-c:a','flac',str(part)],input=pcm,check=True,capture_output=True,timeout=60)
                    part.replace(dest)
                    row=dict(schema_version=VERSION,id=ident,source_id=record['source_id'],recording_id=record['id'],
                             audio_path=str(dest.resolve()),audio_sha256=digest(dest),sentence='',prompt=None,
                             text_is_prompt=False,language=record['language'],languages=[record['language']],
                             language_annotation='source candidate; audio language not verified',speech_type='unknown',
                             speaker_id='',duration=count/RATE,num_words=0,split='unassigned',review_status='draft',
                             rights={'training':'unknown','commercial':'unknown','redistribution':'unknown'},
                             provenance={'source_url':record['url'],'original_audio_path':str(path.resolve()),
                                         'original_audio_sha256':record['sha256'],'preparation_config':cfg,
                                         'preparation_config_sha256':cfg_hash,'stage_id':stage,
                                         'start_sample':start,'end_sample':start+count,'offset_sample_rate':RATE},
                             start=start/RATE,end=(start+count)/RATE,asr_status='pending',llm_status='pending',
                             timestamp_kind=cfg['timestamp_kind'],synthetic=False,
                             flags=['speech_presence_unverified','speaker_unknown','language_unverified','alignment_required_for_tts'])
                    rows.append(validate(row));start+=count
                if start!=frames or not rows:
                    raise ValueError('decoded samples were not fully covered')
            pcm_sha=pcm_hash.hexdigest()
            for row in rows:row['provenance']['recording_pcm_sha256']=pcm_sha
            # Window rows become visible only after the complete recording succeeds.
            db.execute('DELETE FROM windows WHERE stage_id=?',(stage,))
            db.executemany('INSERT INTO windows VALUES(?,?,?)',[(r['id'],stage,json.dumps(r,ensure_ascii=False)) for r in rows])
            db.execute("UPDATE recordings SET status='prepared',pcm_sha256=?,duration=?,error=NULL WHERE stage_id=?",(pcm_sha,duration,stage))
            db.commit()
        return {'stage_id':stage,'status':'prepared','windows':len(rows),'duration':duration}
    except Exception as exc:
        db.rollback()
        db.execute("UPDATE recordings SET status='failed',error=? WHERE stage_id=?",(str(exc),stage));db.commit()
        raise


def export_manifest(db,root):
    rows=[json.loads(r[0]) for r in db.execute("SELECT w.row_json FROM windows w JOIN recordings r ON r.stage_id=w.stage_id WHERE r.status='prepared' ORDER BY w.stage_id,w.id")]
    write_jsonl(root/'manifest.jsonl',rows)
    duplicates=[dict(r) for r in db.execute("SELECT pcm_sha256,count(*) AS recordings FROM recordings WHERE status='prepared' GROUP BY pcm_sha256 HAVING count(*)>1")]
    summary={'windows':len(rows),'window_hours':sum(r['duration'] for r in rows)/3600,'transcribed_hours':0,
             'exact_decoded_duplicate_groups':duplicates,
             'notes':'Transport windows only. Empty sentence means pending ASR, never a verified empty transcript. No speech/language/speaker QC or acoustic near-dedup implied.'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--rss-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--max-recordings',type=int,default=2)
    p.add_argument('--window-seconds',type=float,default=30)
    p.add_argument('--min-free-gib',type=float,default=25)
    a=p.parse_args()
    if a.max_recordings < 1 or a.min_free_gib < 0:
        p.error('max-recordings must be positive and min-free-gib nonnegative')
    cfg=config(a.window_seconds);root=Path(a.output);db=connect(root)
    with (root/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config_path=root/'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != cfg:
            raise ValueError('preparation config changed; use a new output root')
        config_path.write_text(json.dumps(cfg,indent=2)+'\n')
        source=sqlite3.connect(f'file:{Path(a.rss_root)/"state.sqlite"}?mode=ro',uri=True);source.row_factory=sqlite3.Row
        processed=0
        for row in source.execute("SELECT i.*,s.language FROM items i JOIN sources s ON s.id=i.source_id WHERE i.status='downloaded' ORDER BY i.updated_at,i.id"):
            record=dict(row)
            result=prepare_one(record,root,db,cfg,a.min_free_gib)
            print(json.dumps(result),flush=True)
            if result['status']!='cached':processed+=1
            if processed>=a.max_recordings:break
        print(json.dumps(export_manifest(db,root)),flush=True)


if __name__=='__main__':main()
