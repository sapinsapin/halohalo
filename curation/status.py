"""Aggregate current ledgers without exposing transcripts, source questions or keys."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent


def snapshot():
    reports=[]
    for label,root in [('hiligaynon-rss',ROOT/'outputs/rss'),('taglish-rss',ROOT/'outputs/taglish/shared')]:
        path=root/'state.sqlite'
        if not path.exists(): continue
        db=sqlite3.connect(f'file:{path}?mode=ro',uri=True);db.row_factory=sqlite3.Row
        report={'name':label,'root':str(root.relative_to(ROOT.parent)),
                'groups':[dict(r) for r in db.execute('SELECT source_id,status,count(*) AS files,sum(duration)/3600 AS measured_file_hours,sum(estimated_seconds)/3600 AS feed_estimated_hours FROM items GROUP BY source_id,status')],
                'measured_downloaded_hours':db.execute("SELECT coalesce(sum(duration),0)/3600 FROM items WHERE status='downloaded'").fetchone()[0]}
        wp=root/'worker.json'
        if wp.exists():
            worker=json.loads(wp.read_text());pid=worker.get('pid')
            if isinstance(pid,int):
                result=subprocess.run(['ps','-p',str(pid),'-o','pid=,etime=,command='],capture_output=True,text=True)
                report['worker']={'pid':pid,'observation':result.stdout.strip(),'process_present':result.returncode==0}
        receipt=root/'hub-checkpoint.json'
        if receipt.exists():report['hub_checkpoint']=json.loads(receipt.read_text())
        reports.append(report);db.close()
    archives={}
    for path in (ROOT/'outputs/archive_sources').glob('*/*.json'):
        try:r=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if not isinstance(r,dict) or 'status' not in r or 'source_id' not in r:continue
        group=archives.setdefault(r['source_id'],{'downloaded':0,'failed':0,'measured_file_hours':0})
        if r['status']=='downloaded':
            group['downloaded']+=1;group['measured_file_hours']+=r.get('duration',0)/3600
        elif r['status']=='failed':group['failed']+=1
    radio=[]
    for path in (ROOT/'outputs/radio').glob('*/*/status.json'):
        r=json.loads(path.read_text())
        radio.append({k:r.get(k) for k in ['station','state','duration','duration_seconds','runner_pid','ffmpeg_pid','started_at','ended_at','bytes','exclude_from_dataset'] if k in r})
    asr_hours = {}
    for language, directory in [('hil', 'asr-client-hil'), ('fil_taglish', 'asr-client-fil')]:
        path = ROOT/'outputs'/directory/'summary.json'
        asr_hours[language] = json.loads(path.read_text()).get('transcribed_hours', 0) if path.exists() else 0
    return dict(updated_at=datetime.now(timezone.utc).isoformat(),tracking_issue='https://github.com/sapinsapin/halohalo/issues/8',
                target_retained_transcribed_hours={'fil_taglish':250,'hil':250},
                priority='acquisition first; ASR bulk deferred after server finalize_timeout',rss=reports,archives=archives,radio=radio,
                new_audio_transcribed_hours=asr_hours,
                pld_pilot={'recordings':229,'hours':2.9521978298611113,'reserved_test_recordings':39,'audio_verified':0},
                limitations=['Raw runtime includes ads/music/silence. Source language labels unverified.',
                             'Cross-source and acoustic near-duplicate checks pending. Not unique speech-hour totals.',
                             'Original PLD pilot is separate from newly collected Chika audio.',
                             'ASR figures are draft transport-window runtime, not verified unique speech. Segmentation, new-audio review and final training exports remain incomplete.'])


if __name__=='__main__':
    target=ROOT/'STATUS.json'
    target.write_text(json.dumps(snapshot(),indent=2)+'\n')
    print(target)
