"""Aggregate current ledgers without exposing transcripts, source questions or keys."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent


def snapshot():
    targets=json.loads((ROOT/'targets.json').read_text())['targets']
    reports=[]
    for target in targets:
        label,root=target['namespace'],ROOT/target['rss_root']
        path=root/'state.sqlite'
        if not path.exists(): continue
        db=sqlite3.connect(f'file:{path}?mode=ro',uri=True);db.row_factory=sqlite3.Row
        report={'name':label,'root':str(root.relative_to(ROOT.parent)),
                'groups':[dict(r) for r in db.execute('SELECT source_id,status,count(*) AS files,sum(duration)/3600 AS measured_file_hours,sum(estimated_seconds)/3600 AS feed_estimated_hours FROM items GROUP BY source_id,status')],
                'measured_downloaded_hours':db.execute("SELECT coalesce(sum(duration),0)/3600 FROM items WHERE status='downloaded'").fetchone()[0]}
        wp=next((p for p in [root/'worker.json',root.parent/'worker.json',root.parent/'process.json'] if p.exists()),root/'worker.json')
        if wp.exists():
            worker=json.loads(wp.read_text());pid=worker.get('pid')
            if isinstance(pid,int):
                result=subprocess.run(['ps','-p',str(pid),'-o','pid=,etime=,command='],capture_output=True,text=True)
                report['worker']={'pid':pid,'observation':result.stdout.strip(),'process_present':result.returncode==0}
        receipt=root/'hub-checkpoint.json'
        if receipt.exists():report['hub_checkpoint']=json.loads(receipt.read_text())
        reports.append(report);db.close()
    archives_by_language={}
    for language,root,pattern in [('hil',ROOT/'outputs/archive_sources','*/*.json'),('bik',ROOT/'outputs/bikolano/dzgb','*.json')]:
        archives={}
        for path in root.glob(pattern):
            try:r=json.loads(path.read_text())
            except (ValueError,OSError):continue
            if not isinstance(r,dict) or 'status' not in r or 'source_id' not in r:continue
            group=archives.setdefault(r['source_id'],{'downloaded':0,'failed':0,'measured_file_hours':0})
            if r['status']=='downloaded':
                group['downloaded']+=1;group['measured_file_hours']+=r.get('duration',0)/3600
            elif r['status']=='failed':group['failed']+=1
        archives_by_language[language]=archives
    archives=archives_by_language['hil']
    radio=[]
    for path in (ROOT/'outputs/radio').glob('*/*/status.json'):
        r=json.loads(path.read_text())
        radio.append({k:r.get(k) for k in ['station','state','duration','duration_seconds','runner_pid','ffmpeg_pid','started_at','ended_at','bytes','exclude_from_dataset'] if k in r})
    from .account import account
    inputs=[]
    for language,directories in [('hil',['review-new-hil/joined-manifest.jsonl','vad-asr-batch-v1/hil/asr/manifest.jsonl','collected-hil-vad-200-reviewed-v2/manifest.jsonl']),('fil_taglish',['asr-client-fil/manifest.jsonl','vad-asr-batch-v1/fil/asr/manifest.jsonl','fil-persistent-batch-v1/asr/manifest.jsonl'])]:
        for relative in directories:
            path=ROOT/'outputs'/relative
            if path.exists(): inputs.append((language,path))
    coverage=account(inputs)
    asr_hours={t['group']:coverage['groups'].get(t['group'],{}).get('covered_source_hours',{}).get('draft_transcribed',0) for t in targets}
    batch_path=ROOT/'outputs/vad-asr-batch-v1/worker.json'
    batch=json.loads(batch_path.read_text()) if batch_path.exists() else None
    if batch and isinstance(batch.get('pid'),int):
        process=subprocess.run(['ps','-p',str(batch['pid']),'-o','pid=,etime=,command='],capture_output=True,text=True)
        batch['process_present']=process.returncode==0
        batch['process_observation']=process.stdout.strip()
    archive_backups=[]
    for path in sorted((ROOT/'outputs').glob('archive-backup-*/worker.json')):
        worker=json.loads(path.read_text())
        report={k:worker[k] for k in ('pid','status','started_at','finished_at','expected_recordings','max_bytes','error_type') if k in worker}
        report['root']=str(path.parent.relative_to(ROOT.parent))
        report['local_receipt_files']=len(list((path.parent/'state/receipts').glob('*.json')))
        report['receipt_count_note']='Local receipt files; not an independent remote revalidation or unique-recording count.'
        if isinstance(worker.get('pid'),int):
            process=subprocess.run(['ps','-p',str(worker['pid']),'-o','pid=,etime=,command='],capture_output=True,text=True)
            report['process_present']=process.returncode==0
            report['process_observation']=process.stdout.strip()
        archive_backups.append(report)
    return dict(updated_at=datetime.now(timezone.utc).isoformat(),tracking_issue='https://github.com/sapinsapin/halohalo/issues/8',
                target_retained_transcribed_hours={t['group']:t['hours'] for t in targets},
                priority='acquisition first; segmented ASR retry is bounded and circuit-limited',rss=reports,archives=archives,archives_by_language=archives_by_language,radio=radio,
                new_audio_transcribed_hours=asr_hours,source_coverage=coverage,segmented_asr_batch=batch,archive_backups=archive_backups,
                pld_pilot={'recordings':229,'hours':2.9521978298611113,'reserved_test_recordings':39,'audio_verified':0},
                limitations=['Raw runtime includes ads/music/silence. Source language labels unverified.',
                             'Cross-source and acoustic near-duplicate checks pending. Not unique speech-hour totals.',
                             'Original PLD pilot is separate from newly collected Chika audio.',
                             'ASR figures are draft transport-window runtime, not verified unique speech. Segmentation, new-audio review and final training exports remain incomplete.'])


if __name__=='__main__':
    target=ROOT/'STATUS.json'
    target.write_text(json.dumps(snapshot(),indent=2)+'\n')
    print(target)
