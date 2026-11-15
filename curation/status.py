"""Aggregate current ledgers without exposing transcripts, source questions or keys."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent


def job_observation(path, summary_path=None):
    """Report allowlisted state plus an actual process observation, not a PID claim."""
    worker=json.loads(path.read_text())
    report={k:worker[k] for k in ('pid','status','started_at','finished_at','returncode','rows','cached_items','max_new_items','stage','completed_recordings') if k in worker}
    report['root']=str(path.parent.relative_to(ROOT.parent))
    if isinstance(worker.get('pid'),int):
        process=subprocess.run(['ps','-p',str(worker['pid']),'-o','pid=,etime=,command='],capture_output=True,text=True)
        report['process_present']=process.returncode==0
        report['process_matches_job']=process.returncode==0 and str(path.parent.name) in process.stdout
        report['process_observation']=process.stdout.strip()
    if summary_path and summary_path.exists():
        summary=json.loads(summary_path.read_text())
        report['summary']={k:(len(summary[k]) if isinstance(summary[k],list) else summary[k])
                           for k in ('selected','completed','failed','stopped_early','transcribed_hours','empty','manifest_rows','checkpoint_every') if k in summary}
    return report


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
    for language,root,pattern in [('hil',ROOT/'outputs/archive_sources','*/*.json'),('bik',ROOT/'outputs/bikolano/dzgb','*.json'),('ceb',ROOT/'outputs/cebuano/archive-expansion/audio','*.json'),('ilo',ROOT/'outputs/ilocano/archive-expansion/media','*/*.json')]:
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
    for path in [* (ROOT/'outputs/radio').glob('*/*/status.json'), * (ROOT/'outputs/ilocano/radio').glob('*/*/status.json')]:
        r=json.loads(path.read_text())
        radio.append({k:r.get(k) for k in ['station','candidate_language','state','duration','duration_seconds','runner_pid','ffmpeg_pid','started_at','ended_at','bytes','exclude_from_dataset'] if k in r})
    from .account import account
    inputs=[]
    for language,directories in [('hil',['review-new-hil/joined-manifest.jsonl','vad-asr-batch-v1/hil/asr/manifest.jsonl','collected-hil-vad-200-reviewed-v2/manifest.jsonl','collected-hil-667-v1/manifest.jsonl','collected-hil-667-v2/manifest.jsonl','collected-hil-718-v1/manifest.jsonl','hil-persistent-new-10-v1/asr/manifest.jsonl','hil-persistent-new-batch-v1/asr/manifest.jsonl']),('fil_taglish',['asr-client-fil/manifest.jsonl','vad-asr-batch-v1/fil/asr/manifest.jsonl','fil-persistent-batch-v1/asr/manifest.jsonl','fil-reconciled-14-v1/manifest.jsonl','fil-persistent-batch-v2/asr/manifest.jsonl','fil-persistent-batch-v3/asr/manifest.jsonl','fil-persistent-batch-v4/asr/manifest.jsonl','fil-persistent-batch-v5/asr/manifest.jsonl','fil-persistent-batch-v6/asr/manifest.jsonl','collected-fil-482-v1/manifest.jsonl','collected-fil-982-v1/manifest.jsonl'])]:
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
    preparation_root=ROOT/'outputs/five-language-preparation-v1'
    preparation_path=preparation_root/'worker.json'
    preparation=json.loads(preparation_path.read_text()) if preparation_path.exists() else None
    if preparation:
        if isinstance(preparation.get('pid'),int):
            process=subprocess.run(['ps','-p',str(preparation['pid']),'-o','pid=,etime=,command='],capture_output=True,text=True)
            preparation['process_present']=process.returncode==0
            preparation['process_observation']=process.stdout.strip()
        preparation['summaries']={}
        for target in targets:
            group=target['group']
            for stage in ('prepared','vad'):
                summary=preparation_root/group/stage/'summary.json'
                if summary.exists():
                    preparation['summaries'].setdefault(group,{})[stage]=json.loads(summary.read_text())
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
    persistent_batches=[job_observation(path,path.parent/'asr/summary.json')
                        for pattern in ('fil-persistent-batch-v*/worker.json','hil-persistent-new-*/worker.json')
                        for path in sorted((ROOT/'outputs').glob(pattern))]
    review_batches=[job_observation(path,path.parent/'reviews/summary.json')
                    for pattern in ('fil-review-*/worker.json','collected-hil-remaining-*/worker.json','collected-hil-new-*-review-*/worker.json')
                    for path in sorted((ROOT/'outputs').glob(pattern))]
    storage_jobs=[job_observation(path) for path in sorted((ROOT/'outputs').glob('*-preemptive-backup-*/worker.json'))]
    asr_queues=[job_observation(path) for path in sorted((ROOT/'outputs').glob('*-asr-queue-*/worker.json'))]
    preparation_batches=[job_observation(path) for path in sorted((ROOT/'outputs').glob('*-preparation-next-*/worker.json'))]
    return dict(updated_at=datetime.now(timezone.utc).isoformat(),tracking_issue='https://github.com/sapinsapin/halohalo/issues/8',
                target_retained_transcribed_hours={t['group']:t['hours'] for t in targets},
                priority='acquisition first; segmented ASR retry is bounded and circuit-limited',rss=reports,archives=archives,archives_by_language=archives_by_language,radio=radio,
                new_audio_transcribed_hours=asr_hours,source_coverage=coverage,segmented_asr_batch=batch,archive_backups=archive_backups,candidate_preparation=preparation,preparation_batches=preparation_batches,asr_queues=asr_queues,storage_jobs=storage_jobs,persistent_asr_batches=persistent_batches,text_review_batches=review_batches,
                pld_pilot={'recordings':229,'hours':2.9521978298611113,'reserved_test_recordings':39,'audio_verified':0},
                limitations=['Raw runtime includes ads/music/silence. Source language labels unverified.',
                             'Cross-source and acoustic near-duplicate checks pending. Not unique speech-hour totals.',
                             'Original PLD pilot is separate from newly collected Chika audio.',
                             'ASR figures are draft transport-window runtime, not verified unique speech. Segmentation, new-audio review and final training exports remain incomplete.'])


if __name__=='__main__':
    target=ROOT/'STATUS.json'
    target.write_text(json.dumps(snapshot(),indent=2)+'\n')
    print(target)
