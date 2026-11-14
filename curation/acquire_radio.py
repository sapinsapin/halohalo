"""Bounded public radio capture, with source provenance and recoverable status.

python -m curation.acquire_radio --station bombo-iloilo --seconds 1800
Each invocation records a new UTC interval; it does not resume a missing live interval.
No ASR, upload or rights clearance is implied by a successful capture.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

STATIONS = {
    'bombo-vigan': ('https://stream.zenolive.com/ebcyau9nz9duv', 'https://www.bomboradyo.com/live-stream'),
    'rmn-iloilo': ('https://us1.amfmph.com/ssl/rmniloilo?mp=%2Fstream', 'https://onlineradio.ph/934-rmn-bacolod.html'),
    'rmn-bacolod': ('http://us2.amfmph.com:8848/live', 'https://onlineradio.ph/934-rmn-bacolod.html'),
    'bombo-iloilo': ('https://stream.zenolive.com/gzudz59nz9duv', 'https://www.bomboradyo.com/live-stream'),
    'bombo-bacolod': ('https://stream.zenolive.com/wg8yn4cnz9duv', 'https://www.bomboradyo.com/live-stream'),
    'bombo-roxas': ('https://stream.zenolive.com/gngddnapz9duv', 'https://www.bomboradyo.com/live-stream'),
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def capture(station, seconds, root, min_free_gib=25):
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < min_free_gib * 2**30:
        raise RuntimeError('Free disk below capture floor')
    url, source_page = STATIONS[station]
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory = root / station / stamp
    directory.mkdir(parents=True, exist_ok=False)
    # Matroska survives interrupted writes; preserve original audio codec/sample rate.
    audio = directory / 'capture.mka'
    status_path = directory / 'status.json'
    status = dict(id=f'{station}:{stamp}', station=station, source_url=url,
                  source_page=source_page, source_kind='live_radio',
                  started_at=now(), requested_seconds=seconds, runner_pid=os.getpid(),
                  candidate_language='ilo' if station == 'bombo-vigan' else 'hil', language_verified=False, code_switching_expected=True,
                  rights={'training':'unknown','commercial':'unknown','redistribution':'unknown'},
                  transcript_status='not_started', state='starting', audio_path=str(audio.resolve()))
    atomic(status_path, status)
    command = ['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'warning',
               '-rw_timeout', '20000000', '-i', url, '-t', str(seconds),
               '-map', '0:a:0', '-vn', '-c:a', 'copy', '-f', 'matroska', str(audio)]
    reason = None
    with (directory / 'ffmpeg.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        status.update(state='recording', ffmpeg_pid=process.pid)
        atomic(status_path, status)
        began = time.monotonic()
        deadline = began + seconds + 60
        def stop(signum, frame):
            nonlocal reason
            reason = 'signal_' + str(signum)
            process.terminate()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while process.poll() is None:
                if time.monotonic() - began > 90 and (not audio.exists() or not audio.stat().st_size):
                    reason = 'no_audio_timeout'; process.terminate(); break
                if shutil.disk_usage(root).free < min_free_gib * 2**30:
                    reason = 'disk_floor'; process.terminate(); break
                if time.monotonic() > deadline:
                    reason = 'wall_timeout'; process.terminate(); break
                status.update(heartbeat_at=now(), bytes=audio.stat().st_size if audio.exists() else 0)
                atomic(status_path, status)
                time.sleep(5)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        except BaseException:
            process.terminate()
            process.wait(timeout=15)
            raise
    status.update(ended_at=now(), returncode=process.returncode, stop_reason=reason)
    if audio.exists() and audio.stat().st_size:
        result = subprocess.run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(audio)],capture_output=True,text=True,timeout=30)
        if result.returncode == 0:
            probe = json.loads(result.stdout)
            atomic(directory / 'ffprobe.json', probe)
            duration = float(probe['format'].get('duration',0))
            h=hashlib.sha256()
            with audio.open('rb') as f:
                for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
            status.update(duration=duration, audio_sha256=h.hexdigest(), bytes=audio.stat().st_size,
                          state='acquired' if process.returncode == 0 and duration >= seconds-2 else 'partial')
        else:
            status.update(state='failed', probe_error=result.stderr[-2000:])
    else:
        status['state']='failed'
    atomic(status_path, status)
    print(json.dumps(status), flush=True)
    return status


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--station',choices=STATIONS,required=True)
    p.add_argument('--seconds',type=int,default=1800)
    p.add_argument('--output',type=Path,default=Path('curation/outputs/radio'))
    p.add_argument('--min-free-gib',type=float,default=25)
    a=p.parse_args()
    if not 1 <= a.seconds <= 86400: p.error('--seconds must be between 1 and 86400')
    status=capture(a.station,a.seconds,a.output,a.min_free_gib)
    raise SystemExit(0 if status['state']=='acquired' else 1)

if __name__=='__main__': main()
