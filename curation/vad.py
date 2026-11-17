"""Private bounded Silero ONNX speech candidates, not language/music/speaker QC.

Inference contract follows upstream utils_vad.OnnxWrapper (MIT). Segmentation is
our hysteresis + silence-aware partition, not upstream get_speech_timestamps.
Run: python -m curation.vad --manifest ... --output ... --model ...
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import os
import tempfile
import urllib.request

import numpy as np
import soundfile as sf
from .schema import digest, load_manifest, validate

RATE = 16000
MAX_SEGMENT_SAMPLES = 27 * RATE  # Headroom below observed Boses 28-second automatic cuts.
REVISION = '066409f3d9db45a30d21dfcbf9147826ffdd7052'
MODEL_URL = f'https://raw.githubusercontent.com/snakers4/silero-vad/{REVISION}/src/silero_vad/data/silero_vad.onnx'
MODEL_SHA256 = '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_file(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            temp = Path(tmp.name)
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=90) as response, temp.open('wb') as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if digest(temp) != MODEL_SHA256:
                raise ValueError('model checksum mismatch')
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    if digest(path) != MODEL_SHA256:
        raise ValueError('model checksum mismatch')
    return path


class Silero:
    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_file(path)), sess_options=opts,
                                            providers=['CPUExecutionProvider'])

    def probabilities(self, audio):
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)
        result = []
        for start in range(0, len(audio), 512):
            chunk = np.zeros((1, 512), dtype=np.float32)
            samples = audio[start:start + 512]
            chunk[0, :len(samples)] = samples
            batch = np.concatenate([context, chunk], axis=1)
            probability, state = self.session.run(None, {'input': batch, 'state': state,
                                                         'sr': np.array(RATE, dtype=np.int64)})
            result.append(float(probability.reshape(-1)[0]))
            context = batch[:, -64:]
        return result


def intervals(probabilities, frames, threshold=.5, pad=2400, cap=MAX_SEGMENT_SAMPLES):
    """Hysteresis; 128ms silence ends region, 256ms minimum before padding.

    Padding is merged before cap splitting so every output sample is unique.
    Splits prefer a low-probability frame in the last 5 seconds before cap;
    continuous speech necessarily uses hard cuts (alignment remains required).
    """
    regions = []; start = None; silence = None
    for i, p in enumerate(probabilities):
        pos = i * 512
        if p >= threshold:
            if start is None: start = pos
            silence = None
        elif start is not None and p < max(.01, threshold - .15):
            if silence is None: silence = pos
            if pos + 512 - silence >= 2048:
                if silence - start >= 4096: regions.append((start, silence))
                start = silence = None
    if start is not None:
        end = silence if silence is not None else frames
        if end - start >= 4096: regions.append((start, end))
    merged = []
    for start, end in regions:
        start, end = max(0, start-pad), min(frames, end+pad)
        if merged and start <= merged[-1][1]: merged[-1] = (merged[-1][0], end)
        else: merged.append((start, end))
    output = []
    for start, end in merged:
        while end-start > cap:
            limit = start+cap
            candidates = [i*512 for i in range(max(1, (limit-5*RATE)//512), limit//512+1)
                          if start < i*512 <= limit and probabilities[i] < threshold-.15]
            cut = candidates[-1] if candidates else limit
            output.append((start, cut)); start = cut
        output.append((start, end))
    return output


def derive(parent, path, start, end, cfg):
    provenance = parent['provenance']
    offset = provenance['start_sample']
    row = {key: copy.deepcopy(parent[key]) for key in
           ('schema_version', 'source_id', 'recording_id', 'prompt', 'text_is_prompt',
            'language', 'languages', 'speech_type', 'speaker_id', 'split', 'rights')}
    for key in ('synthetic', 'language_annotation'):
        if key in parent: row[key] = copy.deepcopy(parent[key])
    row.update(id=identity({'parent': identity(parent), 'start': start, 'end': end, 'config': cfg}),
               audio_path=str(path.resolve()), audio_sha256=digest(path), audio_bytes=path.stat().st_size,
               sentence='', num_words=0, review_status='draft', asr_status='pending', llm_status='pending',
               duration=(end-start)/RATE, start=(offset+start)/RATE, end=(offset+end)/RATE,
               timestamp_kind='VAD sample boundaries; not word alignment',
               flags=list(dict.fromkeys(parent.get('flags', []) +
                      ['vad_speech_candidate', 'language_unverified', 'speaker_unknown',
                       'music_overlap_unverified', 'alignment_required_for_tts'])),
               provenance={**copy.deepcopy(provenance), 'parent_record': copy.deepcopy(parent),
                           'parent_manifest_id': parent['id'], 'vad_config': cfg,
                           'start_sample': offset+start, 'end_sample': offset+end,
                           'parent_start_sample': start, 'parent_end_sample': end,
                           'offset_sample_rate': RATE})
    return validate(row)


def run(manifest, output, model, max_minutes=2, max_rows=4, threshold=.5):
    if not math.isfinite(max_minutes) or max_minutes <= 0 or max_rows < 1 or not 0 < threshold < 1:
        raise ValueError('invalid bounds/threshold')
    import onnxruntime
    cfg = {'version': 3, 'model_url': MODEL_URL, 'model_sha256': MODEL_SHA256,
           'model_license': 'MIT', 'onnxruntime': onnxruntime.__version__, 'threshold': threshold,
           'padding_samples': 2400, 'max_segment_samples': MAX_SEGMENT_SAMPLES,
           'max_minutes': max_minutes, 'max_rows': max_rows, 'sample_rate': RATE}
    parents = load_manifest(manifest)
    selected = []; samples = 0; ends = {}
    for row in parents:
        if len(selected) >= max_rows: break
        info = sf.info(row['audio_path'])
        if info.samplerate != RATE or info.channels != 1 or info.subtype != 'PCM_16':
            raise ValueError('requires prepared 16k mono PCM16')
        if samples + info.frames > max_minutes*60*RATE: break
        if digest(row['audio_path']) != row['audio_sha256']: raise ValueError('parent checksum mismatch')
        prov = row['provenance']; start = prov.get('start_sample'); end = prov.get('end_sample')
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end-start != info.frames or prov.get('offset_sample_rate') != RATE:
            raise ValueError('invalid source sample mapping')
        key = (row['source_id'], row['recording_id'])
        for a,b in ends.setdefault(key, []):
            if start < b and end > a: raise ValueError('overlapping parent windows')
        ends[key].append((start,end)); selected.append(row); samples += info.frames
    if not selected: raise ValueError('no complete input windows within budget')
    root = Path(output); root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    with (root/'vad.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = {'config': cfg, 'manifest_sha256': digest(manifest), 'parents': [identity(r) for r in selected]}
        snap_path = root/'snapshot.json'
        if snap_path.exists() and json.loads(snap_path.read_text()) != snapshot:
            raise ValueError('input/config changed; use a new output root')
        snap_path.write_text(json.dumps(snapshot, indent=2)+'\n')
        dest = root/'manifest.jsonl'
        if dest.exists():
            cached = load_manifest(dest)
            summary = json.loads((root/'summary.json').read_text())
            if digest(dest) != summary['manifest_sha256']: raise ValueError('stale output manifest')
            if all(digest(r['audio_path']) == r['audio_sha256'] for r in cached): return summary
            raise ValueError('stale output audio; use a new output root')
        vad = Silero(model); rows = []
        for parent in selected:
            audio, _ = sf.read(parent['audio_path'], dtype='float32')
            spans = intervals(vad.probabilities(audio), len(audio), threshold)
            for start,end in spans:
                path = root/(identity({'parent':identity(parent),'start':start,'end':end,'cfg':cfg})+'.flac')
                sf.write(path, audio[start:end], RATE, subtype='PCM_16')
                rows.append(derive(parent,path,start,end,cfg))
        tmp = root/'manifest.tmp'
        tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        summary = {'input_windows':len(selected),'input_hours':samples/RATE/3600,
                   'segments':len(rows),'candidate_hours':sum(r['duration'] for r in rows)/3600,
                   'candidate_hours_kind':'padded VAD segments, including retained silence; not verified speech',
                   'transcribed_hours':0,'verified_hours':0,'manifest_sha256':digest(tmp),
                   'limitations':'Speech heuristic only; languages not benchmarked; no music, speaker, language or rights verification. Window-edge context resets; hard cuts require alignment.'}
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        tmp.replace(dest)
        return summary


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--model', default='curation/outputs/vad-model/silero_vad.onnx')
    p.add_argument('--max-minutes',type=float,default=2); p.add_argument('--max-rows',type=int,default=4)
    p.add_argument('--threshold',type=float,default=.5)
    a=p.parse_args()
    os.umask(0o077)
    print(json.dumps(run(a.manifest,a.output,a.model,a.max_minutes,a.max_rows,a.threshold)))


if __name__=='__main__': main()
