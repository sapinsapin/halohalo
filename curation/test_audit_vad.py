import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from .audit_vad import audit
from .schema import VERSION, digest
from .vad import derive


class AuditTests(unittest.TestCase):
    def test_independent_parent_detects_rehashed_wrong_audio_and_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = np.arange(32000, dtype=np.int16)
            audio = root/'parent.flac'
            sf.write(audio, data, 16000, subtype='PCM_16')
            parent = dict(schema_version=VERSION, id='p', source_id='s', recording_id='r',
                          audio_path=str(audio), audio_sha256=digest(audio), sentence='',
                          num_words=0, prompt=None, text_is_prompt=False, language='ilo',
                          languages=['ilo'], speech_type='unknown', speaker_id='', duration=2,
                          split='unassigned', review_status='draft',
                          rights={k:'unknown' for k in ('training','commercial','redistribution')},
                          provenance={'start_sample':160000,'end_sample':192000,'offset_sample_rate':16000})
            parents = root/'parents.jsonl'
            parents.write_text(json.dumps(parent)+'\n')
            child = root/'child.flac'
            sf.write(child, data[8000:24000], 16000, subtype='PCM_16')
            row = derive(parent, child, 8000, 24000, {})
            candidates = root/'candidates.jsonl'
            def save(): candidates.write_text(json.dumps(row)+'\n')
            save()
            self.assertEqual(audit(parents, candidates)['exact_pcm_samples'], 16000)
            row['rights']['training'] = 'allowed'
            row['rights']['evidence'] = 'untrusted candidate claim'
            save()
            with self.assertRaisesRegex(ValueError, 'inherited metadata'):
                audit(parents, candidates)
            row['rights']['training'] = 'unknown'
            del row['rights']['evidence']
            row['provenance']['parent_record']['language'] = 'hil'
            save()
            with self.assertRaisesRegex(ValueError, 'embedded parent'):
                audit(parents, candidates)
            row['provenance']['parent_record']['language'] = 'ilo'
            row['start'] += 1
            save()
            with self.assertRaisesRegex(ValueError, 'seconds differ'):
                audit(parents, candidates)
            row['start'] -= 1
            sf.write(child, data[8001:24001], 16000, subtype='PCM_16')
            row.update(audio_sha256=digest(child), audio_bytes=child.stat().st_size)
            save()
            with self.assertRaisesRegex(ValueError, 'PCM differs'):
                audit(parents, candidates)


if __name__ == '__main__':
    unittest.main()
