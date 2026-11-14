import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import status


class JobObservationTests(unittest.TestCase):
    def test_actual_process_and_summary_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'curation';root=base/'outputs/job';root.mkdir(parents=True)
            worker=root/'worker.json';worker.write_text(json.dumps({'pid':123,'status':'running','command':['private'],'key':'never expose'}))
            summary=root/'summary.json';summary.write_text(json.dumps({'selected':3,'completed':[{'id':'a'},{'id':'b'}],'failed':[{'id':'c'}],'sentence':'private transcript'}))
            with patch.object(status,'ROOT',base), patch.object(status.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='')):
                result=status.job_observation(worker,summary)
            self.assertFalse(result['process_present'])
            self.assertFalse(result['process_matches_job'])
            self.assertEqual(result['summary'],{'selected':3,'completed':2,'failed':1})
            self.assertNotIn('private',json.dumps(result))
            with patch.object(status,'ROOT',base), patch.object(status.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='123 another-process')):
                result=status.job_observation(worker,summary)
            self.assertTrue(result['process_present'])
            self.assertFalse(result['process_matches_job'])


if __name__=='__main__':unittest.main()
