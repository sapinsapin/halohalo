import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .acquire_inventory import acquire


class InventoryTests(unittest.TestCase):
    def test_language_hash_cache_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ident='abcdefghijk';item={'id':ident}
            def download(*args,**kwargs):
                (root/(ident+'.m4a')).write_bytes(b'actual audio fixture')
                (root/(ident+'.info.json')).write_text('{}')
                return 0
            with patch('curation.acquire_inventory.run_transfer',side_effect=download) as run, patch('curation.acquire_inventory.probe',return_value={'format':{'duration':'3'},'streams':[{'codec_type':'audio','sample_rate':'16000'}]}):
                r=acquire(item,root,'bik','dzgb',{'variety_claim':'unverified'},0)
                self.assertEqual(r['language'],'bik');self.assertEqual(r['duration'],3)
                self.assertEqual(r['rights']['training'],'unknown')
                self.assertEqual(acquire(item,root,'bik','dzgb',{},0),r)
                self.assertEqual(run.call_count,1)
                (root/(ident+'.m4a')).write_bytes(b'corrupt')
                r=acquire(item,root,'bik','dzgb',{},0)
                self.assertEqual(run.call_count,2);self.assertEqual(r['attempts'],2)
            with self.assertRaises(ValueError):acquire(item,root,'hil','dzgb',{},0)
            (root/(ident+'.m4a')).write_bytes(b'corrupt again')
            exhausted=acquire(item,root,'bik','dzgb',{},0)
            self.assertEqual(exhausted['status'],'failed')
            self.assertEqual(exhausted['attempts'],2)

    def test_live_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('curation.acquire_inventory.run_transfer') as run:
                r=acquire({'id':'abcdefghijk','live_status':'is_live'},Path(tmp),'bik','dzgb',{},0)
                self.assertEqual(r['status'],'excluded');run.assert_not_called()
                with self.assertRaises(ValueError):acquire({'id':'../escape'},Path(tmp),'bik','dzgb',{},0)


if __name__=='__main__':unittest.main()
