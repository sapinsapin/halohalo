import copy
import unittest
from .deduplicate import select


class DeduplicateTests(unittest.TestCase):
    def rows(self):
        base=dict(id='a',audio_sha256='f'*64,sentence='Hello gid',split='unassigned',language='hil',languages=['hil','en'],rights={'training':'unknown'},duration=1.0,review_status='draft')
        other=copy.deepcopy(base);other.update(id='b',review_status='text_reviewed',recording_id='other')
        return [base,other]

    def test_preserve_best_row_and_full_alias(self):
        rows=self.rows();before=copy.deepcopy(rows)
        canonical,aliases=select(rows)
        self.assertEqual(canonical,[rows[1]])
        self.assertEqual(aliases[0]['original_row'],rows[0])
        self.assertEqual(rows,before)
        self.assertEqual(select(list(reversed(rows))), (canonical,aliases))

    def test_conflicts_fail(self):
        for key,value in [('sentence','Different'),('split','train'),('rights',{'training':'allowed'}),('languages',['hil']),('duration',2.0)]:
            with self.subTest(key=key):
                rows=self.rows();rows[1][key]=value
                with self.assertRaises(ValueError):select(rows)

    def test_equal_sentences_different_audio_retained(self):
        rows=self.rows();rows[1]['audio_sha256']='e'*64
        canonical,aliases=select(rows)
        self.assertEqual(len(canonical),2);self.assertEqual(aliases,[])
