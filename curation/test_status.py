import json
import sqlite3
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

    def test_publication_complete_does_not_imply_card_or_stream_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'curation';root=base/'outputs/release';root.mkdir(parents=True)
            worker=root/'publisher.json';worker.write_text(json.dumps({'status':'complete'}))
            (root/'hub-receipt.json').write_text(json.dumps({'verified':True,'rows':5,'files':{'private':'data'}}))
            with patch.object(status,'ROOT',base):
                result=status.publication_observation(worker)
            self.assertEqual(result['recorded_receipts'],{'hub-receipt.json':{'rows':5,'verified':True}})
            self.assertNotIn('files',json.dumps(result))
            (root/'card-receipt.json').write_text(json.dumps({'revision':'card','config':'release','card_verified':True}))
            (root/'stream-check.json').write_text(json.dumps({'revision':'stream','rows':5,'exact_ids_and_sentences':True}))
            with patch.object(status,'ROOT',base):
                result=status.publication_observation(worker)
            self.assertEqual(result['recorded_receipts']['card-receipt.json']['revision'],'card')
            self.assertEqual(result['recorded_receipts']['stream-check.json']['revision'],'stream')


class RssRetentionTests(unittest.TestCase):
    def test_ledger_accounting_and_missing_backup_table(self):
        db = sqlite3.connect(':memory:')
        self.addCleanup(db.close)
        db.execute(
            'CREATE TABLE items (status TEXT, duration REAL, sha256 TEXT)'
        )
        db.executemany('INSERT INTO items VALUES (?, ?, ?)', [
            ('downloaded', 3600, 'local-1'),
            ('downloaded', 3600, 'local-2'),
            ('remote', 1800, 'remote-backed'),
            ('remote', 1800, 'remote-unbacked'),
            ('pending', 32400, 'pending'),
        ])
        db.execute("""
            CREATE TABLE hub_backups (
                sha256 TEXT, repo TEXT, path TEXT, revision TEXT,
                verified_at TEXT, bytes INTEGER
            )
        """)
        valid = (
            'remote-backed', 'repo', 'audio/item', 'revision',
            '2025-01-01T00:00:00Z', 1800,
        )
        incomplete = (
            'remote-unbacked', 'repo', 'audio/other', '',
            '2025-01-01T00:00:00Z', 1800,
        )
        db.executemany(
            'INSERT INTO hub_backups VALUES (?, ?, ?, ?, ?, ?)',
            [valid, valid, incomplete],
        )

        result = status.rss_retention(db)
        expected = {
            'local_raw_hours': 2.0,
            'backed_up_remote_raw_hours': 0.5,
            'unbacked_remote_raw_hours': 0.5,
            'retained_raw_hours': 2.5,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value)
        self.assertIn('not fresh remote verification', result['retention_note'])

        db.execute('DROP TABLE hub_backups')
        legacy = status.rss_retention(db)
        self.assertEqual(
            {key: legacy[key] for key in expected},
            {
                'local_raw_hours': 2.0,
                'backed_up_remote_raw_hours': 0.0,
                'unbacked_remote_raw_hours': 1.0,
                'retained_raw_hours': 2.0,
            },
        )


if __name__=='__main__':unittest.main()
