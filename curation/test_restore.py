"""Recovery tests: temporary real SQLite checkpoints and a strictly mocked Hub."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from curation.collect import connect, file_hash
from curation.restore import receipt_value, restore

REV = 'a' * 40


class FakeHub:
    def __init__(self):
        self.files = {}
        self.metadata = {}
        self.download_calls = []
        self.metadata_calls = []

    def download(self, repo_id, repo_type, filename, revision):
        assert repo_type == 'dataset'
        key = (repo_id, revision, filename)
        self.download_calls.append(key)
        if key not in self.files:
            raise FileNotFoundError('missing remote object')
        return self.files[key]

    def get_paths_info(self, repo, paths, repo_type, revision):
        assert repo_type == 'dataset'
        self.metadata_calls.append((repo,revision,tuple(paths)))
        return [self.metadata[(repo,revision,p)] for p in paths if (repo,revision,p) in self.metadata]


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.original = self.base/'original'
        self.dest = self.base/'restored'
        self.hub = FakeHub()
        self.db = connect(self.original)
        self.db.execute('CREATE TABLE hub_backups(sha256 TEXT PRIMARY KEY,repo TEXT,path TEXT,revision TEXT,verified_at TEXT,bytes INTEGER)')
        self.db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?)',
                        ('show','https://example.test/rss','fil',json.dumps({'feed_snapshot':'/old/machine/feed.xml'}),'now',None))
        self.rows = []

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def add(self, ident, payload=b'audio bytes', backed=True, state='downloaded'):
        local = self.original/f'{ident}.media'
        local.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        self.db.execute('''INSERT INTO items(id,source_id,url,status,path,sha256,duration,bytes,attempts,etag,last_modified)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                        (ident,'show','https://example.test/'+ident,state,str(local),sha,12.5,len(payload),2,'old-etag','old-modified'))
        if backed:
            remote = f'raw/test/{sha}.media'
            self.db.execute('INSERT OR REPLACE INTO hub_backups VALUES(?,?,?,?,?,?)',
                            (sha,'sapinsapin/chika',remote,REV,'now',len(payload)))
            key = ('sapinsapin/chika',REV,remote)
            self.hub.files[key] = str(local)
            self.hub.metadata[key] = SimpleNamespace(lfs=SimpleNamespace(sha256=sha),size=len(payload))
        self.rows.append((ident,local,sha))
        return sha

    def checkpoint(self):
        self.db.commit()
        path = self.base/'checkpoint.sqlite'
        dest = sqlite3.connect(path)
        self.db.backup(dest)
        dest.close()
        sha = file_hash(path)
        coords = dict(repo='sapinsapin/chika',revision=REV,checkpoint=f'checkpoints/test/{sha}.sqlite',checkpoint_sha256=sha)
        self.hub.files[(coords['repo'],REV,coords['checkpoint'])] = str(path)
        self.receipt = self.base/'receipt.json'
        self.receipt.write_text(json.dumps(coords))
        return coords,path

    def run_restore(self, **kwargs):
        return restore(self.dest,receipt=self.receipt,api=self.hub,downloader=self.hub.download,
                       min_free_gib=0,**kwargs)

    def restored_rows(self):
        db = sqlite3.connect(self.dest/'state.sqlite')
        db.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in db.execute('SELECT * FROM items ORDER BY id')]
        finally:
            db.close()

    def assert_no_publish(self):
        self.assertFalse(self.dest.exists())
        self.assertEqual(list(self.base.glob('.chika-restore-*')), [])

    def test_bounded_restore_rebinds_paths_and_missing_backups_are_pending(self):
        self.add('a',b'first')
        self.add('b',b'second')
        self.add('c',b'no backup',backed=False)
        self.checkpoint()
        original_hashes = {str(p):file_hash(p) for _,p,_ in self.rows}
        summary = self.run_restore(max_files=1)
        a,b,c = self.restored_rows()
        self.assertEqual(summary['local_unique_files'],1)
        self.assertEqual(a['status'],'downloaded')
        self.assertTrue(Path(a['path']).is_relative_to(self.dest))
        self.assertEqual(Path(a['path']).read_bytes(),b'first')
        self.assertEqual(b['status'],'remote')
        self.assertIsNone(b['path'])
        self.assertEqual(c['status'],'pending')
        for k in ('path','sha256','duration','bytes','etag','last_modified'):
            self.assertIsNone(c[k])
        self.assertEqual(c['attempts'],0)
        self.assertEqual(original_hashes,{str(p):file_hash(p) for _,p,_ in self.rows})
        with sqlite3.connect(self.dest/'state.sqlite') as db:
            self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0],'ok')
            meta=json.loads(db.execute('SELECT metadata FROM sources').fetchone()[0])
            self.assertNotIn('feed_snapshot',meta)
            self.assertFalse(meta['feed_snapshot_available'])
            self.assertEqual(db.execute('SELECT count(*) FROM restore_inventory').fetchone()[0],3)

    def test_zero_files_is_metadata_only_and_does_not_request_media(self):
        self.add('a')
        self.checkpoint()
        summary=self.run_restore(max_files=0)
        self.assertEqual(summary['status_counts'],{'remote':1})
        self.assertEqual(len(self.hub.download_calls),1)
        self.assertEqual(self.hub.metadata_calls,[])
        self.assertIsNone(self.restored_rows()[0]['path'])

    def test_same_audio_hash_rehydrates_once_for_multiple_rows(self):
        self.add('a',b'same')
        self.add('b',b'same')
        self.checkpoint()
        summary=self.run_restore(max_files=1)
        self.assertEqual(summary['local_unique_files'],1)
        self.assertEqual(summary['status_counts'],{'downloaded':2})
        self.assertEqual(len(self.hub.download_calls),2)
        a,b=self.restored_rows()
        self.assertEqual(a['path'],b['path'])

    def test_checkpoint_hash_mismatch_does_not_publish(self):
        self.add('a')
        _,p=self.checkpoint()
        p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint SHA256 mismatch'):
            self.run_restore()
        self.assert_no_publish()

    def test_invalid_sqlite_with_correct_hash_does_not_publish(self):
        path=self.base/'bad.sqlite';path.write_bytes(b'not sqlite')
        sha=file_hash(path)
        coords=dict(repo='sapinsapin/chika',revision=REV,checkpoint=f'checkpoints/test/{sha}.sqlite',checkpoint_sha256=sha)
        self.receipt=self.base/'receipt.json';self.receipt.write_text(json.dumps(coords))
        self.hub.files[(coords['repo'],REV,coords['checkpoint'])]=str(path)
        with self.assertRaises(sqlite3.DatabaseError):self.run_restore()
        self.assert_no_publish()

    def test_missing_remote_backup_does_not_publish(self):
        self.add('a')
        self.checkpoint()
        self.hub.metadata.clear()
        with self.assertRaisesRegex(ValueError,'missing from immutable revision'):self.run_restore()
        self.assert_no_publish()

    def test_remote_hash_mismatch_does_not_download_or_publish(self):
        self.add('a')
        self.checkpoint()
        next(iter(self.hub.metadata.values())).lfs.sha256='b'*64
        with self.assertRaisesRegex(ValueError,'remote media hash/size mismatch'):self.run_restore()
        self.assertEqual(len(self.hub.download_calls),1)
        self.assert_no_publish()

    def test_downloaded_bytes_mismatch_does_not_publish(self):
        self.add('a',b'good')
        self.checkpoint()
        corrupted=self.base/'bad.media';corrupted.write_bytes(b'evil')
        media_key=next(iter(self.hub.metadata))
        self.hub.files[media_key]=str(corrupted)
        with self.assertRaisesRegex(ValueError,'downloaded media hash/size mismatch'):self.run_restore()
        self.assert_no_publish()

    def test_nonempty_root_is_never_modified(self):
        self.add('a');self.checkpoint()
        self.dest.mkdir();marker=self.dest/'keep';marker.write_text('untouched')
        with self.assertRaisesRegex(ValueError,'new or empty'):self.run_restore()
        self.assertEqual(marker.read_text(),'untouched')
        self.assertEqual(self.hub.download_calls,[])

    def test_byte_budget_checked_before_media_download(self):
        self.add('a',b'1234567890');self.checkpoint()
        with self.assertRaisesRegex(ValueError,'exceed max-bytes'):self.run_restore(max_bytes=5)
        self.assertEqual(len(self.hub.download_calls),1)
        self.assert_no_publish()

    def test_storage_floor_checked_before_any_network_download(self):
        self.add('a');self.checkpoint()
        with patch('curation.restore.shutil.disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(OSError,'storage floor reached before restore'):
                restore(self.dest,receipt=self.receipt,api=self.hub,downloader=self.hub.download,min_free_gib=1)
        self.assertEqual(self.hub.download_calls,[])
        self.assert_no_publish()

    def test_space_for_cache_and_copy_checked_before_media_download(self):
        self.add('a',b'1234567890');self.checkpoint()
        with patch('curation.restore.shutil.disk_usage', return_value=SimpleNamespace(free=15)):
            with self.assertRaisesRegex(OSError,'storage floor would be crossed'):
                self.run_restore()
        self.assertEqual(len(self.hub.download_calls),1)
        self.assert_no_publish()

    def test_mutable_revision_and_conflicting_hash_rejected(self):
        self.add('a');coords,_=self.checkpoint()
        with self.assertRaisesRegex(ValueError,'immutable'):
            receipt_value(repo=coords['repo'],revision='main',checkpoint=coords['checkpoint'])
        with self.assertRaisesRegex(ValueError,'conflicts'):
            receipt_value(repo=coords['repo'],revision=REV,checkpoint=coords['checkpoint'],checkpoint_sha256='f'*64)

    def test_explicit_coordinates_can_restore_without_receipt(self):
        self.add('a');coords,_=self.checkpoint()
        summary=restore(self.dest,**coords,max_files=0,min_free_gib=0,api=self.hub,downloader=self.hub.download)
        self.assertEqual(summary['status_counts'],{'remote':1})


if __name__=='__main__':unittest.main()
