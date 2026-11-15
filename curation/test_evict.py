"""Eviction proof, live-lock, crash-recovery and restore-compatibility tests."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from curation.collect import connect, file_hash
from curation.evict import evict
from curation.restore import restore
from curation.test_restore import FakeHub, REV


class EvictionHub(FakeHub):
    private = True

    def dataset_info(self, repo):
        return SimpleNamespace(private=self.private)


class EvictTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.root=self.base/'archive'
        self.cache=self.base/'cache';self.cache.mkdir()
        self.db=connect(self.root)
        self.db.execute('CREATE TABLE hub_backups(sha256 TEXT PRIMARY KEY,repo TEXT,path TEXT,revision TEXT,verified_at TEXT,bytes INTEGER)')
        self.db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?)',('show','https://example.test/rss','hil','{}','now',None))
        self.hub=EvictionHub()
        self.media={}

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def add(self,ident='a',payload=b'original audio',backed=True,path=None):
        path=path or self.root/'audio'/'show'/f'{ident}.media'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(payload)
        sha=hashlib.sha256(payload).hexdigest()
        self.db.execute('''INSERT INTO items(id,source_id,url,guid,title,status,path,sha256,duration,bytes,attempts)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                        (ident,'show','https://example.test/'+ident,'guid-'+ident,'title-'+ident,'downloaded',str(path),sha,20.0,len(payload),1))
        if backed:
            remote=f'raw/test/{sha}.media'
            self.db.execute('INSERT OR REPLACE INTO hub_backups VALUES(?,?,?,?,?,?)',
                            (sha,'sapinsapin/chika',remote,REV,'verified',len(payload)))
            cached=self.cache/f'{sha}.media';cached.write_bytes(payload)
            key=('sapinsapin/chika',REV,remote)
            self.hub.files[key]=str(cached)
            self.hub.metadata[key]=SimpleNamespace(lfs={'sha256':sha},size=len(payload))
        self.media[ident]=path
        return path

    def checkpoint(self,mutate=None):
        self.db.commit()
        checkpoint=self.cache/'state.sqlite'
        dest=sqlite3.connect(checkpoint)
        self.db.backup(dest)
        if mutate:mutate(dest)
        dest.commit();dest.close()
        sha=file_hash(checkpoint)
        coords=dict(repo='sapinsapin/chika',revision=REV,checkpoint=f'checkpoints/test/{sha}.sqlite',checkpoint_sha256=sha)
        self.hub.files[(coords['repo'],REV,coords['checkpoint'])]=str(checkpoint)
        self.receipt=self.root/'hub-checkpoint.json';self.receipt.write_text(json.dumps(coords))
        return checkpoint

    def run_evict(self,**kwargs):
        return evict(self.root,api=self.hub,downloader=self.hub.download,**kwargs)

    def row(self,ident='a'):
        return dict(self.db.execute('SELECT * FROM items WHERE id=?',(ident,)).fetchone())

    def journal(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM eviction_journal')]

    def test_dry_run_preserves_files_ledger_and_does_not_create_journal(self):
        path=self.add();self.checkpoint();before=self.row()
        report=self.run_evict()
        self.assertTrue(report['dry_run']);self.assertEqual(report['planned_files'],1)
        self.assertEqual(report['evicted_files'],0);self.assertTrue(path.exists())
        self.assertEqual(before,self.row())
        self.assertIsNone(self.db.execute("SELECT name FROM sqlite_master WHERE name='eviction_journal'").fetchone())

    def test_explicit_hash_subset_ignores_newer_uncheckpointed_backup(self):
        older=self.add('z',b'old audio');self.checkpoint()
        newer=self.add('a',b'new audio');self.db.commit()
        sha=file_hash(older)
        result=self.run_evict(apply=True,max_files=10,only_sha256=[sha])
        self.assertEqual(result['evicted_files'],1)
        self.assertFalse(older.exists());self.assertTrue(newer.exists())
        self.assertEqual(self.row('a')['status'],'downloaded')

    def test_hash_filter_never_bypasses_checkpoint_proof(self):
        self.add('z',b'old');self.checkpoint()
        newer=self.add('a',b'new');self.db.commit()
        with self.assertRaisesRegex(ValueError,'immutable checkpoint'):
            self.run_evict(apply=True,only_sha256=[file_hash(newer)])
        self.assertTrue(newer.exists())

    def test_empty_and_invalid_hash_filters(self):
        path=self.add();self.checkpoint()
        self.assertEqual(self.run_evict(apply=True,only_sha256=[])['evicted_files'],0)
        for value in (['bad'],[None],'a'*64):
            with self.assertRaises(ValueError):self.run_evict(apply=True,only_sha256=value)
        self.assertTrue(path.exists())

    def test_apply_preserves_metadata_and_can_restore_from_same_checkpoint(self):
        path=self.add();payload=path.read_bytes();self.checkpoint();before=self.row()
        report=self.run_evict(apply=True)
        self.assertFalse(path.exists());self.assertEqual(report['evicted_files'],1)
        after=self.row();self.assertEqual(after['status'],'remote');self.assertIsNone(after['path'])
        for k in before.keys()-{'status','path','updated_at'}:self.assertEqual(before[k],after[k])
        self.assertEqual(self.journal()[0]['state'],'unlinked')
        fresh=self.base/'restored'
        summary=restore(fresh,receipt=self.receipt,max_files=1,min_free_gib=0,api=self.hub,downloader=self.hub.download)
        self.assertEqual(summary['status_counts'],{'downloaded':1})
        self.assertEqual(Path(summary['files'][0]['path']).read_bytes(),payload)

    def test_live_downloader_lock_prevents_any_network_or_eviction(self):
        path=self.add();self.checkpoint()
        with (self.root/'download.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(BlockingIOError,'no eviction performed'):self.run_evict(apply=True)
        self.assertTrue(path.exists());self.assertEqual(self.row()['status'],'downloaded')
        self.assertEqual(self.hub.download_calls,[])

    def test_missing_immutable_backup_proof_refuses_to_delete(self):
        path=self.add();self.checkpoint(lambda db:db.execute('DELETE FROM hub_backups'))
        with self.assertRaisesRegex(ValueError,'does not prove'):self.run_evict(apply=True)
        self.assertTrue(path.exists());self.assertEqual(self.row()['status'],'downloaded')

    def test_checkpoint_item_mismatch_refuses_to_delete(self):
        path=self.add();self.checkpoint(lambda db:db.execute('UPDATE items SET duration=999'))
        with self.assertRaisesRegex(ValueError,'metadata mismatch'):self.run_evict(apply=True)
        self.assertTrue(path.exists());self.assertEqual(self.row()['status'],'downloaded')

    def test_checkpoint_checksum_mismatch_refuses_to_delete(self):
        path=self.add();cp=self.checkpoint();cp.write_bytes(cp.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint SHA256 mismatch'):self.run_evict(apply=True)
        self.assertTrue(path.exists())

    def test_remote_hash_mismatch_refuses_to_delete(self):
        path=self.add();self.checkpoint()
        next(iter(self.hub.metadata.values())).lfs['sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'remote media hash/size mismatch'):self.run_evict(apply=True)
        self.assertTrue(path.exists());self.assertEqual(self.row()['status'],'downloaded')

    def test_public_repo_refuses_to_delete(self):
        path=self.add();self.checkpoint();self.hub.private=False
        with self.assertRaisesRegex(ValueError,'private backup repository'):self.run_evict(apply=True)
        self.assertTrue(path.exists())

    def test_local_hash_mismatch_refuses_to_delete(self):
        path=self.add(payload=b'good');self.checkpoint();path.write_bytes(b'evil')
        with self.assertRaisesRegex(ValueError,'hash changed or mismatched'):self.run_evict(apply=True)
        self.assertTrue(path.exists());self.assertEqual(self.row()['status'],'downloaded')

    def test_crash_after_remote_commit_keeps_file_and_next_run_finishes(self):
        path=self.add();self.checkpoint()
        with patch('curation.evict.unlink_verified',side_effect=RuntimeError('simulated crash')):
            with self.assertRaisesRegex(RuntimeError,'simulated crash'):self.run_evict(apply=True)
        self.assertTrue(path.exists());self.assertEqual(self.row()['status'],'remote');self.assertIsNone(self.row()['path'])
        self.assertEqual(self.journal()[0]['state'],'prepared')
        report=self.run_evict(apply=True)
        self.assertEqual(report['evicted_files'],1);self.assertFalse(path.exists())
        self.assertEqual(self.journal()[0]['state'],'unlinked')

    def test_crash_after_unlink_reconciles_without_deleting_anything_else(self):
        path=self.add();self.checkpoint()
        def unlink_then_crash(p,*args):
            p.unlink();raise RuntimeError('after unlink')
        with patch('curation.evict.unlink_verified',side_effect=unlink_then_crash):
            with self.assertRaisesRegex(RuntimeError,'after unlink'):self.run_evict(apply=True)
        self.assertFalse(path.exists());self.assertEqual(self.journal()[0]['state'],'prepared')
        report=self.run_evict(apply=True)
        self.assertEqual(report['unlinked_bytes'],0);self.assertEqual(self.journal()[0]['state'],'unlinked')

    def test_restore_archives_prepared_eviction_journal_without_touching_original(self):
        path=self.add();payload=path.read_bytes();self.checkpoint()
        with patch('curation.evict.unlink_verified',side_effect=RuntimeError('before unlink')):
            with self.assertRaisesRegex(RuntimeError,'before unlink'):self.run_evict(apply=True)
        self.assertEqual(self.journal()[0]['state'],'prepared')
        self.assertTrue(path.exists())
        # Back up precisely the intermediate remote-committed, not-yet-unlinked state.
        self.checkpoint()
        fresh=self.base/'restored-after-prepared'
        summary=restore(fresh,receipt=self.receipt,max_files=1,min_free_gib=0,api=self.hub,downloader=self.hub.download)
        self.assertEqual(summary['status_counts'],{'downloaded':1})
        self.assertEqual(path.read_bytes(),payload)
        self.assertEqual(self.row()['status'],'remote')
        self.assertEqual(self.journal()[0]['state'],'prepared')
        with sqlite3.connect(fresh/'state.sqlite') as db:
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='eviction_journal'").fetchone())
            history=db.execute('SELECT payload FROM restore_eviction_history').fetchall()
            self.assertEqual(len(history),1)
            original=json.loads(history[0][0])
            self.assertEqual(original['original_path'],str(path))
            self.assertEqual(original['state'],'prepared')
            new_path=db.execute('SELECT path FROM items').fetchone()[0]
            self.assertTrue(Path(new_path).is_relative_to(fresh))
            self.assertEqual(Path(new_path).read_bytes(),payload)
        # Old-root paths are provenance only; a dry run on new-root sees its own file.
        plan=evict(fresh,api=self.hub,downloader=self.hub.download)
        self.assertEqual(plan['planned_files'],1)
        self.assertTrue(Path(plan['files'][0]['path']).is_relative_to(fresh))
        self.assertTrue(path.exists())

    def test_outside_root_path_and_symlink_are_never_deleted(self):
        path=self.add();self.checkpoint()
        outside=self.base/'outside.media';shutil.copyfile(path,outside)
        self.db.execute('UPDATE items SET path=?',(str(outside),));self.db.commit()
        with self.assertRaisesRegex(ValueError,'outside archive'):self.run_evict(apply=True)
        self.assertTrue(outside.exists())
        link=path.parent/'link.media';link.symlink_to(path)
        self.db.execute('UPDATE items SET path=?',(str(link),));self.db.commit()
        with self.assertRaisesRegex(ValueError,'symlink'):self.run_evict(apply=True)
        self.assertTrue(link.is_symlink());self.assertTrue(path.exists())

    def test_max_files_bytes_and_free_space_target_limit_plan(self):
        a=self.add('a',b'1234');b=self.add('b',b'56789');self.checkpoint()
        self.assertEqual(self.run_evict(max_files=1)['planned_files'],1)
        self.assertEqual(self.run_evict(max_files=2,max_bytes=3)['planned_files'],0)
        with patch('curation.evict.shutil.disk_usage',return_value=SimpleNamespace(free=1024**3)):
            self.assertEqual(self.run_evict(target_free_gib=.5)['planned_files'],0)
        self.assertTrue(a.exists());self.assertTrue(b.exists())

    def test_shared_local_file_transitions_all_proven_items_together(self):
        path=self.add('a',b'same');self.add('b',b'same',path=path);self.checkpoint()
        report=self.run_evict(apply=True,max_files=1)
        self.assertEqual(report['evicted_files'],1);self.assertFalse(path.exists())
        self.assertEqual(self.row('a')['status'],'remote');self.assertEqual(self.row('b')['status'],'remote')
        self.assertEqual(set(json.loads(self.journal()[0]['item_ids'])),{'a','b'})

    def test_no_backup_means_no_eligible_deletion(self):
        path=self.add(backed=False);self.checkpoint()
        report=self.run_evict(apply=True)
        self.assertEqual(report['evicted_files'],0);self.assertTrue(path.exists())
        self.assertEqual(self.row()['status'],'downloaded')


if __name__=='__main__':unittest.main()
