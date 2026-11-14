import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from curation import archive_checkpoint as cp
from curation import acquire_archives as archives


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_is_deterministic_and_excludes_audio_and_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); root = tmp/'archive_sources'; root.mkdir()
            (root/'inventory.json').write_text('{"entries":[{"id":"example"}]}')
            (root/'clip.flac').write_bytes(b'not metadata')
            (root/'unfinished.tmp.json').write_text('{}')
            (root/'partial.json.tmp').write_text('{')
            a, b = tmp/'a.tar.gz', tmp/'b.tar.gz'
            report = cp.pack([root], a); cp.pack([root], b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(report['files'], 1)
            restored = tmp/'restored'; cp.restore(a, restored)
            self.assertEqual((root/'inventory.json').read_bytes(), (restored/'archive_sources/inventory.json').read_bytes())
            self.assertFalse((restored/'archive_sources/clip.flac').exists())
            with self.assertRaises(ValueError): cp.restore(a, restored)

    def test_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); root=tmp/'root'; root.mkdir()
            (tmp/'outside.json').write_text('{}')
            (root/'linked.json').symlink_to(tmp/'outside.json')
            with self.assertRaises(ValueError): cp.pack([root], tmp/'bundle.tar.gz')

    def test_corrupt_metadata_rejected_before_bundle_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);root=tmp/'root';root.mkdir();(root/'bad.json').write_text('{')
            with self.assertRaises(json.JSONDecodeError):cp.pack([root],tmp/'bundle.tar.gz')
            self.assertFalse((tmp/'bundle.tar.gz').exists())

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'evil.tar.gz'
            with tarfile.open(p,'w:gz') as archive:
                item=tarfile.TarInfo('../outside.json');item.size=2;archive.addfile(item,io.BytesIO(b'{}'))
            with self.assertRaises(ValueError):cp.verify(p)

    def test_modified_payload_fails_manifest_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);root=tmp/'root';root.mkdir();(root/'metadata.json').write_text('{}')
            cp.pack([root],tmp/'good.tar.gz')
            with tarfile.open(tmp/'good.tar.gz','r:gz') as source,tarfile.open(tmp/'bad.tar.gz','w:gz') as target:
                for member in source.getmembers():
                    data=source.extractfile(member).read()
                    if member.name=='root/metadata.json':data=b'[]'
                    target.addfile(member,io.BytesIO(data))
            with self.assertRaises(ValueError):cp.verify(tmp/'bad.tar.gz')

    def test_upcoming_video_is_exclusion_not_download_failure(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(archives,'ROOT',Path(tmp)),patch.object(archives,'free_check'):
            with patch.object(archives.subprocess,'run') as run:
                row=archives.acquire_youtube({'id':'futurevideo','live_status':'is_upcoming'},'example')
            run.assert_not_called()
            self.assertEqual(row['status'],'excluded')
            self.assertEqual(row['exclusion_reason'],'ongoing_or_upcoming_live')


if __name__=='__main__':unittest.main()
