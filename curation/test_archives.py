import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from curation import acquire_archives as archives


class ArchiveTests(unittest.TestCase):
    def test_bombo_section_does_not_include_other_languages(self):
        parser = archives.BomboIlonggoParser()
        parser.feed('''<h4>TAGALOG DRAMA</h4><a data-video-id="excluded1"></a>
            <h4><span>ILONGGO DRAMA</span></h4>
            <a data-video-id="included"><div class="td_video_title">A &amp; B</div></a>
            <h4>CEBUANO DRAMA</h4><a data-video-id="excluded2"></a>''')
        self.assertTrue(parser.found_section)
        self.assertEqual(parser.entries, [{'id':'included','title':'A & B'}])

    def test_missing_inventory_cannot_finish_successfully(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(archives, 'ROOT', Path(tmp)):
            with contextlib.redirect_stdout(io.StringIO()):
                report = archives.queue_all()
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(len(report['incomplete_sources']), len(archives.YOUTUBE_SOURCES))
            persisted = json.loads((Path(tmp)/'archive_queue_checkpoint.json').read_text())
            self.assertEqual(persisted, report)

    def test_partial_inventory_processes_available_source_but_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(archives, 'ROOT', Path(tmp)):
            archives.atomic(Path(tmp)/'ilonggatalks_inventory.json', {'entries':[{'id':'example'}]})
            with patch.object(archives, 'free_check'), patch.object(archives, 'acquire_youtube', return_value={'status':'downloaded'}) as download:
                with contextlib.redirect_stdout(io.StringIO()):
                    report = archives.queue_all()
            download.assert_called_once_with({'id':'example'}, 'ilonggatalks')
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['sources']['ilonggatalks']['processed'], 1)

    def test_discovery_continues_after_source_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(archives, 'ROOT', Path(tmp)):
            with patch.object(archives, 'discover_deditos', side_effect=ValueError('page changed')), \
                 patch.object(archives, 'discover_bombo') as bombo, \
                 patch.object(archives, 'discover_kuan') as kuan, \
                 patch.object(archives, 'discover_gma'), patch.object(archives, 'discover_channels'):
                report = archives.discover()
            bombo.assert_called_once()
            kuan.assert_called_once()
            self.assertEqual(report['deditos_inventory']['status'], 'discovery_failed')
            self.assertTrue((Path(tmp)/'deditos_inventory.error.json').exists())

    def test_executable_prefers_path(self):
        with patch.object(archives.shutil, 'which', return_value='/usr/bin/ffmpeg'):
            self.assertEqual(archives.executable('ffmpeg'), '/usr/bin/ffmpeg')


if __name__ == '__main__':
    unittest.main()
