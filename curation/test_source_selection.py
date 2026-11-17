"""Offline source selection tests; no network or running collector changes."""
import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.sax.saxutils import escape

from curation import collect


class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(root=self.root / 'rss', sources=self.root / 'sources.json')
        self.source = dict(id='mixed', url='https://example.org/feed', language='ilo',
                           include_title_regex='(?i)ilocano|ilokano')

    def discover(self, titles):
        self.args.sources.write_text(json.dumps([self.source]))
        body = ('<rss><channel>' + ''.join(
            f'<item><guid>{i}</guid><title>{escape(title)}</title>'
            f'<enclosure url="https://example.org/{i}.mp3"/></item>'
            for i, title in enumerate(titles)) + '</channel></rss>').encode()
        response = io.BytesIO(body)
        response.url = self.source['url']
        with patch.object(collect, 'request', return_value=response), contextlib.redirect_stdout(io.StringIO()):
            collect.discover(self.args)

    def rows(self):
        db = collect.connect(self.args.root)
        try:
            return [dict(r) for r in db.execute('SELECT * FROM items ORDER BY guid')]
        finally:
            db.close()

    def test_matching_selected_and_other_new_titles_excluded_with_reason(self):
        self.discover(['History ILOCANO', 'History Tagalog', 'Ilokano conversation'])
        rows = self.rows()
        self.assertEqual([r['status'] for r in rows], ['pending', 'excluded', 'pending'])
        self.assertIn('include_title_regex', rows[1]['error'])

    def test_rediscovery_cannot_resurrect_exclusion(self):
        self.discover(['Tagalog'])
        self.discover(['Now explicitly Ilocano'])
        self.assertEqual(self.rows()[0]['status'], 'excluded')
        self.assertIsNotNone(self.rows()[0]['error'])

    def test_new_policy_excludes_pending_but_preserves_completed_and_inflight(self):
        self.source.pop('include_title_regex')
        self.discover(['Tagalog'] * 4)
        db = collect.connect(self.args.root)
        for guid, status in [('1', 'downloaded'), ('2', 'remote'), ('3', 'downloading')]:
            db.execute('UPDATE items SET status=? WHERE guid=?', (status, guid))
        db.commit(); db.close()
        self.source['include_title_regex'] = '(?i)ilocano|ilokano'
        self.discover(['Tagalog'] * 4)
        self.assertEqual([r['status'] for r in self.rows()], ['excluded', 'downloaded', 'remote', 'downloading'])

    def test_bad_regex_rejected_before_any_source_mutation_or_request(self):
        self.discover(['Ilocano'])
        db = collect.connect(self.args.root)
        before = tuple(db.execute('SELECT * FROM sources').fetchone())
        db.close()
        self.source.update(include_title_regex='[', description='must not persist')
        self.args.sources.write_text(json.dumps([self.source]))
        with patch.object(collect, 'request') as request, self.assertRaises(re.error):
            collect.discover(self.args)
        request.assert_not_called()
        db = collect.connect(self.args.root)
        self.assertEqual(tuple(db.execute('SELECT * FROM sources').fetchone()), before)
        db.close()

    def test_sources_without_selector_keep_existing_behavior(self):
        self.source.pop('include_title_regex')
        self.discover(['Tagalog', 'English'])
        self.assertEqual([r['status'] for r in self.rows()], ['pending', 'pending'])


if __name__ == '__main__':
    unittest.main()
