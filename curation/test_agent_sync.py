import tempfile
import unittest
from unittest import mock

from . import agent_sync as s


class AgentSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_claim_blocks_other_agent_until_release(self):
        self.assertTrue(s.claim('gpu:dwnd', 'claude', 'ceb', 1, self.root)[0])
        ok, holder = s.claim('gpu:dwnd', 'antigravity', 'x', 1, self.root)
        self.assertFalse(ok)
        self.assertEqual(holder['agent'], 'claude')
        self.assertFalse(s.release('gpu:dwnd', 'antigravity', root=self.root)[0])
        self.assertTrue(s.release('gpu:dwnd', 'claude', root=self.root)[0])
        self.assertTrue(s.claim('gpu:dwnd', 'antigravity', 'x', 1, self.root)[0])

    def test_expired_claim_is_free_and_renew_keeps_since(self):
        s.claim('hub:sapinsapin/chika', 'claude', '', 1, self.root)
        with mock.patch.object(s, 'now', return_value=s.time.time() + 7200):
            self.assertTrue(s.claim('hub:sapinsapin/chika', 'antigravity', '', 1, self.root)[0])
        ok, first = s.claim('data:x', 'claude', '', 1, self.root)
        ok, again = s.claim('data:x', 'claude', 'renew', 2, self.root)
        self.assertEqual(first['since'], again['since'])
        self.assertGreater(again['expires'], first['expires'])


if __name__ == '__main__':
    unittest.main()
