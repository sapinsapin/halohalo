import tempfile
import unittest
from pathlib import Path

from training.fil_moss import guard


class GuardTests(unittest.TestCase):
    def test_should_stop(self):
        self.assertEqual(guard.should_stop(10 * guard.GIB, 1, 25 * guard.GIB, 100), "memory floor")
        self.assertEqual(guard.should_stop(30 * guard.GIB, 101, 25 * guard.GIB, 100), "time bound")
        self.assertIsNone(guard.should_stop(30 * guard.GIB, 1, 25 * guard.GIB, 100))

    def test_meminfo(self):
        with tempfile.NamedTemporaryFile("w", suffix="meminfo", delete=False) as f:
            f.write("MemTotal: 100 kB\nMemAvailable:   2048 kB\n")
        self.assertEqual(guard.mem_available_bytes(f.name), 2048 * 1024)
        Path(f.name).unlink()


if __name__ == "__main__":
    unittest.main()
