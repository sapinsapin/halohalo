import math
import unittest

from training.fil_moss import scoring as S


class TextTests(unittest.TestCase):
    def test_normalise_matches_halolib_rule(self):
        self.assertEqual(S.normalise_text("Ganína, IHÚNONG!  mo’y  di'"), "ganina ihunong mo'y di'")

    def test_normalise_identical_to_halolib_when_importable(self):
        try:
            from halolib.finetune import normalise_text
        except Exception:
            self.skipTest("halolib needs datasets")
        for t in ("Kumusta ka, Ma’am?", "Ánó ba 'yan", "OK lang—talaga!"):
            self.assertEqual(S.normalise_text(t), normalise_text(t))

    def test_cer_counts_after_normalisation(self):
        self.assertEqual(S.cer("Salamat po!", "salamat po"), 0.0)
        self.assertAlmostEqual(S.cer("abcd", "abed"), 0.25)
        with self.assertRaises(ValueError):
            S.cer("...", "x")

    def test_corpus_and_grouped_cer(self):
        self.assertAlmostEqual(S.corpus_cer([("ab", "ab"), ("abcdef", "abcxyz")]), 3 / 8)
        rows = [{"g": "cs", "ref": "ab", "hyp": "ab"}, {"g": "mono", "ref": "abcd", "hyp": "abcx"}]
        self.assertEqual(S.grouped_cer(rows, lambda r: r["g"]), {"cs": 0.0, "mono": 0.25})

    def test_judge_gap(self):
        self.assertAlmostEqual(S.judge_gap({"whisper": 0.02, "mms": 0.10, "omni": 0.05}), 0.08)


class SignalTests(unittest.TestCase):
    def test_silence_fraction(self):
        sr = 1000
        tone = [math.sin(2 * math.pi * 50 * n / sr) for n in range(sr)]
        self.assertEqual(S.silence_fraction(tone + [0.0] * sr, sr), 0.5)
        self.assertEqual(S.silence_fraction([0.0] * 100, sr), 1.0)

    def test_median_and_consistency(self):
        self.assertEqual(S.median([3, 1, 2]), 2)
        self.assertEqual(S.median([4, 1, 2, 3]), 2.5)
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))
        self.assertAlmostEqual(S.mean_pairwise([(1, 0), (1, 0), (0, 1)], dot), 1 / 3)
        self.assertIsNone(S.mean_pairwise([(1, 0)], dot))


if __name__ == "__main__":
    unittest.main()
