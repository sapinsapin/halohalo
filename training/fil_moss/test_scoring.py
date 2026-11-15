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

    def test_corpus_cer_pools_characters(self):
        self.assertAlmostEqual(S.corpus_cer([("ab", "ab"), ("abcdef", "abcxyz")]), 3 / 8)


class JudgeTests(unittest.TestCase):
    def test_worst_judge_takes_the_harshest(self):
        cers = {"whisper": 0.02, "mms": 0.10, "omni": 0.05}
        self.assertAlmostEqual(S.worst_judge(cers), 0.90)
        self.assertAlmostEqual(S.mean_judge(cers), 1 - (0.17 / 3))
        self.assertAlmostEqual(S.judge_gap(cers), 0.08)

    def test_gibberish_scores_zero_everywhere_and_closes_the_gap(self):
        cers = {"whisper": 1.4, "mms": 1.0, "omni": 2.0}
        self.assertEqual(S.worst_judge(cers), 0.0)
        self.assertEqual(S.mean_judge(cers), 0.0)


class RankTests(unittest.TestCase):
    def test_ranks_share_ties(self):
        self.assertEqual(S.ranks([3, 1, 3, 2]), [5 / 6, 0.0, 5 / 6, 1 / 3])
        self.assertEqual(S.ranks([7]), [0.5])

    def test_advantages_zero_without_spread(self):
        self.assertEqual(S.group_advantages([0.4, 0.4, 0.4]), [0.0, 0.0, 0.0])
        adv = S.group_advantages([0.0, 1.0])
        self.assertAlmostEqual(adv[0], -1.0)
        self.assertAlmostEqual(sum(adv), 0.0)

    def test_combine_is_scale_free_and_strict(self):
        a = S.combine_rewards({"x": [1, 2], "y": [10, 0]}, {"x": 1, "y": 1})
        b = S.combine_rewards({"x": [100, 200], "y": [0.1, 0.0]}, {"x": 1, "y": 1})
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            S.combine_rewards({"x": [1]}, {"x": 1, "y": 1})
        with self.assertRaises(ValueError):
            S.combine_rewards({"x": [1]}, {"x": 0})


class SignalTests(unittest.TestCase):
    def test_duration_penalty_symmetric(self):
        self.assertAlmostEqual(S.duration_penalty(2.0), S.duration_penalty(0.5))
        self.assertEqual(S.duration_penalty(1.0), 0.0)
        self.assertEqual(S.duration_penalty(0), float("inf"))

    def test_silence_fraction(self):
        sr = 1000
        tone = [math.sin(2 * math.pi * 50 * n / sr) for n in range(sr)]
        self.assertEqual(S.silence_fraction(tone + [0.0] * sr, sr), 0.5)
        self.assertEqual(S.silence_fraction([0.0] * 100, sr), 1.0)

    def test_token_entropy(self):
        self.assertEqual(S.token_entropy([5] * 10), 0.0)
        self.assertAlmostEqual(S.token_entropy([1, 2, 3, 4]), 2.0)


if __name__ == "__main__":
    unittest.main()
