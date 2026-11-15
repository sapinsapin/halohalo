import json
import tempfile
import unittest
from pathlib import Path

from training.fil_moss import grpo as P
from training.fil_moss import guard

RECIPE = json.loads((Path(__file__).resolve().parent / "recipe.json").read_text())
CFG = RECIPE["stage2"]


def rollout(whisper, mms, omni, sim, ratio=1.0, clvi=None):
    return {"cers": {"whisper": whisper, "mms": mms, "omni": omni}, "cond_sim": sim, "clvi_sim": clvi,
            "duration_ratio": ratio}


class PromptTests(unittest.TestCase):
    def test_filter(self):
        kept, dropped = P.filter_prompts(
            ["Magandang umaga sa inyong lahat", "", "May 3 aso kami", "ok", "Magandang umaga sa inyong lahat!",
             "Salamat po sa tulong ninyo kahapon"],
            ["salamat po sa tulong ninyo kahapon"])
        self.assertEqual(kept, ["Magandang umaga sa inyong lahat"])
        self.assertEqual(dropped, {"empty": 1, "digits": 1, "length": 1, "duplicate": 1, "heldout_transcript": 1})


class RewardTests(unittest.TestCase):
    def test_arm_a_trusts_whisper_only_and_c_does_not(self):
        # rollout 0 games the in-family judge; rollout 1 is honest
        group = [rollout(0.01, 0.40, 0.40, 0.8), rollout(0.05, 0.06, 0.06, 0.8)]
        ra, _, _ = P.group_rewards(CFG["arms"]["a"], CFG, group)
        rc, _, _ = P.group_rewards(CFG["arms"]["c"], CFG, group)
        self.assertGreater(ra[0], ra[1])
        self.assertLess(rc[0], rc[1])

    def test_duration_and_speaker_terms(self):
        group = [rollout(0.05, 0.05, 0.05, 0.8, 1.0), rollout(0.05, 0.05, 0.05, 0.8, 2.0)]
        rewards, adv, _ = P.group_rewards(CFG["arms"]["c"], CFG, group)
        self.assertGreater(rewards[0], rewards[1])
        self.assertGreater(adv[0], 0)
        same, adv0, _ = P.group_rewards(CFG["arms"]["c"], CFG, [rollout(0.05, 0.05, 0.05, 0.8)] * 3)
        self.assertEqual(adv0, [0.0, 0.0, 0.0])

    def test_clvi_mixes_only_in_arm_d(self):
        split = CFG["clvi"]["speaker_split"]
        self.assertAlmostEqual(P.speaker_for_arm(CFG["arms"]["d"], 0.8, 0.6, split), 0.7)
        with self.assertRaises(ValueError):
            P.speaker_for_arm(CFG["arms"]["c"], 0.8, 0.6, split)
        self.assertEqual(P.speaker_for_arm(CFG["arms"]["d"], 0.8, None, split), 0.8)

    def test_unknown_judge_for_arm_a(self):
        with self.assertRaises(KeyError):
            P.intelligibility_for_arm(CFG["arms"]["a"], {"mms": 0.1})


class PacingTests(unittest.TestCase):
    def test_rate_and_expected(self):
        rate = P.speaking_rate([{"text": "abcd efgh", "seconds": 1.0}, {"text": "ab", "seconds": 1.0}])
        self.assertAlmostEqual(rate, 11 / 2)
        self.assertAlmostEqual(P.expected_seconds("abcdefghijk", rate), 2.0)
        self.assertEqual(P.expected_seconds("a", 100.0), 0.5)

    def test_windows(self):
        hist = [{"reward": i, "silence_fraction": 0.0, "token_entropy": 1.0} for i in range(6)]
        self.assertIsNone(P.windows(hist[:3], 2))
        first, last = P.windows(hist, 2)
        self.assertEqual((first["reward"], last["reward"]), (0.5, 4.5))


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
