import json
import unittest
from pathlib import Path

from training.fil_moss import gates as G
from training.fil_moss import grpo as P
from training.fil_moss import scoring as S

RECIPE = json.loads((Path(__file__).resolve().parent / "recipe.json").read_text())
CFG = RECIPE["stage3"]
ARMS = CFG["arms"]


def rollout(whisper, mms, omni, voice, ratio=1.0, clvi=None):
    return {"cers": {"whisper": whisper, "mms": mms, "omni": omni}, "voice_sim": voice, "clvi_sim": clvi,
            "duration_ratio": ratio}


def ev(worst, voice, balanced=4.5, cond="none"):
    m = {"worst_cer": worst, "balanced_validation_loss": balanced, "conditioning": cond, "judge_gap": 0.03}
    m["similarity_to_real_speaker" if cond == "reference" else "voice_consistency"] = voice
    return m


class RewardArithmeticTests(unittest.TestCase):
    def test_worst_judge_and_gibberish(self):
        cers = {"whisper": 0.02, "mms": 0.10, "omni": 0.05}
        self.assertAlmostEqual(S.worst_judge(cers), 0.90)
        self.assertAlmostEqual(S.mean_judge(cers), 1 - 0.17 / 3)
        self.assertEqual(S.worst_judge({"whisper": 1.4, "mms": 1.0}), 0.0)

    def test_ranks_and_advantages(self):
        self.assertEqual(S.ranks([3, 1, 3, 2]), [5 / 6, 0.0, 5 / 6, 1 / 3])
        self.assertEqual(S.ranks([7]), [0.5])
        self.assertEqual(S.group_advantages([0.4, 0.4, 0.4]), [0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(S.group_advantages([0.0, 1.0, 0.5])), 0.0)

    def test_combine_is_scale_free_and_strict(self):
        a = S.combine_rewards({"x": [1, 2], "y": [10, 0]}, {"x": 1, "y": 1})
        b = S.combine_rewards({"x": [100, 200], "y": [0.1, 0.0]}, {"x": 1, "y": 1})
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            S.combine_rewards({"x": [1]}, {"x": 1, "y": 1})

    def test_pacing_and_entropy(self):
        self.assertAlmostEqual(S.duration_penalty(2.0), S.duration_penalty(0.5))
        self.assertEqual(S.token_entropy([5] * 10), 0.0)
        self.assertAlmostEqual(S.token_entropy([1, 2, 3, 4]), 2.0)


class ArmTests(unittest.TestCase):
    def test_speaker_mode_follows_stage2(self):
        self.assertEqual(P.speaker_mode(ARMS["c"], "none"), "anchor")
        self.assertEqual(P.speaker_mode(ARMS["c"], "reference"), "conditioning")
        self.assertEqual(P.speaker_mode(ARMS["d"], "reference"), "conditioning+clvi")
        with self.assertRaises(ValueError):
            P.speaker_mode(ARMS["d"], "none")

    def test_arm_a_trusts_whisper_only_and_c_does_not(self):
        group = [rollout(0.01, 0.40, 0.40, 0.8), rollout(0.05, 0.06, 0.06, 0.8)]  # 0 games the in-family judge
        ra, _, _ = P.group_rewards(ARMS["a"], "anchor", CFG, group)
        rc, _, _ = P.group_rewards(ARMS["c"], "anchor", CFG, group)
        self.assertGreater(ra[0], ra[1])
        self.assertLess(rc[0], rc[1])

    def test_voice_and_pacing_terms(self):
        group = [rollout(0.05, 0.05, 0.05, 0.9, 1.0), rollout(0.05, 0.05, 0.05, 0.6, 1.0),
                 rollout(0.05, 0.05, 0.05, 0.9, 2.0)]
        rewards, adv, _ = P.group_rewards(ARMS["c"], "anchor", CFG, group)
        self.assertEqual(max(range(3), key=lambda i: rewards[i]), 0)
        _, flat, _ = P.group_rewards(ARMS["c"], "anchor", CFG, [rollout(0.05, 0.05, 0.05, 0.8)] * 3)
        self.assertEqual(flat, [0.0, 0.0, 0.0])

    def test_clvi_only_in_its_mode(self):
        split = CFG["clvi"]["speaker_split"]
        self.assertAlmostEqual(P.speaker_value("conditioning+clvi", 0.8, 0.6, split), 0.7)
        with self.assertRaises(ValueError):
            P.speaker_value("anchor", 0.8, 0.6, split)

    def test_prompt_filter(self):
        kept, dropped = P.filter_prompts(
            ["Magandang umaga sa inyong lahat", "", "May 3 aso kami", "ok", "Magandang umaga sa inyong lahat!",
             "Salamat po sa tulong ninyo kahapon"], ["salamat po sa tulong ninyo kahapon"])
        self.assertEqual(kept, ["Magandang umaga sa inyong lahat"])
        self.assertEqual(dropped, {"empty": 1, "digits": 1, "length": 1, "duplicate": 1, "heldout_transcript": 1})

    def test_pacing_helpers(self):
        rate = P.speaking_rate([{"text": "abcd efgh", "seconds": 1.0}, {"text": "ab", "seconds": 1.0}])
        self.assertAlmostEqual(rate, 11 / 2)
        self.assertEqual(P.expected_seconds("a", 100.0), 0.5)
        hist = [{"reward": i, "silence_fraction": 0.0, "token_entropy": 1.0} for i in range(6)]
        self.assertIsNone(P.windows(hist[:3], 2))
        self.assertEqual(P.windows(hist, 2)[1]["reward"], 4.5)


class Stage3GateTests(unittest.TestCase):
    def s2(self, **kw):
        rep = {"status": "complete", "stage1_smoke": {"verdict": G.PASS}, "codes": "/c",
               "best_checkpoint": "/run/stage2/best_checkpoint", "conditioning": "none"}
        rep.update(kw)
        return rep

    def test_entry(self):
        self.assertEqual(G.stage3_entry(self.s2(), "/c")["verdict"], G.PASS)
        pretrained = G.stage3_entry(self.s2(best_checkpoint="/models/MOSS-TTS-Realtime"), "/c")
        self.assertEqual(pretrained["verdict"], G.FAIL)
        self.assertEqual(G.stage3_entry(self.s2(status="running"), "/c")["verdict"], G.FAIL)
        self.assertEqual(G.stage3_entry(self.s2(), "/other")["verdict"], G.FAIL)

    def test_kills_are_relative_to_step_zero(self):
        base = ev(0.10, 0.80)
        self.assertEqual(G.stage3_evaluation_verdict(ev(0.08, 0.81), 100, base, CFG)["verdict"], G.CONTINUE)
        self.assertEqual(G.stage3_evaluation_verdict(ev(0.08, 0.77), 100, base, CFG)["verdict"], G.KILL)
        self.assertEqual(G.stage3_evaluation_verdict(ev(0.08, 0.80, balanced=5.1), 100, base, CFG)["verdict"], G.KILL)
        self.assertEqual(G.stage3_evaluation_verdict(ev(0.10, 0.80), 300, base, CFG)["verdict"], G.KILL)
        ref = ev(0.1, 0.70, cond="reference")
        self.assertEqual(G.voice_measure(ref), 0.70)

    def test_clvi_descope_and_tripwire(self):
        self.assertEqual(G.clvi_mode(10, 0.8, CFG)["verdict"], "eval_only")
        self.assertEqual(G.clvi_mode(60, None, CFG)["verdict"], "eval_only")
        self.assertEqual(G.clvi_mode(60, 0.5, CFG)["verdict"], "dropped")
        self.assertEqual(G.clvi_mode(60, 0.65, CFG)["verdict"], "reward")
        first = {"reward": 0.5, "silence_fraction": 0.1, "token_entropy": 8.0}
        self.assertEqual(G.tripwire(first, dict(first, reward=0.6, silence_fraction=0.3), CFG)["verdict"], G.KILL)
        self.assertEqual(G.tripwire(first, dict(first, reward=0.6, token_entropy=5.0), CFG)["verdict"], G.KILL)
        self.assertEqual(G.tripwire(first, dict(first, reward=0.4, silence_fraction=0.3), CFG)["verdict"], G.CONTINUE)

    def test_adoption(self):
        common = {"start_worst_cer": 0.10, "start_voice": 0.80, "steps": 1000, "aborts": 0}
        b = dict(common, worst_cer=0.060, voice=0.80, judge_gap=0.03)
        c = dict(common, worst_cer=0.055, voice=0.805, judge_gap=0.02, xling_sim=0.62)
        d = dict(common, worst_cer=0.054, voice=0.79, judge_gap=0.02, xling_sim=0.72, start_xling_sim=0.60)
        out = G.arm_decisions({"b": b, "c": c, "d": d}, CFG)
        self.assertEqual(out["c_over_b"]["verdict"], G.PASS)
        self.assertEqual(out["d_over_c"]["verdict"], G.PASS)
        self.assertEqual(out["b"]["verdict"], G.PASS)
        flat = dict(c, worst_cer=0.061)
        self.assertEqual(G.arm_decisions({"b": b, "c": flat}, CFG)["c_over_b"]["verdict"], G.KILL)
        self.assertEqual(G.arm_decisions({"b": dict(b, aborts=2)}, CFG)["b"]["verdict"], G.KILL)
        self.assertEqual(G.arm_decisions({"a": dict(b, worst_cer=0.11)}, CFG)["a"]["verdict"], G.FAIL)


if __name__ == "__main__":
    unittest.main()
