import copy
import json
import unittest
from pathlib import Path

from training.fil_moss import gates as G

RECIPE = json.loads((Path(__file__).resolve().parent / "recipe.json").read_text())
V, T, S2 = RECIPE["stage1"]["verify"], RECIPE["stage1"]["train"], RECIPE["stage2"]


def metrics(**kw):
    m = {"worst_cer": 0.06, "cer": {"mms": 0.06, "whisper": 0.03, "omni": 0.05}, "spk_sim": 0.80,
         "duration_ratio": 1.0, "english_worst_cer": 0.10}
    m.update(kw)
    return m


class VerifyTests(unittest.TestCase):
    def test_parity(self):
        self.assertEqual(G.parity_check(2.0, 2.0005, V["parity_rtol"])["verdict"], G.PASS)
        self.assertEqual(G.parity_check(2.0, 2.1, V["parity_rtol"])["verdict"], G.FAIL)
        self.assertEqual(G.parity_check(float("nan"), 2.0, V["parity_rtol"])["verdict"], G.FAIL)

    def test_overfit_needs_low_loss_and_reference_pace(self):
        self.assertEqual(G.overfit_check(4.1, 0.2, [1.0, 1.03, 0.97], V)["verdict"], G.PASS)
        self.assertEqual(G.overfit_check(4.1, 2.0, [1.0], V)["verdict"], G.FAIL)
        never_stops = G.overfit_check(4.1, 0.2, [1.0, 2.5], V)
        self.assertEqual(never_stops["verdict"], G.FAIL)
        self.assertIn("1 of 2 clips", never_stops["reasons"][0])
        self.assertEqual(G.overfit_check(4.1, 0.2, [], V)["verdict"], G.FAIL)

    def test_verification_needs_all_three(self):
        ok = {"verdict": G.PASS, "reasons": []}
        bad = {"verdict": G.FAIL, "reasons": ["x"]}
        self.assertEqual(G.verification_verdict(ok, ok, True)["verdict"], G.PASS)
        self.assertEqual(G.verification_verdict(ok, ok, False)["verdict"], G.FAIL)
        self.assertEqual(G.verification_verdict(bad, ok, True)["verdict"], G.FAIL)


class SftTests(unittest.TestCase):
    def test_duration_kill(self):
        self.assertEqual(G.sft_evaluation_verdict(metrics(duration_ratio=1.4), metrics(), 0.5, T)["verdict"], G.KILL)
        warn = G.sft_evaluation_verdict(metrics(duration_ratio=1.2), metrics(), 0.5, T)
        self.assertEqual(warn["verdict"], G.CONTINUE)
        self.assertTrue(warn["reasons"])

    def test_worse_than_zero_shot_only_kills_after_the_epochs(self):
        zero = metrics(worst_cer=0.08)
        self.assertEqual(G.sft_evaluation_verdict(metrics(worst_cer=0.09), zero, 1.0, T)["verdict"], G.CONTINUE)
        self.assertEqual(G.sft_evaluation_verdict(metrics(worst_cer=0.09), zero, 2.0, T)["verdict"], G.KILL)

    def test_success_thresholds(self):
        zero = metrics()
        self.assertEqual(G.sft_success(metrics(), zero, T)["verdict"], G.PASS)
        self.assertEqual(G.sft_success(metrics(cer={"mms": 0.09}), zero, T)["verdict"], G.FAIL)
        self.assertEqual(G.sft_success(metrics(spk_sim=0.70), zero, T)["verdict"], G.FAIL)
        self.assertEqual(G.sft_success(metrics(duration_ratio=1.1), zero, T)["verdict"], G.FAIL)
        self.assertEqual(G.sft_success(metrics(english_worst_cer=0.2), zero, T)["verdict"], G.FAIL)

    def test_missing_measurement_is_incomplete_not_pass(self):
        m = metrics()
        del m["english_worst_cer"]
        self.assertEqual(G.sft_success(m, metrics(), T)["verdict"], G.INCOMPLETE)

    def test_checkpoint_selection(self):
        self.assertTrue(G.better_checkpoint(metrics(), None))
        self.assertTrue(G.better_checkpoint(metrics(worst_cer=0.05), metrics()))
        self.assertFalse(G.better_checkpoint(metrics(worst_cer=0.07), metrics()))
        self.assertTrue(G.better_checkpoint(metrics(spk_sim=0.9), metrics()))


class Stage2Tests(unittest.TestCase):
    def passing_stage1(self):
        return {"verification": {"verdict": G.PASS}, "success": {"verdict": G.PASS}, "best_checkpoint": "/x"}

    def test_entry_requires_stage1_pass(self):
        self.assertEqual(G.stage2_entry(self.passing_stage1())["verdict"], G.PASS)
        s = self.passing_stage1()
        s["success"]["verdict"] = G.INCOMPLETE
        self.assertEqual(G.stage2_entry(s)["verdict"], G.FAIL)
        self.assertEqual(G.stage2_entry(None)["verdict"], G.FAIL)

    def test_clvi_descope(self):
        self.assertEqual(G.clvi_mode(10, 0.8, S2)["verdict"], "eval_only")
        self.assertEqual(G.clvi_mode(60, None, S2)["verdict"], "eval_only")
        self.assertEqual(G.clvi_mode(60, 0.5, S2)["verdict"], "dropped")
        self.assertEqual(G.clvi_mode(60, 0.65, S2)["verdict"], "reward")

    def test_tripwire_fires_only_when_reward_rises(self):
        first = {"reward": 0.5, "silence_fraction": 0.1, "token_entropy": 8.0}
        hacked = {"reward": 0.6, "silence_fraction": 0.3, "token_entropy": 8.0}
        collapsed = {"reward": 0.6, "silence_fraction": 0.1, "token_entropy": 5.0}
        self.assertEqual(G.tripwire(first, hacked, S2)["verdict"], G.KILL)
        self.assertEqual(G.tripwire(first, collapsed, S2)["verdict"], G.KILL)
        worse = dict(hacked, reward=0.4)
        self.assertEqual(G.tripwire(first, worse, S2)["verdict"], G.CONTINUE)

    def test_stage2_kills(self):
        base = {"worst_cer": 0.06, "spk_sim": 0.8}
        self.assertEqual(G.stage2_evaluation_verdict({"worst_cer": 0.05, "spk_sim": 0.7}, 100, base, S2)["verdict"],
                         G.KILL)
        self.assertEqual(G.stage2_evaluation_verdict({"worst_cer": 0.06, "spk_sim": 0.8}, 300, base, S2)["verdict"],
                         G.KILL)
        self.assertEqual(G.stage2_evaluation_verdict({"worst_cer": 0.05, "spk_sim": 0.8}, 300, base, S2)["verdict"],
                         G.CONTINUE)

    def test_arm_adoption(self):
        b = {"worst_cer": 0.060, "spk_sim": 0.80, "judge_gap": 0.03, "steps": 1000}
        c = {"worst_cer": 0.055, "spk_sim": 0.805, "judge_gap": 0.02, "steps": 1000, "xling_sim": 0.62}
        d = {"worst_cer": 0.054, "spk_sim": 0.79, "judge_gap": 0.02, "steps": 1000, "xling_sim": 0.72}
        out = G.arm_decisions({"b": b, "c": c, "d": d}, S2)
        self.assertEqual(out["c_over_b"]["verdict"], G.PASS)
        self.assertEqual(out["d"]["verdict"], G.PASS)
        flat = dict(c, worst_cer=0.061)
        self.assertEqual(G.arm_decisions({"b": b, "c": flat}, S2)["c_over_b"]["verdict"], G.KILL)
        dropped_sim = dict(d, spk_sim=0.75)
        self.assertEqual(G.arm_decisions({"c": c, "d": dropped_sim}, S2)["d"]["verdict"], G.FAIL)
        aborted = dict(b, aborts=2)
        self.assertEqual(G.arm_decisions({"b": aborted}, S2)["b"]["verdict"], G.KILL)


class RecipeTests(unittest.TestCase):
    def test_recipe_keys_the_gates_read(self):
        r = copy.deepcopy(RECIPE)
        self.assertEqual(set(r["stage2"]["arms"]), {"a", "b", "c", "d"})
        self.assertEqual(set(r["stage2"]["reward_weights"]), {"intelligibility", "speaker", "duration"})
        self.assertIn(r["stage1"]["train"]["success"]["primary_judge"], r["judges"]["fil"])
        self.assertEqual(r["status"], "authorized_pending_data")


if __name__ == "__main__":
    unittest.main()
