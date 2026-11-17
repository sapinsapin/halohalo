import json
import unittest
from pathlib import Path

from training.fil_moss import gates as G

RECIPE = json.loads((Path(__file__).resolve().parent / "recipe.json").read_text())
S1 = RECIPE["stage1"]
OK = {"verdict": G.PASS, "reasons": []}


def ev(balanced, domains, worst=0.1, cs=0.12):
    return {"balanced_validation_loss": balanced, "domain_losses": domains, "cer": {"mms": worst, "whisper": worst / 2},
            "worst_cer": worst, "code_switched_worst_cer": cs}


class Stage1Tests(unittest.TestCase):
    def test_parity(self):
        self.assertEqual(G.parity_check(2.0, 2.0005, S1["parity_rtol"])["verdict"], G.PASS)
        self.assertEqual(G.parity_check(2.0, 2.1, S1["parity_rtol"])["verdict"], G.FAIL)
        self.assertEqual(G.parity_check(float("nan"), 2.0, S1["parity_rtol"])["verdict"], G.FAIL)

    def test_batch_parity(self):
        self.assertEqual(G.batch_parity_check([-2.0, -3.0], [-2.01, -2.99], 0.02)["verdict"], G.PASS)
        self.assertEqual(G.batch_parity_check([-2.0, -3.0], [-2.5, -3.0], 0.02)["verdict"], G.FAIL)
        self.assertEqual(G.batch_parity_check([-2.0], [], 0.02)["verdict"], G.FAIL)

    def test_one_step(self):
        good = {"loss": 4.1, "grad_norm": 1.2, "grads_fp32": True, "linear_output_dtype": "torch.bfloat16",
                "probe_changed_elements": 4000}
        self.assertEqual(G.one_step_check(good)["verdict"], G.PASS)
        self.assertEqual(G.one_step_check(dict(good, linear_output_dtype="torch.float32"))["verdict"], G.FAIL)
        self.assertEqual(G.one_step_check(dict(good, probe_changed_elements=0))["verdict"], G.FAIL)
        self.assertEqual(G.one_step_check(dict(good, loss=float("inf")))["verdict"], G.FAIL)

    def test_reload_and_samples(self):
        self.assertEqual(G.reload_check(True, 4.0, 4.0)["verdict"], G.PASS)
        self.assertEqual(G.reload_check(True, 4.0, 4.01)["verdict"], G.FAIL)
        self.assertEqual(G.reload_check(False, 4.0, 4.0)["verdict"], G.FAIL)
        s = {"name": "a", "finite": True, "silence_fraction": 0.2, "truncated": True}
        r = G.sample_check([s])
        self.assertEqual(r["verdict"], G.PASS)
        self.assertEqual(r["truncated"], ["a"])
        self.assertEqual(G.sample_check([dict(s, silence_fraction=0.99)])["verdict"], G.FAIL)
        self.assertEqual(G.sample_check([])["verdict"], G.FAIL)

    def test_smoke_needs_every_check(self):
        self.assertEqual(G.smoke_verdict({"a": OK, "b": OK})["verdict"], G.PASS)
        self.assertEqual(G.smoke_verdict({"a": OK, "b": {"verdict": G.FAIL, "reasons": ["x"]}})["verdict"], G.FAIL)


class Stage2Tests(unittest.TestCase):
    def test_balanced_loss_weights_domains_equally(self):
        self.assertAlmostEqual(G.balanced_validation_loss({"bible": 4.7, "pld": 5.0}), 4.85)
        with self.assertRaises(ValueError):
            G.balanced_validation_loss({})

    def test_selection_reproduces_the_pilot(self):
        # pilot: PLD 5.251 -> 5.029, Bible 3.874 -> 4.705; the step-0 checkpoint stayed best
        zero = ev(G.balanced_validation_loss({"pld": 5.25098, "bible": 3.87360}), {"pld": 5.25098, "bible": 3.87360})
        after = ev(G.balanced_validation_loss({"pld": 5.02941, "bible": 4.70450}), {"pld": 5.02941, "bible": 4.70450})
        self.assertFalse(G.better_checkpoint(after, zero))
        self.assertTrue(G.better_checkpoint(after, None))

    def test_entry(self):
        rep = {"smoke": OK, "codes": "/c"}
        self.assertEqual(G.stage2_entry(rep, "/c")["verdict"], G.PASS)
        self.assertEqual(G.stage2_entry(rep, "/other")["verdict"], G.FAIL)
        self.assertEqual(G.stage2_entry(dict(rep, smoke={"verdict": G.FAIL}), "/c")["verdict"], G.FAIL)
        self.assertEqual(G.stage2_entry(None, "/c")["verdict"], G.FAIL)

    def test_summary_flags_worse_domains_and_never_judges_quality(self):
        zero = ev(4.5, {"fsc": 4.0, "kumu": 5.0}, worst=0.10)
        best = ev(4.4, {"fsc": 3.7, "kumu": 5.1}, worst=0.08)
        s = G.baseline_summary(zero, best, False)
        self.assertEqual(s["domains_worse"], ["kumu"])
        self.assertAlmostEqual(s["deltas_vs_zero_shot"]["worst_cer"], -0.02)
        self.assertEqual(s["quality"], "native listening pending")
        self.assertIn("no finetuned", G.baseline_summary(zero, zero, True)["reasons"][0])


class RecipeTests(unittest.TestCase):
    def test_stages_follow_the_moss_notes(self):
        self.assertIn("README.md", RECIPE["stage1"]["source"])
        self.assertIn("pilot", RECIPE["stage2"]["source"])
        self.assertNotIn("qwen", json.dumps(RECIPE).lower())
        s2 = RECIPE["stage2"]
        self.assertEqual((s2["learning_rate"], s2["warmup_steps"], s2["weight_decay"], s2["betas"]),
                         (1e-5, 20, 0.1, [0.9, 0.95]))
        self.assertEqual(s2["conditioning"], "none")
        self.assertEqual(RECIPE["status"], "authorized_pending_data")


if __name__ == "__main__":
    unittest.main()
