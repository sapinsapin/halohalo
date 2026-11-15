import json
import unittest
from pathlib import Path

from training.fil_moss.timing import estimate, gen_seconds

RECIPE = json.loads((Path(__file__).resolve().parent / "recipe.json").read_text())

T = {"frames_per_second": 12.5, "load_seconds": 60, "save_seconds": 20, "save_reload_seconds": 60,
     "train_step_seconds": {"1": 0.75, "4": 0.77, "8": 1.28}, "forward_seconds": {"1": 0.15, "4": 0.4, "8": 0.7}, "ref_forward_seconds": {"1": 0.1, "4": 0.3, "8": 0.5},
     "sync_seconds": 0.05,
     "gen_prefill_seconds": 0.2, "gen_seconds_per_frame": {"1": 0.09, "8": 0.11},
     "decode_seconds_per_audio_second": 0.05, "trim_seconds": 0.01, "ecapa_seconds": 0.02,
     "judge_local_seconds": {"mms": 0.05, "whisper": 0.5, "omni": 0.05}, "judge_remote_seconds_per_utt": 0.4}
DATA = {"train_hours": 1.0, "validation_hours": 0.1, "mean_clip_seconds": 5.0}


class EstimateTests(unittest.TestCase):
    def test_generation_scales_with_frames(self):
        self.assertAlmostEqual(gen_seconds(T, 4.0, 8), 0.2 + 50 * 0.11)

    def test_updates_follow_the_frozen_dataset(self):
        b1 = estimate(T, DATA, RECIPE, batch_size=1)
        b4 = estimate(T, DATA, RECIPE, batch_size=4)
        self.assertEqual(b1["stage2"]["updates"], 720 * 2)
        self.assertEqual(b4["stage2"]["updates"], 180 * 2)
        self.assertLess(b4["stage2"]["train_seconds"], b1["stage2"]["train_seconds"] / 3)

    def test_remote_judging_is_cheaper(self):
        remote = estimate(T, DATA, RECIPE, remote=True)
        local = estimate(T, DATA, RECIPE, remote=False)
        self.assertLess(remote["stage2"]["eval_seconds_each"], local["stage2"]["eval_seconds_each"])
        self.assertGreater(remote["everything_seconds"], remote["stage2"]["total_seconds"])
        self.assertLess(remote["stage3_arm"]["step_seconds"], local["stage3_arm"]["step_seconds"])
        self.assertLess(remote["stage3_arm"]["killed_at_no_gain_step_seconds"], remote["stage3_arm"]["total_seconds"])

    def test_pipeline_hides_judging_up_to_the_slower_side(self):
        slow = dict(T, judge_remote_seconds_per_utt=10.0)
        self.assertAlmostEqual(estimate(slow, DATA, RECIPE, remote=True)["stage3_arm"]["step_seconds"], 80.0)


class WireTests(unittest.TestCase):
    def test_audio_roundtrip(self):
        try:
            import numpy as np
            from training.fil_moss.judge_server import decode_audio, encode_audio
        except ImportError:
            self.skipTest("numpy required")
        x = np.sin(np.linspace(0, 20, 1600)).astype(np.float32) * 0.5
        y = decode_audio(encode_audio(x))
        self.assertLess(float(np.abs(x - y).max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
