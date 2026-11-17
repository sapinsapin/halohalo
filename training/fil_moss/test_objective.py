"""Runs where torch is installed (dwnd's MOSS environment); skipped elsewhere."""
import unittest

try:
    import torch
    from training.fil_moss.objective import grpo_token_loss
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch required")
class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logp = torch.log_softmax(torch.randn(5, 16, 8), -1)[..., 0].clone().requires_grad_(True)
        self.valid = torch.ones(5, 16, dtype=torch.bool)
        self.valid[-1, 1:] = False

    def test_on_policy_loss_is_minus_advantage_and_kl_zero_at_reference(self):
        loss, kl = grpo_token_loss(self.logp, self.logp.detach(), self.logp.detach(), 1.5, self.valid, 0.2, 0.04)
        self.assertAlmostEqual(float(loss), -1.5, places=6)
        self.assertAlmostEqual(float(kl), 0.0, places=7)

    def test_gradient_pushes_up_positive_advantage_tokens(self):
        loss, _ = grpo_token_loss(self.logp, self.logp.detach(), self.logp.detach(), 1.0, self.valid, 0.2, 0.0)
        loss.backward()
        self.assertTrue(bool((self.logp.grad[self.valid] < 0).all()))
        self.assertTrue(bool((self.logp.grad[~self.valid] == 0).all()))

    def test_kl_penalises_distance_from_reference(self):
        ref = self.logp.detach() - 0.5
        loss, kl = grpo_token_loss(self.logp, self.logp.detach(), ref, 0.0, self.valid, 0.2, 0.1)
        self.assertGreater(float(kl), 0.0)
        self.assertAlmostEqual(float(loss), 0.1 * float(kl), places=6)

    def test_clipping_caps_the_ratio(self):
        old = self.logp.detach() - 1.0  # ratio e > 1 + eps
        loss, _ = grpo_token_loss(self.logp, old, self.logp.detach(), 1.0, self.valid, 0.2, 0.0)
        self.assertAlmostEqual(float(loss), -1.2, places=5)

    def test_no_valid_tokens(self):
        with self.assertRaises(ValueError):
            grpo_token_loss(self.logp, self.logp.detach(), self.logp.detach(), 1.0,
                            torch.zeros_like(self.valid), 0.2, 0.0)


if __name__ == "__main__":
    unittest.main()
