"""GRPO token objective (torch only, no model code), so its arithmetic can be unit-tested on CPU."""
import torch


def grpo_token_loss(logp, logp_old, logp_ref, advantage, valid, clip_epsilon, kl_beta):
    """Clipped-surrogate GRPO loss for one rollout, averaged over its valid audio tokens.

    ``logp``      [N, 16] policy log-probabilities (with grad)
    ``logp_old``  [N, 16] log-probabilities at rollout time (equal to ``logp.detach()`` when ppo_epochs == 1)
    ``logp_ref``  [N, 16] frozen stage 1 reference log-probabilities
    ``advantage`` scalar group-normalised advantage of this rollout
    ``valid``     [N, 16] bool mask of real labels

    KL uses the k3 estimator exp(ref - logp) - (ref - logp) - 1, which is non-negative and zero when the policy
    equals the reference.
    """
    ratio = torch.exp(logp - logp_old)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    delta = logp_ref - logp
    kl = torch.exp(delta) - delta - 1
    per_token = -(surrogate - kl_beta * kl)
    if not bool(valid.any()):
        raise ValueError("rollout has no valid tokens")
    return per_token[valid].mean(), kl[valid].mean().detach()
