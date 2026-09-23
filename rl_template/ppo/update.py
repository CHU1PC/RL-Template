from dataclasses import dataclass, field

import torch
from torch.nn import functional

from rl_template.data.batch import PpoBatch, iterate_minibatches
from rl_template.model import PolicyValueNet
from rl_template.ppo.config import PpoConfig


@dataclass(frozen=True, slots=True)
class PpoMetrics:
    """Summarize PPO losses and diagnostics."""

    policy_loss: float = field()
    value_loss: float = field()
    entropy: float = field()
    approx_kl: float = field()
    clip_fraction: float = field()


@dataclass(slots=True)
class _PpoTotals:
    """Accumulate PPO metrics by name."""

    policy_loss: float = field(default=0.0)
    value_loss: float = field(default=0.0)
    entropy: float = field(default=0.0)
    approx_kl: float = field(default=0.0)
    clip_fraction: float = field(default=0.0)


def ppo_update(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: PpoBatch,
    cfg: PpoConfig,
    generator: torch.Generator | None = None,
) -> PpoMetrics:
    """Run PPO optimization over a prepared batch.

    Args:
        net: Policy and value network to optimize.
        optimizer: Caller-built optimizer used for each minibatch step.
        batch: Prepared PPO tensors from the Rust side.
        cfg: PPO update hyperparameters.
        generator: Optional generator for minibatch permutations.

    Returns:
        Mean losses and diagnostics over all optimization minibatches.
    """
    totals: _PpoTotals = _PpoTotals()
    minibatch_count: int = 0

    for _ in range(cfg.epochs):
        for mb in iterate_minibatches(batch, cfg.minibatch_size, generator):
            logits, value = net(mb.obs, mb.mask)
            log_probs_all = torch.log_softmax(logits, -1)
            log_probs = log_probs_all.gather(1, mb.actions.unsqueeze(1)).squeeze(1)
            ratio = (log_probs - mb.old_log_probs).exp()
            policy_loss = -torch.min(
                ratio * mb.advantages,
                ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb.advantages,
            ).mean()
            value_loss = functional.mse_loss(value, mb.returns)
            if mb.mask is not None:
                per_action = torch.where(
                    mb.mask,
                    log_probs_all.exp() * log_probs_all,
                    torch.zeros_like(log_probs_all),
                )
            else:
                per_action = log_probs_all.exp() * log_probs_all
            entropy = -per_action.sum(dim=-1).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # pyright: ignore[reportUnknownMemberType]  # torch の stub が引数型を Unknown とする
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = log_probs - mb.old_log_probs
                approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > cfg.clip_eps).to(torch.float32).mean()
                totals.policy_loss += policy_loss.item()
                totals.value_loss += value_loss.item()
                totals.entropy += entropy.item()
                totals.approx_kl += approx_kl.item()
                totals.clip_fraction += clip_fraction.item()
                minibatch_count += 1

    return PpoMetrics(
        policy_loss=totals.policy_loss / minibatch_count,
        value_loss=totals.value_loss / minibatch_count,
        entropy=totals.entropy / minibatch_count,
        approx_kl=totals.approx_kl / minibatch_count,
        clip_fraction=totals.clip_fraction / minibatch_count,
    )
