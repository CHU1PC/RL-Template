from dataclasses import dataclass, field

import torch
from torch.nn import functional

from rl_template.alphazero.config import AlphaZeroConfig
from rl_template.data.batch import AlphaZeroBatch, iterate_minibatches
from rl_template.model import PolicyValueNet


@dataclass(frozen=True, slots=True)
class AlphaZeroMetrics:
    """Summarize AlphaZero policy and value losses."""

    policy_loss: float = field()
    value_loss: float = field()


def alphazero_update(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: AlphaZeroBatch,
    cfg: AlphaZeroConfig,
    generator: torch.Generator | None = None,
) -> AlphaZeroMetrics:
    """Run AlphaZero optimization over a prepared batch.

    Args:
        net: Policy and value network to optimize.
        optimizer: Caller-built optimizer used for each minibatch step.
        batch: Prepared AlphaZero tensors from the Rust side.
        cfg: AlphaZero update hyperparameters.
        generator: Optional generator for minibatch permutations.

    Returns:
        Mean policy and value losses over all optimization minibatches.
    """
    policy_loss_total: float = 0.0
    value_loss_total: float = 0.0
    minibatch_count: int = 0

    for _ in range(cfg.epochs):
        for mb in iterate_minibatches(batch, cfg.minibatch_size, generator):
            logits, value = net(mb.obs, mb.mask)
            log_probs = torch.log_softmax(logits, -1)
            per_action = mb.policy_target * log_probs
            if mb.mask is not None:
                per_action = torch.where(mb.mask, per_action, torch.zeros_like(per_action))
            policy_loss = -per_action.sum(dim=-1).mean()
            value_loss = functional.mse_loss(value, mb.value_target)
            loss = policy_loss + cfg.value_coef * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # pyright: ignore[reportUnknownMemberType]  # torch の stub が引数型を Unknown とする
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
            minibatch_count += 1

    return AlphaZeroMetrics(
        policy_loss=policy_loss_total / minibatch_count,
        value_loss=value_loss_total / minibatch_count,
    )
