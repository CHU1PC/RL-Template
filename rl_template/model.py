from typing import Final

import torch
from torch import Tensor, nn

from rl_template.backbones.base import Backbone
from rl_template.backbones.factory import build_backbone
from rl_template.config import TrainConfig

MASKED_LOGIT_VALUE: Final[float] = -1e9  # 有限値なら softmax/log_softmax は全行マスクでも NaN を出さない。-inf は出す。


def apply_action_mask(logits: Tensor, mask: Tensor | None) -> Tensor:
    """Mask illegal actions in logits.

    Args:
        logits: Unmasked action logits.
        mask: Boolean tensor where true entries are legal actions.

    Returns:
        Logits with illegal actions replaced by a finite sentinel.

    Raises:
        ValueError: If any row has no legal action.
    """
    if mask is None:
        return logits
    if not mask.any(dim=-1).all():
        message: str = "mask must contain at least one legal action per row"
        raise ValueError(message)
    return logits.masked_fill(~mask, MASKED_LOGIT_VALUE)


# PolicyValueNet.forward() already applies the mask, so callers holding its
# output should use ``log_softmax`` directly.
def masked_log_softmax(logits: Tensor, mask: Tensor | None) -> Tensor:
    """Compute log probabilities after applying an action mask.

    Args:
        logits: Unmasked action logits.
        mask: Boolean tensor where true entries are legal actions.

    Returns:
        Log probabilities normalized over the action dimension.
    """
    return torch.nn.functional.log_softmax(apply_action_mask(logits, mask), dim=-1)


class PolicyValueNet(nn.Module):
    """Store a configurable observation backbone with policy and value heads."""

    def __init__(
        self,
        backbone: Backbone,
        n_actions: int,
    ) -> None:
        """Initialize the policy and value network.

        Args:
            backbone: Observation feature extractor.
            n_actions: Number of discrete actions.

        Raises:
            ValueError: If the action count is non-positive.
        """
        super().__init__()
        if n_actions <= 0:
            message: str = "n_actions must be positive"
            raise ValueError(message)

        self.backbone: Backbone = backbone
        self.obs_dim: int = backbone.obs_dim
        self.n_actions: int = n_actions
        self.policy_head: nn.Linear = nn.Linear(backbone.out_dim, n_actions)
        self.value_head: nn.Linear = nn.Linear(backbone.out_dim, 1)

    def forward(self, obs: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Compute masked policy logits and scalar state values.

        Args:
            obs: Float observation tensor with shape ``[B, obs_dim]``.
            mask: Optional boolean action mask with shape ``[B, n_actions]``.

        Returns:
            A tuple containing logits with shape ``[B, n_actions]`` and values
            with shape ``[B]``.
        """
        features: Tensor = self.backbone(obs)
        logits: Tensor = apply_action_mask(self.policy_head(features), mask)
        value: Tensor = self.value_head(features).squeeze(-1)
        return logits, value


def build_net(cfg: TrainConfig) -> PolicyValueNet:
    """Build a policy-value network from a complete training configuration.

    Returns:
        A policy-value network with the configured backbone and action count.
    """
    backbone: Backbone = build_backbone(cfg.model, cfg.obs_dim)
    return PolicyValueNet(backbone, cfg.n_actions)
