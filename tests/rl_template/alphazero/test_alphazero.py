import math

import pytest
import torch

from rl_template.alphazero.config import AlphaZeroConfig
from rl_template.alphazero.update import AlphaZeroMetrics, alphazero_update
from rl_template.backbones.mlp import MLPBackbone
from rl_template.data.batch import AlphaZeroBatch
from rl_template.model import PolicyValueNet

OBS_DIM: int = 7
N_ACTIONS: int = 5
BATCH: int = 32
HIDDEN_SIZE: int = 32
TRAINING_STEPS: int = 10
LEARNING_RATE: float = 0.05
pytestmark = pytest.mark.usefixtures("_seed")


def _make_batch(mask: torch.Tensor | None = None) -> AlphaZeroBatch:
    obs = torch.randn(BATCH, OBS_DIM)
    policy_target = torch.softmax(torch.randn(BATCH, N_ACTIONS), dim=-1)
    if mask is not None:
        policy_target = torch.where(mask, policy_target, torch.zeros_like(policy_target))
        policy_target /= policy_target.sum(dim=-1, keepdim=True)
    return AlphaZeroBatch(
        obs=obs,
        policy_target=policy_target,
        value_target=torch.rand(BATCH) * 2.0 - 1.0,
        mask=mask,
    )


def _config() -> AlphaZeroConfig:
    return AlphaZeroConfig(epochs=1, minibatch_size=BATCH)


def test_alphazero_update_returns_finite_metrics() -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM, (HIDDEN_SIZE, HIDDEN_SIZE)), N_ACTIONS)
    batch = _make_batch()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    metrics = alphazero_update(net, optimizer, batch, _config())

    assert isinstance(metrics, AlphaZeroMetrics)
    assert isinstance(metrics.policy_loss, float)
    assert isinstance(metrics.value_loss, float)
    assert math.isfinite(metrics.policy_loss)
    assert math.isfinite(metrics.value_loss)
    assert metrics.policy_loss >= 0.0


def test_alphazero_update_losses_decrease_on_repeated_updates() -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM, (HIDDEN_SIZE, HIDDEN_SIZE)), N_ACTIONS)
    batch = _make_batch()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    policy_losses: list[float] = []
    value_losses: list[float] = []

    for _ in range(TRAINING_STEPS):
        metrics = alphazero_update(net, optimizer, batch, _config())
        policy_losses.append(metrics.policy_loss)
        value_losses.append(metrics.value_loss)

    assert sum(policy_losses[-3:]) / 3 < policy_losses[0]
    assert sum(value_losses[-3:]) / 3 < value_losses[0]


def test_alphazero_update_preserves_zero_probability_for_illegal_actions() -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM, (HIDDEN_SIZE, HIDDEN_SIZE)), N_ACTIONS)
    mask = torch.ones(BATCH, N_ACTIONS, dtype=torch.bool)
    mask[:, 0] = False
    mask[0] = False
    mask[0, 1] = True
    batch = _make_batch(mask)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    metrics = alphazero_update(net, optimizer, batch, _config())
    probabilities = torch.softmax(net(batch.obs, mask)[0], dim=-1)

    assert math.isfinite(metrics.policy_loss)
    assert math.isfinite(metrics.value_loss)
    assert torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask]))
