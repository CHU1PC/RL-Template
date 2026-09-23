import math

import pytest
import torch

from rl_template.backbones.mlp import MLPBackbone
from rl_template.data.batch import PpoBatch
from rl_template.model import PolicyValueNet, masked_log_softmax
from rl_template.ppo.config import PpoConfig
from rl_template.ppo.update import PpoMetrics, ppo_update

OBS_DIM: int = 7
N_ACTIONS: int = 5
BATCH: int = 32
HIDDEN_SIZE: int = 32
TRAINING_STEPS: int = 10
LEARNING_RATE: float = 0.05
pytestmark = pytest.mark.usefixtures("_seed")


def _make_batch(net: PolicyValueNet, mask: torch.Tensor | None = None) -> PpoBatch:
    obs = torch.randn(BATCH, OBS_DIM)
    actions = torch.randint(N_ACTIONS, (BATCH,)) if mask is None else torch.randint(1, N_ACTIONS, (BATCH,))
    with torch.no_grad():
        logits, _ = net(obs, mask)
        old_log_probs = masked_log_softmax(logits, mask).gather(1, actions.unsqueeze(1)).squeeze(1)
    return PpoBatch(
        obs=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=torch.randn(BATCH),
        returns=torch.randn(BATCH),
        mask=mask,
    )


def _config() -> PpoConfig:
    return PpoConfig(epochs=1, minibatch_size=BATCH)


def test_ppo_update_returns_finite_metrics() -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM, (HIDDEN_SIZE, HIDDEN_SIZE)), N_ACTIONS)
    batch = _make_batch(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    metrics = ppo_update(net, optimizer, batch, _config())

    assert isinstance(metrics, PpoMetrics)
    values = (
        metrics.policy_loss,
        metrics.value_loss,
        metrics.entropy,
        metrics.approx_kl,
        metrics.clip_fraction,
    )
    assert all(isinstance(value, float) and math.isfinite(value) for value in values)
    assert metrics.entropy >= 0.0


def test_ppo_update_policy_loss_decreases_on_repeated_updates() -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM, (HIDDEN_SIZE, HIDDEN_SIZE)), N_ACTIONS)
    batch = _make_batch(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    losses: list[float] = []

    for _ in range(TRAINING_STEPS):
        metrics = ppo_update(net, optimizer, batch, _config())
        losses.append(metrics.policy_loss)

    assert sum(losses[-3:]) / 3 < losses[0]


def test_ppo_update_preserves_zero_probability_for_illegal_actions() -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM, (HIDDEN_SIZE, HIDDEN_SIZE)), N_ACTIONS)
    mask = torch.ones(BATCH, N_ACTIONS, dtype=torch.bool)
    mask[:, 0] = False
    batch = _make_batch(net, mask)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    ppo_update(net, optimizer, batch, _config())
    probabilities = torch.softmax(net(batch.obs, mask)[0], dim=-1)

    assert torch.equal(probabilities[:, 0], torch.zeros(BATCH))


def test_ppo_update_approx_kl_is_zero_for_current_log_probs() -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM, (HIDDEN_SIZE, HIDDEN_SIZE)), N_ACTIONS)
    batch = _make_batch(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    metrics = ppo_update(net, optimizer, batch, _config())

    assert metrics.approx_kl == pytest.approx(0.0, abs=1e-7)
