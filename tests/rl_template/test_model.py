import pytest
import torch
from torch import Tensor

from rl_template.backbones.mlp import MLPBackbone
from rl_template.data.batch import PpoBatch, iterate_minibatches
from rl_template.model import (
    MASKED_LOGIT_VALUE,
    PolicyValueNet,
    apply_action_mask,
    masked_log_softmax,
)

OBS_DIM: int = 7
N_ACTIONS: int = 5
BATCH_SIZE: int = 32
TOTAL_ROWS: int = 23
MINIBATCH_SIZE: int = 7
EXPECTED_REMAINDER: int = 2
TRAINING_STEPS: int = 8
SEED: int = 1234
LEARNING_RATE: float = 0.05
MASK_PROBABILITY: float = 0.25


def test_forward_shapes_without_mask() -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM), N_ACTIONS)
    obs = torch.randn(BATCH_SIZE, OBS_DIM)

    logits, value = net(obs)

    assert logits.shape == (BATCH_SIZE, N_ACTIONS)
    assert value.shape == (BATCH_SIZE,)


def test_forward_masks_illegal_actions() -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM), N_ACTIONS)
    obs = torch.randn(BATCH_SIZE, OBS_DIM)
    mask = torch.ones(BATCH_SIZE, N_ACTIONS, dtype=torch.bool)
    mask[:, 1] = False

    logits, _ = net(obs, mask)
    probabilities = torch.softmax(logits, dim=-1)

    assert torch.all(logits[:, 1] == MASKED_LOGIT_VALUE)
    assert torch.equal(probabilities[:, 1], torch.zeros_like(probabilities[:, 1]))


def test_masked_log_softmax_is_normalized_and_finite() -> None:
    logits = torch.randn(BATCH_SIZE, N_ACTIONS)
    mask = torch.ones(BATCH_SIZE, N_ACTIONS, dtype=torch.bool)
    mask[0, :] = False
    mask[0, 0] = True

    result = masked_log_softmax(logits, mask)

    assert torch.allclose(result.exp().sum(dim=-1), torch.ones(BATCH_SIZE))
    assert torch.isfinite(result).all()


def test_apply_action_mask_is_idempotent() -> None:
    logits = torch.randn(BATCH_SIZE, N_ACTIONS)
    mask = torch.rand(BATCH_SIZE, N_ACTIONS) > MASK_PROBABILITY
    mask[:, 0] = True

    once = apply_action_mask(logits, mask)
    twice = apply_action_mask(once, mask)

    assert torch.equal(once, twice)


def test_policy_value_net_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="obs_dim"):
        MLPBackbone(0)
    with pytest.raises(ValueError, match="hidden_sizes"):
        MLPBackbone(OBS_DIM, hidden_sizes=())


def test_policy_value_net_loss_decreases_on_random_data() -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM, hidden_sizes=(32, 32)), N_ACTIONS)
    obs = torch.randn(BATCH_SIZE, OBS_DIM)
    target_logits = torch.randn(BATCH_SIZE, N_ACTIONS)
    target_value = torch.randn(BATCH_SIZE)
    initial_loss: float | None = None
    final_loss: float = 0.0

    for _ in range(TRAINING_STEPS):
        for parameter in net.parameters():
            parameter.grad = None
        logits, value = net(obs)
        loss: Tensor = torch.nn.functional.mse_loss(logits, target_logits) + torch.nn.functional.mse_loss(
            value, target_value
        )
        if initial_loss is None:
            initial_loss = loss.item()
        torch.autograd.backward(loss)
        with torch.no_grad():
            for parameter in net.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-LEARNING_RATE)
        final_loss = loss.item()

    assert initial_loss is not None
    assert final_loss < initial_loss


def _make_batch() -> PpoBatch:
    row_ids = torch.arange(TOTAL_ROWS, dtype=torch.float32).unsqueeze(1)
    obs = row_ids.repeat(1, OBS_DIM)
    return PpoBatch(
        obs=obs,
        actions=torch.zeros(TOTAL_ROWS, dtype=torch.int64),
        old_log_probs=torch.zeros(TOTAL_ROWS),
        advantages=torch.zeros(TOTAL_ROWS),
        returns=torch.zeros(TOTAL_ROWS),
    )


def test_iterate_minibatches_covers_rows_and_repeats_seed() -> None:
    batch = _make_batch()
    first_generator = torch.Generator().manual_seed(SEED)
    second_generator = torch.Generator().manual_seed(SEED)

    first_minibatches = list(iterate_minibatches(batch, MINIBATCH_SIZE, first_generator))
    second_minibatches = list(iterate_minibatches(batch, MINIBATCH_SIZE, second_generator))
    first_partition = [minibatch.obs[:, 0] for minibatch in first_minibatches]
    second_partition = [minibatch.obs[:, 0] for minibatch in second_minibatches]
    all_ids = torch.cat(first_partition).sort().values

    assert torch.equal(all_ids, torch.arange(TOTAL_ROWS, dtype=torch.float32))
    assert first_partition[-1].shape == (EXPECTED_REMAINDER,)
    assert tuple(first.shape for first in first_partition) == tuple(second.shape for second in second_partition)
    assert torch.equal(torch.cat(first_partition), torch.cat(second_partition))
