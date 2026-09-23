import pytest
import torch

from rl_template.data.batch import AlphaZeroBatch, PpoBatch
from rl_template.model import apply_action_mask


def test_ppo_batch_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        PpoBatch(
            obs=torch.empty(0, 7),
            actions=torch.empty(0, dtype=torch.int64),
            old_log_probs=torch.empty(0),
            advantages=torch.empty(0),
            returns=torch.empty(0),
        )


def test_alphazero_batch_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        AlphaZeroBatch(
            obs=torch.empty(0, 7),
            policy_target=torch.empty(0, 5),
            value_target=torch.empty(0),
        )


def test_action_mask_rejects_rows_without_legal_actions() -> None:
    logits = torch.zeros(2, 3)
    mask = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match="at least one legal action per row"):
        apply_action_mask(logits, mask)
