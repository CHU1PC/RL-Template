from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]  # safetensors の PathLike[Unknown] 起因
from torch import Tensor

from rl_template.data.batch import AlphaZeroBatch, PpoBatch
from rl_template.data.samples import (
    load_alphazero_samples,
    load_ppo_samples,
    save_alphazero_samples,
    save_ppo_samples,
)

OBS_DIM: int = 3
N_ACTIONS: int = 4
BATCH_SIZE: int = 5


def _ppo_batch() -> PpoBatch:
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, True, True, False],
            [True, False, False, True],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    return PpoBatch(
        obs=torch.arange(BATCH_SIZE * OBS_DIM, dtype=torch.float32).reshape(BATCH_SIZE, OBS_DIM),
        actions=torch.tensor([0, 2, 1, 3, 0], dtype=torch.int64),
        old_log_probs=torch.arange(BATCH_SIZE, dtype=torch.float32),
        advantages=torch.arange(BATCH_SIZE, dtype=torch.float32) / 10.0,
        returns=torch.arange(BATCH_SIZE, dtype=torch.float32) / 5.0,
        mask=mask,
    )


def _alphazero_batch() -> AlphaZeroBatch:
    mask = torch.ones(BATCH_SIZE, N_ACTIONS, dtype=torch.bool)
    mask[:, 0] = False
    policy_target = torch.full((BATCH_SIZE, N_ACTIONS), 0.25, dtype=torch.float32)
    policy_target[:, 0] = 0.0
    policy_target[:, 1:] /= policy_target[:, 1:].sum(dim=-1, keepdim=True)
    return AlphaZeroBatch(
        obs=torch.arange(BATCH_SIZE * OBS_DIM, dtype=torch.float32).reshape(BATCH_SIZE, OBS_DIM),
        policy_target=policy_target,
        value_target=torch.arange(BATCH_SIZE, dtype=torch.float32) / 10.0,
        mask=mask,
    )


def _ppo_tensors() -> dict[str, Tensor]:
    batch = _ppo_batch()
    assert batch.mask is not None
    return {
        "obs": batch.obs,
        "mask": batch.mask,
        "actions": batch.actions,
        "old_log_probs": batch.old_log_probs,
        "advantages": batch.advantages,
        "returns": batch.returns,
    }


def test_ppo_samples_round_trip_preserves_tensors_and_dtypes(tmp_path: Path) -> None:
    batch = _ppo_batch()
    path = tmp_path / "ppo.safetensors"

    assert save_ppo_samples(batch, path, N_ACTIONS) == path
    loaded = load_ppo_samples(path, OBS_DIM, N_ACTIONS)

    assert torch.equal(loaded.obs, batch.obs)
    assert torch.equal(loaded.actions, batch.actions)
    assert torch.equal(loaded.old_log_probs, batch.old_log_probs)
    assert torch.equal(loaded.advantages, batch.advantages)
    assert torch.equal(loaded.returns, batch.returns)
    assert loaded.mask is not None
    assert batch.mask is not None
    assert torch.equal(loaded.mask, batch.mask)
    assert loaded.obs.dtype == torch.float32
    assert loaded.mask.dtype == torch.bool
    assert loaded.actions.dtype == torch.int64
    assert loaded.old_log_probs.dtype == torch.float32
    assert loaded.advantages.dtype == torch.float32
    assert loaded.returns.dtype == torch.float32


def test_alphazero_samples_round_trip_preserves_tensors_and_dtypes(tmp_path: Path) -> None:
    batch = _alphazero_batch()
    path = tmp_path / "alphazero.safetensors"

    assert save_alphazero_samples(batch, path) == path
    loaded = load_alphazero_samples(path, OBS_DIM, N_ACTIONS)

    assert torch.equal(loaded.obs, batch.obs)
    assert torch.equal(loaded.policy_target, batch.policy_target)
    assert torch.equal(loaded.value_target, batch.value_target)
    assert loaded.mask is not None
    assert batch.mask is not None
    assert torch.equal(loaded.mask, batch.mask)
    assert loaded.obs.dtype == torch.float32
    assert loaded.mask.dtype == torch.bool
    assert loaded.policy_target.dtype == torch.float32
    assert loaded.value_target.dtype == torch.float32


def test_loading_with_wrong_obs_dim_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "ppo.safetensors"
    save_ppo_samples(_ppo_batch(), path, N_ACTIONS)

    with pytest.raises(ValueError, match=r"key 'obs'.*trailing shape.*expected"):
        load_ppo_samples(path, OBS_DIM + 1, N_ACTIONS)


def test_loading_with_wrong_n_actions_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "alphazero.safetensors"
    save_alphazero_samples(_alphazero_batch(), path)

    with pytest.raises(ValueError, match=r"key 'mask'.*trailing shape.*expected"):
        load_alphazero_samples(path, OBS_DIM, N_ACTIONS + 1)


def test_loading_file_missing_required_key_raises_value_error(tmp_path: Path) -> None:
    tensors = _ppo_tensors()
    del tensors["returns"]
    path = tmp_path / "missing.safetensors"
    save_file(tensors, path)  # pyright: ignore[reportUnknownVariableType]  # safetensors の PathLike[Unknown] 起因

    with pytest.raises(ValueError, match="missing required key 'returns'"):
        load_ppo_samples(path, OBS_DIM, N_ACTIONS)


def test_loading_file_with_wrong_dtype_raises_value_error(tmp_path: Path) -> None:
    tensors = _ppo_tensors()
    tensors["actions"] = tensors["actions"].to(torch.float32)
    path = tmp_path / "wrong-dtype.safetensors"
    save_file(tensors, path)  # pyright: ignore[reportUnknownVariableType]  # safetensors の PathLike[Unknown] 起因

    with pytest.raises(ValueError, match=r"key 'actions'.*dtype.*expected"):
        load_ppo_samples(path, OBS_DIM, N_ACTIONS)


def test_loading_file_with_unexpected_key_is_strict(tmp_path: Path) -> None:
    tensors = _ppo_tensors()
    tensors["extra"] = torch.ones(1, dtype=torch.float32)
    path = tmp_path / "extra.safetensors"
    save_file(tensors, path)  # pyright: ignore[reportUnknownVariableType]  # safetensors の PathLike[Unknown] 起因

    with pytest.raises(ValueError, match="unexpected extra key 'extra'"):
        load_ppo_samples(path, OBS_DIM, N_ACTIONS)


def test_ppo_samples_with_missing_mask_use_explicit_action_count(tmp_path: Path) -> None:
    source = _ppo_batch()
    batch = PpoBatch(
        obs=source.obs,
        actions=torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64),
        old_log_probs=source.old_log_probs,
        advantages=source.advantages,
        returns=source.returns,
    )
    path = tmp_path / "ppo-no-mask.safetensors"

    save_ppo_samples(batch, path, 10)
    loaded = load_ppo_samples(path, OBS_DIM, 10)

    assert loaded.mask is not None
    assert loaded.mask.shape == (BATCH_SIZE, 10)
    assert loaded.mask.all()


def test_save_ppo_samples_rejects_wrong_dtype(tmp_path: Path) -> None:
    source = _ppo_batch()
    batch = PpoBatch(
        obs=source.obs,
        actions=source.actions,
        old_log_probs=source.old_log_probs.to(torch.float64),
        advantages=source.advantages,
        returns=source.returns,
        mask=source.mask,
    )

    with pytest.raises(ValueError, match=r"key 'old_log_probs'.*dtype.*expected"):
        save_ppo_samples(batch, tmp_path / "wrong-ppo-dtype.safetensors", N_ACTIONS)


def test_save_ppo_samples_rejects_wrong_mask_width(tmp_path: Path) -> None:
    source = _ppo_batch()
    assert source.mask is not None
    batch = PpoBatch(
        obs=source.obs,
        actions=source.actions,
        old_log_probs=source.old_log_probs,
        advantages=source.advantages,
        returns=source.returns,
        mask=source.mask[:, :-1],
    )

    with pytest.raises(ValueError, match="mask width"):
        save_ppo_samples(batch, tmp_path / "wrong-ppo-mask.safetensors", N_ACTIONS)


def test_save_alphazero_samples_rejects_wrong_dtype(tmp_path: Path) -> None:
    source = _alphazero_batch()
    assert source.mask is not None
    batch = AlphaZeroBatch(
        obs=source.obs,
        policy_target=source.policy_target,
        value_target=source.value_target,
        mask=source.mask.to(torch.int8),
    )

    with pytest.raises(ValueError, match=r"key 'mask'.*dtype.*expected"):
        save_alphazero_samples(batch, tmp_path / "wrong-alphazero-dtype.safetensors")


def test_save_alphazero_samples_rejects_wrong_mask_width(tmp_path: Path) -> None:
    source = _alphazero_batch()
    assert source.mask is not None
    batch = AlphaZeroBatch(
        obs=source.obs,
        policy_target=source.policy_target,
        value_target=source.value_target,
        mask=source.mask[:, :-1],
    )

    with pytest.raises(ValueError, match="mask width"):
        save_alphazero_samples(batch, tmp_path / "wrong-alphazero-mask.safetensors")
