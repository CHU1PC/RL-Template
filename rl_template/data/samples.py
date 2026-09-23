"""safetensors 形式の自己対戦サンプルを読み書きする。

ファイルのキー、dtype、shape は Rust 実装と共有する固定仕様になる。

| Algorithm | Key | dtype | shape |
| --- | --- | --- | --- |
| PPO | `obs` | `f32` | `[N, obs_dim]` |
| PPO | `mask` | `bool` | `[N, n_actions]` |
| PPO | `actions` | `i64` | `[N]` |
| PPO | `old_log_probs` | `f32` | `[N]` |
| PPO | `advantages` | `f32` | `[N]` |
| PPO | `returns` | `f32` | `[N]` |
| AlphaZero | `obs` | `f32` | `[N, obs_dim]` |
| AlphaZero | `mask` | `bool` | `[N, n_actions]` |
| AlphaZero | `policy_target` | `f32` | `[N, n_actions]` |
| AlphaZero | `value_target` | `f32` | `[N]` |

`mask` はファイル上で必須で、メモリ上の `None` は全要素が `True` の mask として保存する。
ローダーはキー集合と tensor の dtype、rank、shape、先頭次元を厳密に検証し、余分なキーを拒否する。
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import (
    load_file,  # pyright: ignore[reportUnknownVariableType]  # safetensors の PathLike[Unknown] 起因
    save_file,  # pyright: ignore[reportUnknownVariableType]  # safetensors の PathLike[Unknown] 起因
)
from torch import Tensor

from rl_template.data.batch import AlphaZeroBatch, PpoBatch

__all__: list[str] = [
    "load_alphazero_samples",
    "load_ppo_samples",
    "save_alphazero_samples",
    "save_ppo_samples",
]

_PPO_KEYS: frozenset[str] = frozenset(
    {
        "obs",
        "mask",
        "actions",
        "old_log_probs",
        "advantages",
        "returns",
    }
)
_ALPHAZERO_KEYS: frozenset[str] = frozenset({"obs", "mask", "policy_target", "value_target"})


@dataclass(frozen=True, slots=True)
class _TensorSpec:
    """Describe one tensor's fixed on-disk layout."""

    dtype: torch.dtype = field()
    rank: int = field()
    trailing_shape: tuple[int, ...] = field()


def _validate_keys(tensors: dict[str, Tensor], expected: frozenset[str]) -> None:
    missing: list[str] = sorted(expected - tensors.keys())
    if missing:
        message: str = f"missing required key '{missing[0]}'; expected keys {sorted(expected)}"
        raise ValueError(message)
    unexpected: list[str] = sorted(tensors.keys() - expected)
    if unexpected:
        message = f"unexpected extra key '{unexpected[0]}'; expected keys {sorted(expected)}"
        raise ValueError(message)


def _validate_tensor(
    value: Tensor,
    key: str,
    spec: _TensorSpec,
    expected_batch_size: int | None,
) -> int:
    if value.dtype != spec.dtype:
        message: str = f"key '{key}' has dtype {value.dtype}, expected {spec.dtype}"
        raise ValueError(message)
    if value.ndim != spec.rank:
        message = f"key '{key}' has rank {value.ndim}, expected {spec.rank}"
        raise ValueError(message)
    actual_trailing_shape: tuple[int, ...] = tuple(value.shape[1:])
    if actual_trailing_shape != spec.trailing_shape:
        message = f"key '{key}' has trailing shape {actual_trailing_shape}, expected {spec.trailing_shape}"
        raise ValueError(message)
    actual_batch_size: int = value.shape[0]
    if expected_batch_size is not None and actual_batch_size != expected_batch_size:
        message = (
            f"key '{key}' has leading dimension {actual_batch_size}, expected {expected_batch_size} to match key 'obs'"
        )
        raise ValueError(message)
    return actual_batch_size


def _load_tensors(path: Path) -> dict[str, Tensor]:
    return load_file(path)  # pyright: ignore[reportUnknownVariableType]  # safetensors の PathLike[Unknown] 起因


def _validate_ppo_tensors(tensors: dict[str, Tensor], obs_dim: int, n_actions: int) -> int:
    _validate_keys(tensors, _PPO_KEYS)
    batch_size: int = _validate_tensor(tensors["obs"], "obs", _TensorSpec(torch.float32, 2, (obs_dim,)), None)
    _validate_tensor(tensors["mask"], "mask", _TensorSpec(torch.bool, 2, (n_actions,)), batch_size)
    _validate_tensor(tensors["actions"], "actions", _TensorSpec(torch.int64, 1, ()), batch_size)
    _validate_tensor(tensors["old_log_probs"], "old_log_probs", _TensorSpec(torch.float32, 1, ()), batch_size)
    _validate_tensor(tensors["advantages"], "advantages", _TensorSpec(torch.float32, 1, ()), batch_size)
    _validate_tensor(tensors["returns"], "returns", _TensorSpec(torch.float32, 1, ()), batch_size)
    return batch_size


def _validate_alphazero_tensors(tensors: dict[str, Tensor], obs_dim: int, n_actions: int) -> int:
    _validate_keys(tensors, _ALPHAZERO_KEYS)
    batch_size: int = _validate_tensor(tensors["obs"], "obs", _TensorSpec(torch.float32, 2, (obs_dim,)), None)
    _validate_tensor(tensors["mask"], "mask", _TensorSpec(torch.bool, 2, (n_actions,)), batch_size)
    _validate_tensor(tensors["policy_target"], "policy_target", _TensorSpec(torch.float32, 2, (n_actions,)), batch_size)
    _validate_tensor(tensors["value_target"], "value_target", _TensorSpec(torch.float32, 1, ()), batch_size)
    return batch_size


def _validate_save_tensor(value: Tensor, key: str, expected_dtype: torch.dtype, expected_rank: int) -> None:
    if value.dtype != expected_dtype:
        message: str = f"key '{key}' has dtype {value.dtype}, expected {expected_dtype}"
        raise ValueError(message)
    if value.ndim != expected_rank:
        message = f"key '{key}' has rank {value.ndim}, expected {expected_rank}"
        raise ValueError(message)


def _ppo_mask(batch: PpoBatch, n_actions: int) -> Tensor:
    if batch.mask is not None:
        return batch.mask
    if (batch.actions < 0).any():
        message: str = "actions must contain only non-negative values when mask is None"
        raise ValueError(message)
    return torch.ones((batch.batch_size, n_actions), dtype=torch.bool, device=batch.actions.device)


def _save_tensors(tensors: dict[str, Tensor], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)  # pyright: ignore[reportUnknownVariableType]  # safetensors の PathLike[Unknown] 起因
    return path


def _validate_mask_width(mask: Tensor, n_actions: int) -> None:
    if mask.shape[1] != n_actions:
        message: str = f"mask width {mask.shape[1]} does not match n_actions {n_actions}"
        raise ValueError(message)


def load_ppo_samples(path: Path, obs_dim: int, n_actions: int) -> PpoBatch:
    """Load and validate PPO samples from a safetensors file.

    Args:
        path: Safetensors file containing PPO samples.
        obs_dim: Expected observation feature count.
        n_actions: Expected discrete action count.

    Returns:
        Validated PPO sample batch.

    Raises:
        ValueError: If dimensions or tensor layouts do not match the fixed PPO format.
    """
    if obs_dim <= 0:
        message: str = f"obs_dim must be positive, got {obs_dim}"
        raise ValueError(message)
    if n_actions <= 0:
        message = f"n_actions must be positive, got {n_actions}"
        raise ValueError(message)
    tensors: dict[str, Tensor] = _load_tensors(path)
    _validate_ppo_tensors(tensors, obs_dim, n_actions)
    return PpoBatch(
        obs=tensors["obs"],
        actions=tensors["actions"],
        old_log_probs=tensors["old_log_probs"],
        advantages=tensors["advantages"],
        returns=tensors["returns"],
        mask=tensors["mask"],
    )


def load_alphazero_samples(path: Path, obs_dim: int, n_actions: int) -> AlphaZeroBatch:
    """Load and validate AlphaZero samples from a safetensors file.

    Args:
        path: Safetensors file containing AlphaZero samples.
        obs_dim: Expected observation feature count.
        n_actions: Expected discrete action count.

    Returns:
        Validated AlphaZero sample batch.

    Raises:
        ValueError: If dimensions or tensor layouts do not match the fixed AlphaZero format.
    """
    if obs_dim <= 0:
        message: str = f"obs_dim must be positive, got {obs_dim}"
        raise ValueError(message)
    if n_actions <= 0:
        message = f"n_actions must be positive, got {n_actions}"
        raise ValueError(message)
    tensors: dict[str, Tensor] = _load_tensors(path)
    _validate_alphazero_tensors(tensors, obs_dim, n_actions)
    return AlphaZeroBatch(
        obs=tensors["obs"],
        policy_target=tensors["policy_target"],
        value_target=tensors["value_target"],
        mask=tensors["mask"],
    )


def save_ppo_samples(batch: PpoBatch, path: Path, n_actions: int) -> Path:
    """Save PPO samples in the fixed safetensors layout and return ``path``.

    Args:
        batch: PPO sample batch to save.
        path: Destination safetensors file.
        n_actions: Number of discrete actions in the saved mask.

    Returns:
        The destination path.

    Raises:
        ValueError: If dimensions, dtypes, or tensor layouts do not match the PPO format.
    """
    if n_actions <= 0:
        message: str = f"n_actions must be positive, got {n_actions}"
        raise ValueError(message)
    _validate_save_tensor(batch.obs, "obs", torch.float32, 2)
    _validate_save_tensor(batch.actions, "actions", torch.int64, 1)
    _validate_save_tensor(batch.old_log_probs, "old_log_probs", torch.float32, 1)
    _validate_save_tensor(batch.advantages, "advantages", torch.float32, 1)
    _validate_save_tensor(batch.returns, "returns", torch.float32, 1)
    mask: Tensor = _ppo_mask(batch, n_actions)
    _validate_save_tensor(mask, "mask", torch.bool, 2)
    _validate_mask_width(mask, n_actions)
    tensors: dict[str, Tensor] = {
        "obs": batch.obs,
        "mask": mask,
        "actions": batch.actions,
        "old_log_probs": batch.old_log_probs,
        "advantages": batch.advantages,
        "returns": batch.returns,
    }
    return _save_tensors(tensors, path)


def save_alphazero_samples(batch: AlphaZeroBatch, path: Path) -> Path:
    """Save AlphaZero samples in the fixed safetensors layout and return ``path``.

    Args:
        batch: AlphaZero sample batch to save.
        path: Destination safetensors file.

    Returns:
        The destination path.

    Raises:
        ValueError: If dimensions, dtypes, or tensor layouts do not match the AlphaZero format.
    """
    _validate_save_tensor(batch.obs, "obs", torch.float32, 2)
    _validate_save_tensor(batch.policy_target, "policy_target", torch.float32, 2)
    _validate_save_tensor(batch.value_target, "value_target", torch.float32, 1)
    n_actions: int = batch.policy_target.shape[1]
    if n_actions <= 0:
        message: str = f"n_actions must be positive, got {n_actions}"
        raise ValueError(message)
    mask: Tensor
    if batch.mask is None:
        mask = torch.ones((batch.batch_size, n_actions), dtype=torch.bool, device=batch.policy_target.device)
    else:
        mask = batch.mask
    _validate_save_tensor(mask, "mask", torch.bool, 2)
    _validate_mask_width(mask, n_actions)
    tensors: dict[str, Tensor] = {
        "obs": batch.obs,
        "mask": mask,
        "policy_target": batch.policy_target,
        "value_target": batch.value_target,
    }
    return _save_tensors(tensors, path)
