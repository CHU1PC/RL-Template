from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, Self, final

import torch
from torch import Tensor


class MinibatchSource(Protocol):
    """Define the operations needed to create typed minibatches."""

    @property
    def batch_size(self) -> int:
        """Number of rows in the batch."""
        ...

    @property
    def device(self) -> torch.device:
        """Device used by the batch tensors."""
        ...

    def select(self, index: Tensor) -> Self:
        """Select rows by index and preserve the concrete batch type."""
        ...


def _leading_dimension(value: Tensor, field_name: str) -> int:
    if value.ndim == 0:
        message: str = f"{field_name} must have a leading dimension"
        raise ValueError(message)
    return value.shape[0]


def _validate_batch_dimension(expected: int, value: Tensor, field_name: str) -> None:
    actual: int = _leading_dimension(value, field_name)
    if actual != expected:
        message: str = f"{field_name} has batch dimension {actual}, expected {expected}"
        raise ValueError(message)


@final
@dataclass(frozen=True, slots=True)
class PpoBatch:
    """Store tensors used by a PPO update."""

    obs: Tensor = field()
    actions: Tensor = field()
    old_log_probs: Tensor = field()
    advantages: Tensor = field()
    returns: Tensor = field()
    mask: Tensor | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate that every tensor has the same leading dimension.

        Raises:
            ValueError: If the batch is empty or has an invalid action mask.
        """
        batch_size: int = _leading_dimension(self.obs, "obs")
        if batch_size == 0:
            message: str = "batch_size must be positive"
            raise ValueError(message)
        _validate_batch_dimension(batch_size, self.actions, "actions")
        _validate_batch_dimension(batch_size, self.old_log_probs, "old_log_probs")
        _validate_batch_dimension(batch_size, self.advantages, "advantages")
        _validate_batch_dimension(batch_size, self.returns, "returns")
        if self.mask is not None:
            _validate_batch_dimension(batch_size, self.mask, "mask")
            if not self.mask.any(dim=-1).all():
                message = "mask must contain at least one legal action per row"
                raise ValueError(message)

    @property
    def batch_size(self) -> int:
        """Number of observations in the batch."""
        return self.obs.shape[0]

    @property
    def device(self) -> torch.device:
        """Device used by the observation tensor."""
        return self.obs.device

    def select(self, index: Tensor) -> Self:
        """Return a batch containing rows selected by index."""
        selected_mask: Tensor | None = None if self.mask is None else self.mask[index]
        return type(self)(
            obs=self.obs[index],
            actions=self.actions[index],
            old_log_probs=self.old_log_probs[index],
            advantages=self.advantages[index],
            returns=self.returns[index],
            mask=selected_mask,
        )


@final
@dataclass(frozen=True, slots=True)
class AlphaZeroBatch:
    """Store tensors used by an AlphaZero update."""

    obs: Tensor = field()
    policy_target: Tensor = field()
    value_target: Tensor = field()
    mask: Tensor | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate that every tensor has the same leading dimension.

        Raises:
            ValueError: If the batch is empty or has an invalid action mask.
        """
        batch_size: int = _leading_dimension(self.obs, "obs")
        if batch_size == 0:
            message: str = "batch_size must be positive"
            raise ValueError(message)
        _validate_batch_dimension(batch_size, self.policy_target, "policy_target")
        _validate_batch_dimension(batch_size, self.value_target, "value_target")
        if self.mask is not None:
            _validate_batch_dimension(batch_size, self.mask, "mask")
            if not self.mask.any(dim=-1).all():
                message = "mask must contain at least one legal action per row"
                raise ValueError(message)

    @property
    def batch_size(self) -> int:
        """Number of observations in the batch."""
        return self.obs.shape[0]

    @property
    def device(self) -> torch.device:
        """Device used by the observation tensor."""
        return self.obs.device

    def select(self, index: Tensor) -> Self:
        """Return a batch containing rows selected by index."""
        selected_mask: Tensor | None = None if self.mask is None else self.mask[index]
        return type(self)(
            obs=self.obs[index],
            policy_target=self.policy_target[index],
            value_target=self.value_target[index],
            mask=selected_mask,
        )


def iterate_minibatches[BatchT: MinibatchSource](
    batch: BatchT,
    minibatch_size: int,
    generator: torch.Generator | None = None,
) -> Iterator[BatchT]:
    """Yield shuffled contiguous minibatches without dropping the remainder.

    Args:
        batch: Batch-like object implementing the minibatch source protocol.
        minibatch_size: Maximum number of rows in each yielded minibatch.
        generator: Optional random generator used for the row permutation.

    Yields:
        Concrete batches selected from the shuffled row permutation.

    Raises:
        ValueError: If ``minibatch_size`` is not positive.
    """
    if minibatch_size <= 0:
        message: str = "minibatch_size must be positive"
        raise ValueError(message)
    n: int = batch.batch_size
    permutation: Tensor = torch.randperm(n, generator=generator)
    permutation = permutation.to(batch.device)
    for start in range(0, n, minibatch_size):
        yield batch.select(permutation[start : start + minibatch_size])
