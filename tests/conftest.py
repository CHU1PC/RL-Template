import pytest
import torch

SEED: int = 1234


@pytest.fixture
def _seed() -> None:
    torch.manual_seed(SEED)  # pyright: ignore[reportUnknownMemberType]  # torch の stub が戻り値型を Unknown とする
