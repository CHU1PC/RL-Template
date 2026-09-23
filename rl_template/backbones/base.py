from torch import Tensor, nn


class Backbone(nn.Module):
    """共通の観測バックボーンの基底クラス。."""

    obs_dim: int
    out_dim: int

    def forward(self, obs_flat: Tensor) -> Tensor:
        """平坦な観測を特徴量へ変換する。."""
        raise NotImplementedError
