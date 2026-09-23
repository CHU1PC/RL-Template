from collections.abc import Sequence

from torch import Tensor, nn

from rl_template.backbones.base import Backbone


class MLPBackbone(Backbone):
    """平坦な観測を多層パーセプトロンで特徴量へ変換する。."""

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int] = (256, 256)) -> None:
        """Initialize the MLP backbone.

        Args:
            obs_dim: Number of flattened observation features.
            hidden_sizes: Widths of the hidden layers.

        Raises:
            ValueError: If a dimension is invalid or no hidden layer exists.
        """
        super().__init__()
        if obs_dim <= 0:
            message: str = "obs_dim must be positive"
            raise ValueError(message)
        if not hidden_sizes:
            message = "hidden_sizes must not be empty"
            raise ValueError(message)
        if any(hidden_size <= 0 for hidden_size in hidden_sizes):
            message = "hidden_sizes must contain only positive values"
            raise ValueError(message)

        self.obs_dim = obs_dim
        self.obs_shape: tuple[int, ...] = (obs_dim,)
        self.out_dim = hidden_sizes[-1]
        layers: list[nn.Module] = []
        in_features: int = obs_dim
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(in_features, hidden_size), nn.ReLU()))
            in_features = hidden_size
        self.network: nn.Sequential = nn.Sequential(*layers)

    def forward(self, obs_flat: Tensor) -> Tensor:
        """Convert ``[B, obs_dim]`` observations into ``[B, out_dim]`` features.

        Returns:
            Features with shape ``[B, out_dim]``.

        Raises:
            ValueError: If the last observation dimension does not match ``obs_dim``.
        """
        if obs_flat.shape[-1] != self.obs_dim:
            message: str = f"obs_flat must have shape [B, {self.obs_dim}]"
            raise ValueError(message)
        return self.network(obs_flat)
