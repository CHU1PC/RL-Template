from torch import Tensor, nn

from rl_template.backbones.base import Backbone


class _ResidualBlock(nn.Module):
    """A residual convolutional block without batch-dependent normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu: nn.ReLU = nn.ReLU()

    def forward(self, features: Tensor) -> Tensor:
        """Apply two convolutions and add the residual input.

        Returns:
            Residual features with the same shape as the input.
        """
        residual: Tensor = features
        features = self.relu(self.conv1(features))
        features = self.conv2(features)
        return self.relu(features + residual)


class CNNBackbone(Backbone):
    """Convert ``[B, C, H, W]`` observations with residual convolutions."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        channels: int = 64,
        blocks: int = 4,
        obs_dim: int | None = None,
    ) -> None:
        """Initialize the convolutional backbone.

        Args:
            obs_shape: Observation shape ``(channels, height, width)``.
            channels: Number of channels in the convolutional trunk.
            blocks: Number of residual blocks.
            obs_dim: Optional flattened dimension used to validate the shape.

        Raises:
            ValueError: If dimensions or the flattened shape are invalid.
        """
        super().__init__()
        if any(dimension <= 0 for dimension in obs_shape):
            message: str = "obs_shape must contain only positive values"
            raise ValueError(message)
        if channels <= 0:
            message = "channels must be positive"
            raise ValueError(message)
        if blocks <= 0:
            message = "blocks must be positive"
            raise ValueError(message)

        flattened_dim: int = obs_shape[0] * obs_shape[1] * obs_shape[2]
        if obs_dim is not None and obs_dim != flattened_dim:
            message = "obs_dim must equal the product of obs_shape"
            raise ValueError(message)

        self.obs_shape = obs_shape
        self.obs_dim = flattened_dim if obs_dim is None else obs_dim
        self.out_dim = 256
        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(obs_shape[0], channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # バッチ正規化はバッチ統計量と running stats の差が学習中の ONNX 書き出しを不安定にするため使わない。
        self.blocks: nn.ModuleList = nn.ModuleList([_ResidualBlock(channels) for _ in range(blocks)])
        self.projection: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * obs_shape[1] * obs_shape[2], self.out_dim),
        )

    def forward(self, obs_flat: Tensor) -> Tensor:
        """Convert flattened observations into convolutional features.

        Returns:
            Features with shape ``[B, out_dim]``.
        """
        observations: Tensor = obs_flat.reshape(obs_flat.shape[0], *self.obs_shape)
        features: Tensor = self.stem(observations)
        for block in self.blocks:
            features = block(features)
        return self.projection(features)
