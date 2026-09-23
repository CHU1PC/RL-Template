import torch
from torch import Tensor, nn

from rl_template.backbones.base import Backbone


class AttentionBackbone(Backbone):
    """Encode entity tokens and pool only tokens that are present.

    各トークンの最後の特徴量 ``obs[..., -1]`` は存在フラグであり、存在する
    トークンでは 1.0、パディングされたトークンでは 0.0 になる。このフラグを
    Rust 側のエンコーダと共有し、反転した値を Transformer の
    ``key_padding_mask`` に渡す。
    """

    def __init__(
        self,
        obs_shape: tuple[int, int],
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        obs_dim: int | None = None,
    ) -> None:
        """Initialize the attention backbone.

        Args:
            obs_shape: Token shape ``(number_of_tokens, token_features)``.
            d_model: Transformer embedding width.
            n_heads: Number of self-attention heads.
            n_layers: Number of Transformer encoder layers.
            obs_dim: Optional flattened dimension used to validate the shape.

        Raises:
            ValueError: If dimensions or the flattened shape are invalid.
        """
        super().__init__()
        if any(dimension <= 0 for dimension in obs_shape):
            message: str = "obs_shape must contain only positive values"
            raise ValueError(message)
        if d_model <= 0:
            message = "d_model must be positive"
            raise ValueError(message)
        if n_heads <= 0:
            message = "n_heads must be positive"
            raise ValueError(message)
        if n_layers <= 0:
            message = "n_layers must be positive"
            raise ValueError(message)
        if d_model % n_heads != 0:
            message = "d_model must be divisible by n_heads"
            raise ValueError(message)

        flattened_dim: int = obs_shape[0] * obs_shape[1]
        if obs_dim is not None and obs_dim != flattened_dim:
            message = "obs_dim must equal the product of obs_shape"
            raise ValueError(message)

        self.obs_shape = obs_shape
        self.obs_dim = flattened_dim if obs_dim is None else obs_dim
        self.out_dim = 256
        self.token_projection: nn.Linear = nn.Linear(obs_shape[1], d_model)
        layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.projection: nn.Linear = nn.Linear(d_model, self.out_dim)

    def forward(self, obs_flat: Tensor) -> Tensor:
        """Convert flattened token observations into pooled features.

        Returns:
            Features with shape ``[B, out_dim]``.
        """
        observations: Tensor = obs_flat.reshape(obs_flat.shape[0], *self.obs_shape)
        exists: Tensor = observations[..., -1].to(torch.bool)
        encoded: Tensor = self.encoder(
            self.token_projection(observations),
            src_key_padding_mask=~exists,
        )
        weights: Tensor = exists.unsqueeze(-1).to(encoded.dtype)
        count: Tensor = weights.sum(dim=1).clamp_min(1.0)
        pooled: Tensor = (encoded * weights).sum(dim=1) / count
        return self.projection(pooled)
