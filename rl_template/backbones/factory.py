from rl_template.backbones.attention import AttentionBackbone
from rl_template.backbones.base import Backbone
from rl_template.backbones.cnn import CNNBackbone
from rl_template.backbones.mlp import MLPBackbone
from rl_template.config import CNNModelConfig, MLPModelConfig, ModelConfig


def build_backbone(cfg: ModelConfig, obs_dim: int) -> Backbone:
    """Build the configured observation backbone.

    Returns:
        Configured feature extractor.

    """
    if isinstance(cfg, MLPModelConfig):
        return MLPBackbone(obs_dim, cfg.hidden_sizes)
    if isinstance(cfg, CNNModelConfig):
        return CNNBackbone(cfg.obs_shape, cfg.channels, cfg.blocks, obs_dim=obs_dim)
    return AttentionBackbone(
        cfg.obs_shape,
        cfg.d_model,
        cfg.n_heads,
        cfg.n_layers,
        obs_dim=obs_dim,
    )
