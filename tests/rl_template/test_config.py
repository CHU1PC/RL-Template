from pathlib import Path

import pytest
from pydantic import ValidationError

from rl_template.alphazero.config import AlphaZeroConfig
from rl_template.config import (
    AttentionModelConfig,
    CNNModelConfig,
    MLPModelConfig,
    TrainConfig,
    load_config,
)
from rl_template.ppo.config import PpoConfig

EXAMPLE_CONFIG: Path = Path(__file__).resolve().parents[2] / "configs" / "example.yml"
CNN_CHANNELS: int = 4
CNN_BLOCKS: int = 1
ATTENTION_D_MODEL: int = 8
ATTENTION_HEADS: int = 2
ATTENTION_LAYERS: int = 1


def test_example_config_loads_with_nested_defaults() -> None:
    config: TrainConfig = load_config(EXAMPLE_CONFIG)

    assert config.ppo == PpoConfig()
    assert config.alphazero == AlphaZeroConfig()
    assert config.model.backbone == "mlp"
    assert config.model.hidden_sizes == (256, 256)
    assert config.seed == 0


def test_unknown_config_field_is_forbidden(tmp_path: Path) -> None:
    path: Path = tmp_path / "unknown.yml"
    path.write_text(
        "algorithm: ppo\nobs_dim: 3\nn_actions: 2\nsamples_path: samples.safetensors\n"
        "export_path: model.onnx\nunknown: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="extra"):
        load_config(path)


def test_missing_required_config_field_raises_validation_error(tmp_path: Path) -> None:
    path: Path = tmp_path / "missing.yml"
    path.write_text(
        "algorithm: ppo\nobs_dim: 3\nn_actions: 2\nsamples_path: samples.safetensors\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="export_path"):
        load_config(path)


def test_missing_config_file_raises_clear_error(tmp_path: Path) -> None:
    path: Path = tmp_path / "does-not-exist.yml"

    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        load_config(path)


def test_mlp_model_variant_round_trips_from_yaml(tmp_path: Path) -> None:
    path: Path = tmp_path / "mlp.yml"
    path.write_text(
        "algorithm: ppo\nobs_dim: 7\nn_actions: 5\n"
        "model:\n  backbone: mlp\n  hidden_sizes: [8, 16]\n"
        "samples_path: samples.safetensors\nexport_path: model.onnx\n",
        encoding="utf-8",
    )

    config: TrainConfig = load_config(path)

    assert isinstance(config.model, MLPModelConfig)
    assert config.model.hidden_sizes == (8, 16)


def test_cnn_model_variant_round_trips_from_yaml(tmp_path: Path) -> None:
    path: Path = tmp_path / "cnn.yml"
    path.write_text(
        "algorithm: ppo\nobs_dim: 126\nn_actions: 5\n"
        "model:\n  backbone: cnn\n  obs_shape: [3, 6, 7]\n  channels: 4\n  blocks: 1\n"
        "samples_path: samples.safetensors\nexport_path: model.onnx\n",
        encoding="utf-8",
    )

    config: TrainConfig = load_config(path)

    assert isinstance(config.model, CNNModelConfig)
    assert config.model.obs_shape == (3, 6, 7)
    assert config.model.channels == CNN_CHANNELS
    assert config.model.blocks == CNN_BLOCKS


def test_attention_model_variant_round_trips_from_yaml(tmp_path: Path) -> None:
    path: Path = tmp_path / "attention.yml"
    path.write_text(
        "algorithm: ppo\nobs_dim: 40\nn_actions: 5\n"
        "model:\n  backbone: attention\n  obs_shape: [8, 5]\n  d_model: 8\n  n_heads: 2\n  n_layers: 1\n"
        "samples_path: samples.safetensors\nexport_path: model.onnx\n",
        encoding="utf-8",
    )

    config: TrainConfig = load_config(path)

    assert isinstance(config.model, AttentionModelConfig)
    assert config.model.obs_shape == (8, 5)
    assert config.model.d_model == ATTENTION_D_MODEL
    assert config.model.n_heads == ATTENTION_HEADS
    assert config.model.n_layers == ATTENTION_LAYERS


def test_cnn_model_variant_rejects_mismatched_observation_dimension(tmp_path: Path) -> None:
    path: Path = tmp_path / "cnn-mismatch.yml"
    path.write_text(
        "algorithm: ppo\nobs_dim: 125\nn_actions: 5\n"
        "model:\n  backbone: cnn\n  obs_shape: [3, 6, 7]\n"
        "samples_path: samples.safetensors\nexport_path: model.onnx\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=r"product of model\.obs_shape"):
        load_config(path)


def test_attention_model_variant_rejects_mismatched_observation_dimension(tmp_path: Path) -> None:
    path: Path = tmp_path / "attention-mismatch.yml"
    path.write_text(
        "algorithm: ppo\nobs_dim: 39\nn_actions: 5\n"
        "model:\n  backbone: attention\n  obs_shape: [8, 5]\n"
        "samples_path: samples.safetensors\nexport_path: model.onnx\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=r"product of model\.obs_shape"):
        load_config(path)
