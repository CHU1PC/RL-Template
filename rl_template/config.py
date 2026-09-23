from math import prod
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

from rl_template.alphazero.config import AlphaZeroConfig
from rl_template.ppo.config import PpoConfig


class MLPModelConfig(BaseModel):
    """Configure the MLP observation backbone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backbone: Literal["mlp"] = Field(default="mlp", description="Backbone discriminator.")
    hidden_sizes: tuple[int, ...] = Field(
        default=(256, 256),
        min_length=1,
        description="Widths of the shared policy-value network hidden layers.",
    )

    @field_validator("hidden_sizes")
    @classmethod
    def validate_hidden_sizes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject empty or non-positive hidden-layer widths.

        Returns:
            Validated hidden-layer widths.

        Raises:
            ValueError: If a width is not positive.
        """
        if any(hidden_size <= 0 for hidden_size in value):
            message: str = "hidden_sizes must contain only positive values"
            raise ValueError(message)
        return value


class CNNModelConfig(BaseModel):
    """Configure the residual convolutional observation backbone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backbone: Literal["cnn"] = Field(default="cnn", description="Backbone discriminator.")
    obs_shape: tuple[int, int, int] = Field(description="Observation shape as channels, height, and width.")
    channels: int = Field(default=64, gt=0, description="Convolutional channel width.")
    blocks: int = Field(default=4, gt=0, description="Number of residual convolutional blocks.")

    @field_validator("obs_shape")
    @classmethod
    def validate_obs_shape(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        """Reject non-positive spatial dimensions.

        Returns:
            Validated observation shape.

        Raises:
            ValueError: If a dimension is not positive.
        """
        if any(dimension <= 0 for dimension in value):
            message: str = "obs_shape must contain only positive values"
            raise ValueError(message)
        return value


class AttentionModelConfig(BaseModel):
    """Configure the Transformer attention observation backbone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backbone: Literal["attention"] = Field(default="attention", description="Backbone discriminator.")
    obs_shape: tuple[int, int] = Field(description="Observation shape as token count and token width.")
    d_model: int = Field(default=128, gt=0, description="Transformer embedding width.")
    n_heads: int = Field(default=4, gt=0, description="Number of attention heads.")
    n_layers: int = Field(default=2, gt=0, description="Number of Transformer encoder layers.")

    @field_validator("obs_shape")
    @classmethod
    def validate_obs_shape(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Reject non-positive token dimensions.

        Returns:
            Validated observation shape.

        Raises:
            ValueError: If a dimension is not positive.
        """
        if any(dimension <= 0 for dimension in value):
            message: str = "obs_shape must contain only positive values"
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def validate_attention_width(self) -> Self:
        """Require an embedding width divisible by the head count.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If the embedding width is not divisible by the head count.
        """
        if self.d_model % self.n_heads != 0:
            message: str = "d_model must be divisible by n_heads"
            raise ValueError(message)
        return self


type ModelConfig = Annotated[
    MLPModelConfig | CNNModelConfig | AttentionModelConfig,
    Field(discriminator="backbone"),
]


class TrainConfig(BaseSettings):
    """Define the dimensions, paths, and update settings for one training run."""

    model_config = SettingsConfigDict(frozen=True, extra="forbid")

    algorithm: Literal["ppo", "alphazero"] = Field(description="Training algorithm to run.")
    obs_dim: int = Field(gt=0, description="Number of encoded observation features.")
    n_actions: int = Field(gt=0, description="Number of discrete actions.")
    model: ModelConfig = Field(default=MLPModelConfig(), description="Observation backbone configuration.")
    samples_path: Path = Field(description="Path to the safetensors samples file.")
    export_path: Path = Field(description="Destination path for the exported ONNX model.")
    seed: int = Field(default=0, ge=0, description="Non-negative random seed for reproducible training.")
    ppo: PpoConfig = Field(default=PpoConfig(), description="PPO update hyperparameters.")
    alphazero: AlphaZeroConfig = Field(default=AlphaZeroConfig(), description="AlphaZero update hyperparameters.")

    @model_validator(mode="after")
    def validate_observation_shape(self) -> Self:
        """Ensure structured backbones cover the configured flat observation.

        Returns:
            The validated training configuration.

        Raises:
            ValueError: If the structured and flat dimensions differ.
        """
        if isinstance(self.model, (CNNModelConfig, AttentionModelConfig)):
            expected_dim: int = prod(self.model.obs_shape)
            if expected_dim != self.obs_dim:
                message: str = "the product of model.obs_shape must equal obs_dim"
                raise ValueError(message)
        return self


def load_config(path: Path) -> TrainConfig:
    """Load and validate a training configuration from YAML.

    Args:
        path: YAML configuration file to read.

    Returns:
        Validated training configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if not path.is_file():
        message: str = f"configuration file not found: {path}"
        raise FileNotFoundError(message)
    yaml_source: YamlConfigSettingsSource = YamlConfigSettingsSource(
        TrainConfig,
        yaml_file=path,
        yaml_file_encoding="utf-8",
    )
    return TrainConfig.model_validate(yaml_source())
