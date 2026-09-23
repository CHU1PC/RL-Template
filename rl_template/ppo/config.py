from pydantic import BaseModel, ConfigDict, Field


class PpoConfig(BaseModel):
    """Configure updates; lr and weight_decay are for the caller-built optimizer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lr: float = Field(default=3e-4, gt=0.0, description="Learning rate for the caller-built optimizer.")
    weight_decay: float = Field(default=0.0, ge=0.0, description="Weight decay for the caller-built optimizer.")
    clip_eps: float = Field(default=0.2, gt=0.0, description="PPO policy ratio clipping width.")
    value_coef: float = Field(default=0.5, ge=0.0, description="Value loss coefficient.")
    entropy_coef: float = Field(default=0.01, ge=0.0, description="Entropy bonus coefficient.")
    epochs: int = Field(default=4, ge=1, description="Number of passes over the batch.")
    minibatch_size: int = Field(default=256, ge=1, description="Maximum number of rows per minibatch.")
    max_grad_norm: float = Field(default=0.5, gt=0.0, description="Maximum global gradient norm.")
