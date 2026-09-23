from pydantic import BaseModel, ConfigDict, Field


class AlphaZeroConfig(BaseModel):
    """Configure updates; lr and weight_decay are for the caller-built optimizer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lr: float = Field(default=1e-3, gt=0.0, description="Learning rate for the caller-built optimizer.")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="Weight decay for the caller-built optimizer.")
    value_coef: float = Field(default=1.0, ge=0.0, description="Value loss coefficient.")
    epochs: int = Field(default=1, ge=1, description="Number of passes over the batch.")
    minibatch_size: int = Field(default=256, ge=1, description="Maximum number of rows per minibatch.")
    max_grad_norm: float = Field(default=1.0, gt=0.0, description="Maximum global gradient norm.")
