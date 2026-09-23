from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx

# onnxruntime は型スタブを同梱していない。
import onnxruntime  # pyright: ignore[reportMissingTypeStubs]
import pytest
import torch
from torch import Tensor

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rl_template.backbones.base import Backbone

from rl_template.alphazero.config import AlphaZeroConfig
from rl_template.alphazero.update import AlphaZeroMetrics, alphazero_update
from rl_template.backbones.attention import AttentionBackbone
from rl_template.backbones.cnn import CNNBackbone
from rl_template.backbones.mlp import MLPBackbone
from rl_template.data.batch import AlphaZeroBatch, PpoBatch
from rl_template.export.onnx import export_onnx
from rl_template.model import PolicyValueNet, masked_log_softmax
from rl_template.ppo.config import PpoConfig
from rl_template.ppo.update import PpoMetrics, ppo_update

N_ACTIONS: int = 4
TRAINING_BATCH: int = 16
TRAINING_STEPS: int = 8
MLP_OBS_DIM: int = 7
CNN_OBS_DIM: int = 3 * 6 * 7
ATTENTION_OBS_DIM: int = 8 * 5
CNN_CHANNELS: int = 4
CNN_BLOCKS: int = 1
ATTENTION_D_MODEL: int = 8
ATTENTION_HEADS: int = 2
ATTENTION_LAYERS: int = 1
EXPECTED_IO_COUNT: int = 2
pytestmark = pytest.mark.usefixtures("_seed")


def _build_mlp() -> PolicyValueNet:
    backbone: Backbone = MLPBackbone(MLP_OBS_DIM, hidden_sizes=(8, 8))
    return PolicyValueNet(backbone, N_ACTIONS)


def _build_cnn() -> PolicyValueNet:
    backbone: Backbone = CNNBackbone((3, 6, 7), channels=CNN_CHANNELS, blocks=CNN_BLOCKS)
    return PolicyValueNet(backbone, N_ACTIONS)


def _build_attention() -> PolicyValueNet:
    backbone: Backbone = AttentionBackbone(
        (8, 5), d_model=ATTENTION_D_MODEL, n_heads=ATTENTION_HEADS, n_layers=ATTENTION_LAYERS
    )
    return PolicyValueNet(backbone, N_ACTIONS)


def _attention_observations() -> Tensor:
    observations: Tensor = torch.randn(TRAINING_BATCH, 8, 5)
    observations[..., -1] = 1.0
    return observations.reshape(TRAINING_BATCH, ATTENTION_OBS_DIM)


def _random_observations(obs_dim: int) -> Tensor:
    return torch.randn(TRAINING_BATCH, obs_dim)


def _assert_forward_shapes(factory: Callable[[], PolicyValueNet], obs: Tensor) -> None:
    net: PolicyValueNet = factory()
    features: Tensor = net.backbone(obs)
    logits: Tensor
    value: Tensor
    logits, value = net(obs, torch.ones(obs.shape[0], N_ACTIONS, dtype=torch.bool))

    assert features.shape == (obs.shape[0], net.backbone.out_dim)
    assert logits.shape == (obs.shape[0], N_ACTIONS)
    assert value.shape == (obs.shape[0],)


def _load_onnx_model(path: Path) -> onnx.ModelProto:
    return onnx.load(  # pyright: ignore[reportUnknownMemberType]  # ONNX の load 型情報が部分的に未知。
        str(path),
    )


def _assert_onnx_export(factory: Callable[[], PolicyValueNet], obs_dim: int, tmp_path: Path) -> None:
    net: PolicyValueNet = factory()
    path: Path = export_onnx(net, obs_dim, N_ACTIONS, tmp_path / "model.onnx")
    model: onnx.ModelProto = _load_onnx_model(path)
    onnx.checker.check_model(  # pyright: ignore[reportUnknownMemberType]  # ONNX の checker 型情報が部分的に未知。
        model,
    )

    input_names: list[str] = [input_.name for input_ in model.graph.input]
    output_names: list[str] = [output.name for output in model.graph.output]
    assert input_names == ["obs", "mask"]
    assert output_names == ["logits", "value"]
    assert len(model.graph.input) == EXPECTED_IO_COUNT
    assert len(model.graph.output) == EXPECTED_IO_COUNT
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert model.graph.input[1].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert model.graph.output[1].type.tensor_type.shape.dim[0].dim_param == "batch"

    session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    for batch_size in (1, 3):
        observations: Tensor = torch.randn(batch_size, obs_dim)
        mask: Tensor = torch.ones(batch_size, N_ACTIONS, dtype=torch.bool)
        with torch.no_grad():
            torch_logits: Tensor
            torch_value: Tensor
            torch_logits, torch_value = net(observations, mask)
        obs_input: NDArray[np.float32] = np.asarray(observations.numpy(), dtype=np.float32)
        mask_input: NDArray[np.bool_] = np.asarray(mask.numpy(), dtype=np.bool_)
        outputs: Sequence[object] = session.run(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # onnxruntime の run 型情報が未知。
            None,
            {"obs": obs_input, "mask": mask_input},
        )
        output_logits: NDArray[np.float32] = np.asarray(
            outputs[0],  # pyright: ignore[reportUnknownArgumentType]  # onnxruntime の出力型が未知。
            dtype=np.float32,
        )
        output_value: NDArray[np.float32] = np.asarray(
            outputs[1],  # pyright: ignore[reportUnknownArgumentType]  # onnxruntime の出力型が未知。
            dtype=np.float32,
        )

        assert output_logits.shape == (batch_size, N_ACTIONS)
        assert output_value.shape == (batch_size,)
        np.testing.assert_allclose(output_logits, torch_logits.numpy(), atol=1e-4, rtol=0.0)
        np.testing.assert_allclose(output_value, torch_value.numpy(), atol=1e-4, rtol=0.0)


def _assert_ppo_loss_decreases(factory: Callable[[], PolicyValueNet], obs: Tensor) -> None:
    net: PolicyValueNet = factory()
    mask: Tensor = torch.ones(TRAINING_BATCH, N_ACTIONS, dtype=torch.bool)
    actions: Tensor = torch.zeros(TRAINING_BATCH, dtype=torch.int64)
    with torch.no_grad():
        initial_logits: Tensor
        initial_logits, _ = net(obs, mask)
        old_log_probs: Tensor = masked_log_softmax(initial_logits, mask).gather(1, actions.unsqueeze(1)).squeeze(1)
    batch: PpoBatch = PpoBatch(
        obs=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=torch.ones(TRAINING_BATCH),
        returns=torch.zeros(TRAINING_BATCH),
        mask=mask,
    )
    optimizer: torch.optim.Optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    config: PpoConfig = PpoConfig(epochs=1, minibatch_size=TRAINING_BATCH, entropy_coef=0.0)
    generator: torch.Generator = torch.Generator().manual_seed(0)
    losses: list[float] = []
    for _ in range(TRAINING_STEPS):
        metrics: PpoMetrics = ppo_update(net, optimizer, batch, config, generator)
        losses.append(metrics.policy_loss)

    assert losses[-1] < losses[0]


def _assert_alphazero_loss_decreases(factory: Callable[[], PolicyValueNet], obs: Tensor) -> None:
    net: PolicyValueNet = factory()
    mask: Tensor = torch.ones(TRAINING_BATCH, N_ACTIONS, dtype=torch.bool)
    policy_target: Tensor = torch.zeros(TRAINING_BATCH, N_ACTIONS)
    policy_target[:, 0] = 1.0
    batch: AlphaZeroBatch = AlphaZeroBatch(
        obs=obs,
        policy_target=policy_target,
        value_target=torch.zeros(TRAINING_BATCH),
        mask=mask,
    )
    optimizer: torch.optim.Optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    config: AlphaZeroConfig = AlphaZeroConfig(
        epochs=1,
        minibatch_size=TRAINING_BATCH,
        value_coef=0.0,
    )
    generator: torch.Generator = torch.Generator().manual_seed(0)
    losses: list[float] = []
    for _ in range(TRAINING_STEPS):
        metrics: AlphaZeroMetrics = alphazero_update(net, optimizer, batch, config, generator)
        losses.append(metrics.policy_loss)

    assert losses[-1] < losses[0]


def test_mlp_forward_shapes() -> None:
    _assert_forward_shapes(_build_mlp, _random_observations(MLP_OBS_DIM))


def test_cnn_forward_shapes() -> None:
    _assert_forward_shapes(_build_cnn, _random_observations(CNN_OBS_DIM))


def test_attention_forward_shapes() -> None:
    _assert_forward_shapes(_build_attention, _attention_observations())


def test_mlp_onnx_export_runs_for_dynamic_batches(tmp_path: Path) -> None:
    _assert_onnx_export(_build_mlp, MLP_OBS_DIM, tmp_path)


def test_cnn_onnx_export_runs_for_dynamic_batches(tmp_path: Path) -> None:
    _assert_onnx_export(_build_cnn, CNN_OBS_DIM, tmp_path)


def test_attention_onnx_export_runs_for_dynamic_batches(tmp_path: Path) -> None:
    _assert_onnx_export(_build_attention, ATTENTION_OBS_DIM, tmp_path)


def test_mlp_ppo_loss_decreases() -> None:
    _assert_ppo_loss_decreases(_build_mlp, _random_observations(MLP_OBS_DIM))


def test_cnn_ppo_loss_decreases() -> None:
    _assert_ppo_loss_decreases(_build_cnn, _random_observations(CNN_OBS_DIM))


def test_attention_ppo_loss_decreases() -> None:
    _assert_ppo_loss_decreases(_build_attention, _attention_observations())


def test_mlp_alphazero_loss_decreases() -> None:
    _assert_alphazero_loss_decreases(_build_mlp, _random_observations(MLP_OBS_DIM))


def test_cnn_alphazero_loss_decreases() -> None:
    _assert_alphazero_loss_decreases(_build_cnn, _random_observations(CNN_OBS_DIM))


def test_attention_alphazero_loss_decreases() -> None:
    _assert_alphazero_loss_decreases(_build_attention, _attention_observations())


def test_attention_padding_token_perturbation_does_not_change_output() -> None:
    net: PolicyValueNet = _build_attention()
    observations: Tensor = _attention_observations()[:2]
    observations.reshape(2, 8, 5)[:, 3, -1] = 0.0
    observations_perturbed: Tensor = observations.clone()
    reshaped: Tensor = observations_perturbed.reshape(2, 8, 5)
    reshaped[:, 3, -1] = 0.0
    reshaped[:, 3, :4] += 100.0
    mask: Tensor = torch.ones(2, N_ACTIONS, dtype=torch.bool)

    with torch.no_grad():
        original_logits: Tensor
        original_value: Tensor
        perturbed_logits: Tensor
        perturbed_value: Tensor
        original_logits, original_value = net(observations, mask)
        perturbed_logits, perturbed_value = net(observations_perturbed, mask)

    assert torch.equal(original_logits, perturbed_logits)
    assert torch.equal(original_value, perturbed_value)
