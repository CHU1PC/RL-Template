from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx

# onnxruntime は型スタブを同梱していない。
import onnxruntime  # pyright: ignore[reportMissingTypeStubs]
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

from rl_template.backbones.mlp import MLPBackbone
from rl_template.export.onnx import export_onnx
from rl_template.model import PolicyValueNet

OBS_DIM: int = 7
N_ACTIONS: int = 5
SINGLE_BATCH: int = 1
LARGE_BATCH: int = 5
ATOL: float = 1e-4


def _load_onnx_model(path: Path) -> onnx.ModelProto:
    return onnx.load(  # pyright: ignore[reportUnknownMemberType]  # ONNX の load 型情報が部分的に未知。
        str(path),
    )


def _check_onnx_model(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(  # pyright: ignore[reportUnknownMemberType]  # ONNX の checker 型情報が部分的に未知。
        model,
    )


def _create_inference_session(path: Path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )


def test_exported_onnx_loads_runs_and_matches_torch(tmp_path: Path) -> None:
    net = PolicyValueNet(MLPBackbone(OBS_DIM), N_ACTIONS)
    path = export_onnx(net, OBS_DIM, N_ACTIONS, tmp_path / "models" / "rl_v1.onnx")

    assert path.is_file()
    model = _load_onnx_model(path)
    _check_onnx_model(model)
    assert [input_.name for input_ in model.graph.input] == ["obs", "mask"]
    session = _create_inference_session(path)

    for batch_size in (SINGLE_BATCH, LARGE_BATCH):
        obs = torch.randn(batch_size, OBS_DIM)
        mask = torch.ones(batch_size, N_ACTIONS, dtype=torch.bool)
        torch_logits, torch_value = net(obs, mask)
        obs_input: NDArray[np.float32] = np.asarray(obs.numpy(), dtype=np.float32)
        mask_input: NDArray[np.bool_] = np.asarray(mask.numpy(), dtype=np.bool_)
        outputs = session.run(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # onnxruntime の run 型情報が部分的に未知。
            None,
            {"obs": obs_input, "mask": mask_input},
        )
        outputs = [
            np.asarray(outputs[0], dtype=np.float32),  # pyright: ignore[reportUnknownArgumentType]  # onnxruntime の出力型が部分的に未知。
            np.asarray(outputs[1], dtype=np.float32),  # pyright: ignore[reportUnknownArgumentType]  # onnxruntime の出力型が部分的に未知。
        ]
        assert outputs[0].shape == (batch_size, N_ACTIONS)
        assert outputs[1].shape == (batch_size,)
        np.testing.assert_allclose(
            outputs[0],
            torch_logits.detach().numpy(),
            atol=ATOL,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            outputs[1],
            torch_value.detach().numpy(),
            atol=ATOL,
            rtol=0.0,
        )
