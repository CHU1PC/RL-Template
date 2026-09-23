from pathlib import Path

import torch

from rl_template.model import PolicyValueNet


def export_onnx(
    net: PolicyValueNet,
    obs_dim: int,
    n_actions: int,
    path: Path,
    opset: int = 18,
) -> Path:
    """Export a policy-value network to an ONNX file.

    Exported graphs require ``mask``; callers without action masking must pass
    an all-true boolean tensor, and Rust callers must ensure each row has a legal action.

    Args:
        net: Policy-value network to export.
        obs_dim: Number of observation features expected by the network.
        n_actions: Number of discrete actions expected by the network.
        path: Destination path for the ONNX file.
        opset: ONNX operator set version.

    Returns:
        The destination path.
    """
    was_training: bool = net.training
    was_fastpath_enabled: bool = torch.backends.mha.get_fastpath_enabled()
    net.eval()
    # 融合カーネルには ONNX の symbolic がないため、書き出し中だけ無効にする。
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        obs: torch.Tensor = torch.zeros(1, obs_dim, dtype=torch.float32)
        mask: torch.Tensor = torch.ones(1, n_actions, dtype=torch.bool)
        # dynamo エクスポーターは onnxscript を要求するため、従来のエクスポーターを使う。
        with torch.no_grad():
            torch.onnx.export(  # pyright: ignore[reportUnknownMemberType]  # PyTorch の export 型情報が部分的に未知。
                net,
                (obs, mask),
                path,
                input_names=["obs", "mask"],
                output_names=["logits", "value"],
                dynamic_axes={
                    "obs": {0: "batch"},
                    "mask": {0: "batch"},
                    "logits": {0: "batch"},
                    "value": {0: "batch"},
                },
                opset_version=opset,
                dynamo=False,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(was_fastpath_enabled)
        net.train(was_training)
    return path
