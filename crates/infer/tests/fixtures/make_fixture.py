import json
from pathlib import Path
from typing import Final

import torch

from rl_template.backbones.mlp import MLPBackbone
from rl_template.export.onnx import export_onnx
from rl_template.model import PolicyValueNet

OBS_DIM: Final[int] = 7
N_ACTIONS: Final[int] = 5
FIXTURE_DIR: Final[Path] = Path(__file__).parent
OBS: Final[tuple[tuple[float, ...], ...]] = (
    (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    (-0.5, 0.25, 1.0, -1.25, 0.75, 0.0, 0.125),
    (1.0, -1.0, 0.5, 0.25, -0.75, 0.125, -0.25),
)
MASK: Final[tuple[tuple[bool, ...], ...]] = (
    (True, False, True, True, False),
    (True, True, False, True, True),
    (False, True, True, False, True),
)


def main() -> None:
    """Generate the deterministic ONNX model and its reference outputs."""
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]  # torch のスタブ起因
    network = PolicyValueNet(MLPBackbone(obs_dim=OBS_DIM, hidden_sizes=(8, 8)), n_actions=N_ACTIONS)
    observations = torch.tensor(OBS, dtype=torch.float32)
    action_mask = torch.tensor(MASK, dtype=torch.bool)

    export_onnx(network, OBS_DIM, N_ACTIONS, FIXTURE_DIR / "tiny.onnx")
    with torch.no_grad():
        logits, values = network(observations, action_mask)

    expected: dict[str, object] = {
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "obs": OBS,
        "mask": MASK,
        "logits": logits.tolist(),
        "values": values.tolist(),
    }
    expected_path = FIXTURE_DIR / "expected.json"
    expected_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
