from baloot.loaders import load_torch
from baloot.typing import TorchTensor


def haversin(x: TorchTensor) -> TorchTensor:
    torch = load_torch()

    if isinstance(x, (int, float)):
        x = torch.tensor(x)

    return 0.5 * (1.0 - torch.cos(x))
