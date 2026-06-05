from baloot.helpers import load_torch
from baloot.typing import TorchTensor


def arccoth(x: TorchTensor | int) -> TorchTensor:
    torch = load_torch()

    if isinstance(x, int):
        x = torch.tensor(x)

    return 0.5 * torch.log((x + 1.0) / (x - 1.0))
