from baloot.loaders import load_torch
from baloot.typing import TorchTensor


def arccot(x: TorchTensor | int) -> TorchTensor:
    torch = load_torch()

    if isinstance(x, int):
        x = torch.tensor(x)

    return (torch.pi / 2) - torch.arctan(x)
