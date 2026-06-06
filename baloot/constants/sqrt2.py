from baloot.loaders import load_torch
from baloot.typing import TorchTensor


def sqrt2() -> TorchTensor:
    torch = load_torch()
    return torch.sqrt(torch.tensor(2.0))
