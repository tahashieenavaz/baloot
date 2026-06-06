from baloot.loaders import load_torch
from baloot.typing import TorchTensor


def golden() -> TorchTensor:
    torch = load_torch()
    return (1.0 + torch.sqrt(torch.tensor(5.0))) / 2.0
