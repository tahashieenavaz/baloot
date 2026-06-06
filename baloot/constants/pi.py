from baloot.loaders import load_torch
from baloot.typing import TorchTensor


def pi() -> TorchTensor:
    torch = load_torch()
    return torch.tensor(torch.pi)
