from baloot.helpers import load_torch
from baloot.typing import TorchTensor


def e() -> TorchTensor:
    torch = load_torch()
    return torch.tensor(torch.e)
