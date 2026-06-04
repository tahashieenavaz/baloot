from baloot.helpers import load_torch
from baloot.typing import TorchTensor


def sqrt3() -> TorchTensor:
    torch = load_torch()
    return torch.sqrt(torch.tensor(5.0))
