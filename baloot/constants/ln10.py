from baloot.helpers import load_torch
from baloot.typing import TorchTensor


def ln10() -> TorchTensor:
    torch = load_torch()
    return torch.log(torch.tensor(10.0))
