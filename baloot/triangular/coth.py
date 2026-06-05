from baloot.helpers import load_torch
from baloot.typing import TorchTensor


def coth(x: TorchTensor) -> TorchTensor:
    torch = load_torch()

    if isinstance(x, int):
        x = torch.tensor(x)

    return 1.0 / torch.tanh(x)
