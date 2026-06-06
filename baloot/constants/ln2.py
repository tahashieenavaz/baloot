from baloot.loaders import load_torch
from baloot.typing import TorchTensor


def ln2() -> TorchTensor:
    torch = load_torch()
    return torch.log(torch.tensor(2.0))
