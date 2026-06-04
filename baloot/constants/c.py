from baloot.helpers import load_torch
from baloot.typing import TorchTensor


def c() -> TorchTensor:
    torch = load_torch()
    return torch.tensor(299792458)
