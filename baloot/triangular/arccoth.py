from baloot.helpers import load_torch
from baloot.typing import TorchTensor


def arccoth(x: TorchTensor) -> TorchTensor:
    torch = load_torch()
    assert isinstance(
        x, torch.Tensor
    ), f"baloot.arccoth accepts torch.Tensor {type(x)} was given."
    return 0.5 * torch.log((x + 1.0) / (x - 1.0))
