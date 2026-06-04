from baloot.helpers import load_torch
from baloot.typing import TorchTensor


def arccot(x: TorchTensor) -> TorchTensor:
    torch = load_torch()
    assert isinstance(
        x, torch.Tensor
    ), f"baloot.arccot accepts torch.Tensor {type(x)} was given."
    return (torch.pi / 2) - torch.arctan(x)
