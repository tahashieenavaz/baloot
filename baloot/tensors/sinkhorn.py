from baloot.loaders import load_torch
from baloot.typing import TorchTensor


def sinkhorn(
    x: TorchTensor,
    iterations: int = 10,
    epsilon: float = 1e-6,
    dim1: int = -2,
    dim2: int = -1,
) -> TorchTensor:
    torch = load_torch()
    x = torch.exp(x)
    for _ in range(iterations):
        x = x / (x.sum(dim=dim1, keepdim=True) + epsilon)
        x = x / (x.sum(dim=dim2, keepdim=True) + epsilon)
    return x
