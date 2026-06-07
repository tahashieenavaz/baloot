from baloot.typing import TorchTensor


def k_smallest(
    x: TorchTensor, k: int, dim: int = -1, sorted: bool = True
) -> TorchTensor:
    return x.topk(dim=dim, k=k, largest=False, sorted=sorted)
