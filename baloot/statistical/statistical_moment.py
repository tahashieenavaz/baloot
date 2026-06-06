from baloot.typing import TorchTensor


def statistical_moment(
    x: TorchTensor, n: int, dim: int = -1, epsilon: float = 1e-8
) -> TorchTensor:
    if n == 0:
        return x

    if n == 1:
        return x.mean(dim=dim)

    if n == 2:
        return x.std(dim=dim)

    std = x.std(dim=dim)
    mean = x.mean(dim=dim)
    normalized_x = (x - mean) / (std + epsilon)
    return normalized_x.pow(n).mean(dim=dim)
