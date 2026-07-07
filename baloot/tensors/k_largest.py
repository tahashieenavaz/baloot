from baloot.typing import TorchTensor


def k_largest(
    x: TorchTensor,
    k: int,
    dim: int = -1,
    sorted: bool = True,
    index: bool = False,
    with_index: bool = False,
) -> TorchTensor:
    topk_data = x.topk(dim=dim, k=k, largest=True, sorted=sorted)

    if index and with_index:
        raise ValueError(
            "For baloot.k_largest set either index=True or with_index=True, not both."
        )

    if index:
        return topk_data.indices

    if with_index:
        return topk_data.values, topk_data.indices

    return topk_data.values
