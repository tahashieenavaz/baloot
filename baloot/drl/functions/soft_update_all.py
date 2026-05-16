import torch
from typing import List
from itertools import chain


def soft_update_all(
    *, sources: List[torch.nn.Module], targets: List[torch.nn.Module], tau: float
):
    for source_parameter, target_parameter in chain.from_iterable(
        *(source.parameters() for source in sources),
        *(target.parameters() for target in targets),
    ):
        target_parameter.data.mul_(1 - tau).add_(source_parameter.data, alpha=tau)
