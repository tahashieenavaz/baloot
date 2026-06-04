from typing import List
from baloot.typing import TorchTensor
from .hard_update import hard_update


def hard_update_all(*, sources: List[TorchTensor], targets: List[TorchTensor]) -> None:
    for source, target in zip(sources, targets):
        hard_update(source=source, target=target)
