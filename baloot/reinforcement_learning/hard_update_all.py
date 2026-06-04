from typing import List
from baloot.typing import TorchModule
from .hard_update import hard_update


def hard_update_all(*, sources: List[TorchModule], targets: List[TorchModule]) -> None:
    for source, target in zip(sources, targets):
        hard_update(source=source, target=target)
