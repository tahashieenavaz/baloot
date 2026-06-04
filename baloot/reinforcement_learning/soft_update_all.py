from typing import List
from baloot.typing import TorchModule
from .soft_update import soft_update


def soft_update_all(
    *, sources: List[TorchModule], targets: List[TorchModule], tau: float
) -> None:
    for source, target in zip(sources, targets):
        soft_update(source=source, target=target, tau=tau)
