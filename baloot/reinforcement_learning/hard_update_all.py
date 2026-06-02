from typing import List
from .hard_update import hard_update


def hard_update_all(*, sources: List, targets: List):
    for source, target in zip(sources, targets):
        hard_update(source=source, target=target)
