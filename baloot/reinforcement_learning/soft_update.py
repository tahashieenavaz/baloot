from baloot.helpers import load_torch
from baloot.typing import TorchModule


def soft_update(*, source: TorchModule, target: TorchModule, tau: float) -> None:
    torch = load_torch()
    assert isinstance(source, torch.nn.Module)
    assert isinstance(target, torch.nn.Module)
    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters()
    ):
        target_parameter.data.mul_(1 - tau).add_(source_parameter.data, alpha=tau)
