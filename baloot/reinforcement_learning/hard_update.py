from baloot.loaders import load_torch
from baloot.typing import TorchModule


def hard_update(*, source: TorchModule, target: TorchModule) -> None:
    torch = load_torch()
    assert isinstance(source, torch.nn.Module)
    assert isinstance(target, torch.nn.Module)
    target.load_state_dict(source.state_dict())
