from baloot.helpers import load_torch


def hard_update(*, source, target):
    torch = load_torch()
    assert isinstance(source, torch.nn.Module)
    assert isinstance(target, torch.nn.Module)
    target.load_state_dict(source.state_dict())
