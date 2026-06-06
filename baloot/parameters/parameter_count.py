from baloot.loaders import load_torch
from baloot.typing import TorchModule


def parameter_count(model: TorchModule) -> int:
    torch = load_torch()

    assert isinstance(model, torch.nn.Module)

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if getattr(parameter, "requires_grad", False)
    )
