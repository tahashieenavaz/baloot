from baloot.helpers import load_torch


def parameter_count(model) -> int:
    torch = load_torch()

    assert isinstance(model, torch.nn.Module)

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if getattr(parameter, "requires_grad", False)
    )
