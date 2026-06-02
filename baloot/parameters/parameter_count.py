import device_maganement


def parameter_count(model: device_maganement.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if getattr(parameter, "requires_grad", False)
    )
