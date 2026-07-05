import time
from typing import List, Tuple, Optional
from baloot.loaders import load_torch
from baloot.device_management import acceleration_device
from baloot.typing import TorchModule, TorchTensor, TorchOptimizer, TorchDevice


def training_speed(
    modules: List[TorchModule],
    dummy_input: TorchTensor,
    device: TorchDevice = acceleration_device(),
    warmup_iterations: int = 10,
    measure_iterations: int = 50,
    optimizer: Optional[TorchOptimizer] = None,
    lr: float = 0.1,
) -> List[Tuple[TorchModule, float]]:
    torch = load_torch()

    if optimizer is None:
        optimizer = torch.optim.SGD

    assert isinstance(dummy_input, torch.Tensor)
    assert isinstance(device, torch.device)
    assert isinstance(optimizer, torch.optim.Optimizer)

    results = []
    dummy_input = dummy_input.to(device)

    for _, module in enumerate(modules):
        module = module.to(device)
        module.train()
        optimizer = optimizer(module.parameters(), lr=lr)
        for _ in range(warmup_iterations):
            optimizer.zero_grad()
            output = module(dummy_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(measure_iterations):
            optimizer.zero_grad()
            output = module(dummy_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time_ms = (total_time / measure_iterations) * 1000
        results.append((module, avg_time_ms))
        del optimizer, output, loss
        torch.cuda.empty_cache()

    results.sort(key=lambda x: x[1])
    return results
