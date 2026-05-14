from .seed_cuda import seed_cuda
from .seed_numpy import seed_numpy
from .seed_torch import seed_torch
from .seed_python import seed_python


def seed_everything(
    seed: int,
    python: bool = True,
    numpy: bool = True,
    torch: bool = True,
    cuda: bool = True,
) -> None:
    seed = int(seed)

    if python:
        seed_python(seed)

    if numpy:
        seed_numpy(seed)

    if torch:
        seed_torch(seed)

    if cuda:
        seed_cuda(seed)
