from .seed_cuda import seed_cuda
from .seed_numpy import seed_numpy
from .seed_torch import seed_torch
from .seed_python import seed_python


def seed_everything(seed: int) -> None:
    seed = int(seed)
    seed_python(seed)
    seed_numpy(seed)
    seed_torch(seed)
    seed_cuda(seed)
