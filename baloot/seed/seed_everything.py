import torch
import numpy
import random


def seed_torch(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_numpy(seed: int) -> None:
    seed = int(seed)
    numpy.random.seed(seed)


def seed_python(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)


def seed_everything(seed: int) -> None:
    seed = int(seed)
    seed_python(seed)
    seed_numpy(seed)
    seed_torch(seed)
