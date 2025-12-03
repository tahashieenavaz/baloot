import random
import torch
import numpy
from typing import Any


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


def _try_seed_space(space: Any, seed: int) -> None:
    if space is None:
        return

    try:
        space.seed(seed)
    except (AttributeError, TypeError):
        pass


def seed_gymnasium(seed: int, environment: Any) -> None:
    seed = int(seed)
    if hasattr(environment, "reset"):
        environment.reset(seed=seed)
    _try_seed_space(getattr(environment, "action_space", None), seed)
    _try_seed_space(getattr(environment, "observation_space", None), seed)


def seed_gym(seed: int, environment: Any, seed_observation: bool = False) -> None:
    seed = int(seed)
    _try_seed_space(getattr(environment, "action_space", None), seed)
    if seed_observation:
        _try_seed_space(getattr(environment, "observation_space", None), seed)


def seed_everything(seed: int) -> None:
    seed = int(seed)
    seed_python(seed)
    seed_numpy(seed)
    seed_torch(seed)
