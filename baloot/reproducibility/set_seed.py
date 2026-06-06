from .seed_everything import seed_everything


def set_seed(*args, **kwargs) -> None:
    seed_everything(*args, **kwargs)
