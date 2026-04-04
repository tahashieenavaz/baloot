import pickle
from typing import Any


def _save_file(thing: Any, file_location: str) -> bool:
    try:
        with open(file_location, "wb") as file:
            pickle.dump(thing, file)
        return True
    except (OSError, pickle.PickleError, AttributeError):
        return False


def _load_file(file_location: str) -> Any | None:
    try:
        with open(file_location, "rb") as file:
            return pickle.load(file)
    except (OSError, pickle.PickleError, EOFError):
        return None


def funnel(file_location: str, thing: Any | None = None) -> bool | None | Any:
    if thing is None:
        return _load_file(file_location)

    return _save_file(thing=thing, file_location=file_location)
