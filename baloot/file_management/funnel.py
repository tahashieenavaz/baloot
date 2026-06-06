from typing import Any, Optional
from baloot.loaders import load_pickle


def _save_file(*, instance: Any, file_location: str) -> bool:
    pickle = load_pickle()
    try:
        with open(file_location, "wb") as file:
            pickle.dump(instance, file)
        return True
    except (OSError, pickle.PickleError, AttributeError):
        return False


def _load_file(*, file_location: str) -> Any:
    pickle = load_pickle()
    try:
        with open(file_location, "rb") as file:
            return pickle.load(file)
    except (OSError, pickle.PickleError, EOFError):
        return None


def funnel(file_location: str, instance: Optional[Any] = None) -> Any:
    if instance is None:
        return _load_file(file_location=file_location)

    return _save_file(instance=instance, file_location=file_location)
