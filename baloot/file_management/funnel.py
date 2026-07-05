import logging
import os
from pathlib import Path
from typing import Any, overload, Union
from baloot.loaders import load_pickle

logger = logging.getLogger(__name__)

_MISSING = object()


@overload
def funnel(file_location: Union[str, os.PathLike]) -> Any: ...


@overload
def funnel(file_location: Union[str, os.PathLike], instance: Any) -> bool: ...


def funnel(file_location: Union[str, os.PathLike], instance: Any = _MISSING) -> Any:
    path = Path(file_location)
    if instance is _MISSING:
        return _load_file(file_path=path)

    return _save_file(instance=instance, file_path=path)


def _save_file(*, instance: Any, file_path: Path) -> bool:
    pickle = load_pickle()
    try:
        with file_path.open("wb") as file:
            pickle.dump(instance, file)
        return True
    except (OSError, pickle.PickleError, AttributeError) as e:
        logger.error(f"Failed to save pickle to {file_path}: {e}")
        return False


def _load_file(*, file_path: Path) -> Any:
    pickle = load_pickle()
    try:
        with file_path.open("rb") as file:
            return pickle.load(file)
    except (OSError, pickle.PickleError, EOFError) as e:
        logger.error(f"Failed to load pickle from {file_path}: {e}")
        return None
