import pickle
from typing import Any


def _save_thing(thing: Any, file_location: str) -> bool:
    try:
        with open(file_location, "wb") as file:
            pickle.dump(thing, file)
        return True
    except (OSError, pickle.PickleError, AttributeError):
        return False


def _load_thing(file_location: str) -> Any | None:
    try:
        with open(file_location, "rb") as file:
            return pickle.load(file)
    except (OSError, pickle.PickleError, EOFError):
        return None


def funnel(file_location: str, thing: Any | None = None) -> bool | None | Any:
    if thing is None:
        return _load_thing(file_location)

    return _save_thing(thing=thing, file_location=file_location)


def render_template(
    template_location: str, target_location: str, **replacements: Any
) -> bool:
    try:
        with open(template_location, "r", encoding="utf-8") as f:
            content = f.read()

        for key, value in replacements.items():
            content = content.replace(f"%{key}%", str(value))

        with open(target_location, "w", encoding="utf-8") as f:
            f.write(content)

        return True
    except OSError:
        return False
