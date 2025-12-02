import pickle


def _save_thing(thing, file_location: str) -> bool:
    try:
        with open(file_location, "wb") as file_handler:
            pickle.dump(thing, file_handler)
        return True
    except:
        return False


def _load_thing(file_location: str) -> object | None:
    try:
        with open(file_location, "rb") as file_handler:
            return pickle.load(file_handler)
    except:
        return None


def funnel(file_location: str, thing: object | None = None) -> bool | None | object:
    if thing is None:
        return _load_thing(file_location)

    return _save_thing(thing=thing, file_location=file_location)


def render_template(
    template_location: str, target_location: str, **replacements
) -> bool:
    with open(template_location, "r") as f:
        content = f.read()

    for key, value in replacements.items():
        content = content.replace(f"%{key}%", value)

    with open(target_location, "w") as f:
        f.write(content)

