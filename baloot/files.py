import pickle


def save_object(object, file_location: str):
    with open(file_location, "wb") as file_handler:
        pickle.dump(object, file_handler)


def load_object(file_location: str):
    with open(file_location, "rb") as file_handler:
        return pickle.load(file_handler)


def duplicate(from_location, to_location, **replacements):
    with open(from_location, "r") as f:
        content = f.read()

    for key, value in replacements.items():
        content = content.replace(f"%{key}%", value)

    with open(to_location, "w") as f:
        f.write(content)
