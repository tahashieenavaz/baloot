from typing import Any


def render_template(*, template: str, target: str, **replacements: Any) -> bool:
    try:
        with open(template, "r", encoding="utf-8") as f:
            content = f.read()

        for key, value in replacements.items():
            content = content.replace(f"%{key}%", str(value))

        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

        return True
    except OSError:
        return False
