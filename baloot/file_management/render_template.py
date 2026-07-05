from typing import Any
from string import Template
from pathlib import Path
from typing import Any


def _render_template(*, template: str, target: str, **replacements: Any) -> None:
    raw_content = Path(template).read_text(encoding="utf-8")
    tpl = Template(raw_content)
    final_content = tpl.safe_substitute(replacements)
    Path(target).write_text(final_content, encoding="utf-8")


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
