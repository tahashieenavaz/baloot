import json
from typing import Dict, Any

_settings_cache: Dict[str, Any] | None = None


def settings(key: str, default: Any = None) -> Any:
    global _settings_cache

    if _settings_cache is None:
        with open("settings.json", "r", encoding="utf-8") as f:
            _settings_cache = json.load(f)

    current = _settings_cache
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default

    return current


def get_settings(*args, **kwargs):
    return settings(*args, **kwargs)
