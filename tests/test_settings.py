import json
import builtins
import pytest

from baloot import reload_settings, settings


@pytest.fixture
def fake_settings_file(tmp_path):
    data = {"app": {"name": "MyApp", "version": "1.0"}, "debug": True, "port": 5000}

    file = tmp_path / "settings.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    return file


@pytest.fixture(autouse=True)
def isolate_cwd(tmp_path, monkeypatch):
    """Ensure each test runs in a clean directory"""
    monkeypatch.chdir(tmp_path)
    reload_settings()


def test_reads_basic_value(fake_settings_file):
    assert settings("debug") is True
    assert settings("port") == 5000


def test_reads_nested_value(fake_settings_file):
    assert settings("app.name") == "MyApp"
    assert settings("app.version") == "1.0"


def test_returns_default_if_missing(fake_settings_file):
    assert settings("missing", "fallback") == "fallback"
    assert settings("app.missing", None) is None


def test_caches_file(fake_settings_file, monkeypatch):
    # First read loads it
    assert settings("port") == 5000

    # If file is re-read, this will crash
    def explode(*args, **kwargs):
        raise RuntimeError("File was read twice")

    monkeypatch.setattr(builtins, "open", explode)

    assert settings("debug") is True


def test_reload_settings_reloads(fake_settings_file):
    assert settings("port") == 5000

    fake_settings_file.write_text(json.dumps({"port": 9000}), encoding="utf-8")

    assert settings("port") == 5000

    reload_settings()
    assert settings("port") == 9000
