import baloot


def test_funnel_save_and_load(tmp_path):
    file = tmp_path / "obj.pkl"

    data = {"a": 1, "b": [1, 2, 3]}

    saved = baloot.funnel(str(file), data)
    assert saved is True

    loaded = baloot.funnel(str(file))
    assert loaded == data


def test_load_returns_none_if_missing(tmp_path):
    file = tmp_path / "missing.pkl"

    result = baloot.funnel(str(file))

    assert result is None


def test_save_invalid_location_returns_false(tmp_path):
    bad_path = tmp_path / "non_existent_dir" / "file.pkl"

    result = baloot.funnel(str(bad_path), {"x": 1})

    assert result is False


def test_render_template_basic(tmp_path):
    template = tmp_path / "template.txt"
    output = tmp_path / "rendered.txt"

    template.write_text("Hello %NAME%, you are %AGE% years old.", encoding="utf-8")

    result = baloot.render_template(str(template), str(output), NAME="Alice", AGE=30)

    assert result is True

    content = output.read_text(encoding="utf-8")
    assert content == "Hello Alice, you are 30 years old."


def test_render_template_missing_file(tmp_path):
    template = tmp_path / "missing.txt"
    output = tmp_path / "output.txt"

    result = baloot.render_template(str(template), str(output), NAME="Test")

    assert result is False
