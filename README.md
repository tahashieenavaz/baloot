# Baloot

Small, focused helpers I reuse across experiments and projects. Everything is pure Python, has no runtime dependencies, and is geared toward developer ergonomics.

## Installation

- From PyPI: `pip install baloot`
- Local editable: `pip install -e .`

## Quick start

```python
from baloot import acceleration_device, funnel, seed_everything, settings

device = acceleration_device()  # torch: picks cuda -> mps -> cpu
seed_everything(42)             # python, numpy, torch all aligned
funnel("model.pkl", model)      # save a pickled object
print(settings("training.batch_size", default=32))
```

## API reference

All functions are imported from the top-level `baloot` package for convenience.

### Torch helpers (`baloot.torch`)

- `parameter_count(model) -> int`: Counts parameters with `requires_grad=True`. Useful for reporting trainable footprint.
- `acceleration_device() -> torch.device`: Picks the best available accelerator in order `cuda`, `mps`, then `cpu`. Raises `ImportError` if `torch` is missing.

### File helpers (`baloot.files`)

- `funnel(file_location, thing=None) -> Any | bool | None`: Two-way utility for pickled artifacts. Pass an object to save it (`True` on success); pass `thing=None` to load (`None` on failure).
- `render_template(template_location, target_location, **replacements) -> bool`: Lightweight string substitution. Replaces `%key%` placeholders in the template and writes the rendered file. Returns `True` on success, `False` on I/O errors.
- `_save_thing(thing, file_location) -> bool` and `_load_thing(file_location) -> Any | None`: Internal helpers backing `funnel`; exposed for completeness but intended for private use.

### Seeding helpers (`baloot.seed`)

- `seed_python(seed: int) -> None`: Seeds Python’s `random`.
- `seed_numpy(seed: int) -> None`: Seeds NumPy RNG.
- `seed_torch(seed: int) -> None`: Seeds torch (CPU + CUDA). Forces deterministic/cuDNN-safe settings.
- `seed_gymnasium(seed: int, environment) -> None`: Seeds Gymnasium env via `reset(seed=...)` and its action/observation spaces when present.
- `seed_gym(seed: int, environment, seed_observation: bool = False) -> None`: Seeds classic Gym action space; optionally observation space.
- `seed_everything(seed: int) -> None`: Calls Python, NumPy, and torch seeders for a single consistent seed.

### Settings helpers (`baloot.settings`)

- `settings(key: str, default=None) -> Any`: Loads `settings.json` once and provides dotted-path lookup (e.g., `training.lr`). Returns `default` when missing.
- `reload_settings() -> None`: Clears the cached `settings.json` so the next call reloads it.
- `get_settings(*args, **kwargs)`: Alias of `settings` for legacy imports.

## Patterns and examples

- **Save & load artifacts**:

  ```python
  from baloot import funnel

  funnel("checkpoints/run1.pkl", model_state)  # save
  restored = funnel("checkpoints/run1.pkl")    # load
  ```

- **Render simple templates**:

  ```python
  from baloot import render_template

  render_template(
      "templates/train.sh.tpl",
      "scripts/train.sh",
      run_name="exp_042",
      lr=3e-4,
  )
  ```

- **Control reproducibility end-to-end**:

  ```python
  from baloot import seed_everything, seed_gymnasium

  seed_everything(1337)
  env = make_env()
  seed_gymnasium(1337, env)
  ```

## Notes

- `settings.json` is expected in the working directory; use `reload_settings()` after edits.
- Torch-dependent utilities require `torch` installed; everything else works without it.
