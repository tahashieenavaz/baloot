# Baloot

Small, focused Python helpers for experiments and day-to-day developer workflows. The package is pure Python, has no required runtime dependencies, and keeps the API flat so you can import most helpers directly from `baloot`.

## Overview

- Lightweight utilities for seeding, file helpers, settings, Torch ergonomics, plotting, and small modules.
- Optional integrations: install the libraries you actually use (e.g., `torch`, `numpy`, `matplotlib`).
- A simple, readable API surface intended for quick reuse in scripts and research code.

## Installation

- From PyPI:
  ```bash
  pip install baloot
  ```
- Local editable install:
  ```bash
  pip install -e .
  ```

Requires Python 3.9+.

Optional dependencies (install as needed):
- `torch` for `baloot.torch`, `baloot.modules`, and `seed_torch` / `seed_everything`.
- `numpy` for `seed_numpy` and charting utilities.
- `matplotlib` for `baloot.chart` utilities.

## Usage

Quick start:

```python
from baloot import acceleration_device, funnel, seed_everything, settings

device = acceleration_device()  # torch: cuda -> mps -> cpu
seed_everything(42)             # python, numpy, torch aligned
funnel("model.pkl", model)      # save a pickled object
print(settings("training.batch_size", default=32))
```

### API reference (all implemented methods)

Top-level re-exports (`from baloot import ...`):
- `acceleration_device()` -> `torch.device`
- `parameter_count(model)` -> `int`
- `ksg_mi(x, y, k=3)` -> `torch.Tensor`
- `randomly_replace_layers(model, target_type, replacement_pool)` -> `torch.nn.Module`
- `render_template(template_location, target_location, **replacements)` -> `bool`
- `funnel(file_location, thing=None)` -> `bool | Any | None`
- `seed_python(seed)` -> `None`
- `seed_torch(seed)` -> `None`
- `seed_numpy(seed)` -> `None`
- `seed_everything(seed)` -> `None`
- `seed_gym(seed, environment, seed_observation=False)` -> `None`
- `seed_gymnasium(seed, environment)` -> `None`
- `settings(key, default=None)` -> `Any`
- `get_settings(*args, **kwargs)` -> `Any`
- `reload_settings()` -> `None`

Files & utilities (organized by module):

`baloot.files`
- `_save_thing(thing, file_location)` -> `bool`: Pickle and save an object.
- `_load_thing(file_location)` -> `Any | None`: Load a pickled object.
- `funnel(file_location, thing=None)` -> `bool | Any | None`: Save when `thing` is provided; load when `thing=None`.
- `render_template(template_location, target_location, **replacements)` -> `bool`: Replace `%key%` placeholders in a template.

`baloot.seed`
- `seed_torch(seed)` -> `None`: Seed CPU/CUDA and set deterministic cuDNN flags.
- `seed_numpy(seed)` -> `None`: Seed NumPy RNG.
- `seed_python(seed)` -> `None`: Seed Python `random`.
- `_try_seed_space(space, seed)` -> `None`: Internal helper for Gym-style spaces.
- `seed_gymnasium(seed, environment)` -> `None`: Seed Gymnasium env, action space, and observation space.
- `seed_gym(seed, environment, seed_observation=False)` -> `None`: Seed classic Gym action space (+ optional observation space).
- `seed_everything(seed)` -> `None`: Call Python, NumPy, and Torch seeders.

`baloot.settings`
- `settings(key, default=None)` -> `Any`: Dotted-path lookup in `settings.json` from the working directory.
- `reload_settings()` -> `None`: Clear the settings cache.
- `get_settings(*args, **kwargs)` -> `Any`: Alias for `settings`.

`baloot.torch`
- `parameter_count(model)` -> `int`: Count trainable parameters (`requires_grad=True`).
- `acceleration_device()` -> `torch.device`: Pick `cuda`, `mps`, then `cpu`.
- `randomly_replace_layers(model, target_type, replacement_pool)` -> `torch.nn.Module`: Recursively swap layers by type.
- `ksg_mi(x, y, k=3)` -> `torch.Tensor`: Kraskov-Stoegbauer-Grassberger MI estimate.

`baloot.modules`
- `Reshape.forward(x)` -> `torch.Tensor`: Reshape to `[B, -1, DIM]`.
- `PatchEmbedding.__init__(channels, features, patch_width, patch_height, activate=False, activation=torch.nn.functional.gelu)`
- `PatchEmbedding.forward(x)` -> `torch.Tensor`

`baloot.drl`
- `linear_epsilon(step, k, start_eps, end_eps)` -> `float`: Linear epsilon schedule.
- `cosine_epsilon(step, k, start_eps, end_eps)` -> `float`: Cosine epsilon schedule.

`baloot.chart`
- `size(width, height=None)` -> `None`: Set figure size (square when `height=None`).
- `dpi(resolution)` -> `None`: Set figure DPI.
- `spring(x, y=None, title=None, xtitle=None, ytitle=None, output_path=None, color="#2E86C1", alpha=0.1)` -> `(figure, axis)`

Notes:
- `settings.json` is read from the current working directory. Call `reload_settings()` after edits.
- Torch-dependent utilities require `torch` installed; chart utilities require `matplotlib` and `numpy`.

## Copyright

MIT License. Copyright (c) 2025 Taha Shieenavaz. See `LICENSE`.
