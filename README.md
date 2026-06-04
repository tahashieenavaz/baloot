# Baloot

Small, focused Python helpers for experiments and day-to-day developer workflows. The package is pure Python, has no required runtime dependencies, and keeps the API flat so you can import most helpers directly from `baloot`.

## Overview

- Lightweight utilities for seeding, file helpers, settings, Torch ergonomics, plotting, and small modules.
- Optional integrations: install the libraries you actually use (e.g., `torch`, `numpy`, `matplotlib`).
- A simple, readable API surface intended for quick reuse in scripts and research code.

## Installation

```bash
pip install baloot
```

## API Reference

### Constants

```python
import baloot

print(baloot.pi())
# tensor(3.1416)

print(baloot.e())
# tensor(2.7183)
```

## Copyright

MIT License. Copyright (c) 2025 Taha Shieenavaz. See `LICENSE`.
