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

print(baloot.pi)
# tensor(3.1416)

print(baloot.e)
# tensor(2.7183)

print(baloot.sqrt2)
# tensor(1.4142)

print(baloot.sqrt3)
# tensor(1.7321)

print(baloot.sqrt5)
# tensor(2.2361)

print(baloot.ln2)
# tensor(0.6931)

print(baloot.ln10)
# tensor(2.3026)

print(baloot.c)
# tensor(299792458)
assert baloot.c == baloot.light

print(baloot.golden)
# tensor(1.6180)
```

### Reproducibility

```python
import baloot

assert baloot.seed == baloot.seed_everything

baloot.seed(123456)
```

Alternatively, you can seed different libraries separately.

```python
import baloot

baloot.seed_python(1)
baloot.seed_numpy(2)
baloot.seed_torch(3)
baloot.seed_cuda(4)
```


## Copyright

MIT License. Copyright (c) 2025 Taha Shieenavaz. See `LICENSE`.
