# gaussian-splats

A from-scratch Gaussian splatting implementation for learning and research.

This repository intentionally starts small. It provides a typed, tested Python package and a
reproducible development environment; the renderer and training pipeline will be built here
incrementally.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the environment and install dependencies:

```bash
uv sync
```

Run the checks:

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

## Layout

```text
src/gaussian_splats/  Python package
tests/                 Test suite
```

Python 3.12 is the supported development version. Runtime dependencies are kept to PyTorch,
NumPy, and Pillow.
