# gaussian-splats

A from-scratch Gaussian splatting implementation for learning and research.

This repository intentionally starts small. The renderer and training pipeline will be built
here incrementally.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the environment and install dependencies:

```bash
uv sync
```

Run the linter:

```bash
uv run ruff check .
```

## Layout

```text
src/gaussian_splats/  Python package
```

Python 3.12 or newer is required. Runtime dependencies are kept to PyTorch, NumPy, and Pillow.
