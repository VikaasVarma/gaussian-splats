"""Temporary rasterizer contract for the viewer."""

import numpy as np

_rng = np.random.default_rng()


def rasterize(width: int, height: int) -> np.ndarray:
    """Return an RGB uint8 image shaped ``(height, width, 3)``."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return _rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
