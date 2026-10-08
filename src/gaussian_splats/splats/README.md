# Gaussian splats

Represent Gaussian splats, project them through a camera, and rasterize RGB images.

## Usage

Run `uv sync` from `gaussian-splats/` to install the project, then load and render:

```python
import torch

from gaussian_splats.splats.camera import PinholeCamera
from gaussian_splats.splats.rasterize import rasterize
from gaussian_splats.splats.splats import GaussianSplat

splats = GaussianSplat.from_checkpoint("assets/cornell.pt")
camera = PinholeCamera().fit_to_points(splats.mean)
with torch.no_grad():
    image = rasterize(splats, camera)  # H × W × 3 float RGB
```

For interactive inspection, use the [viewer](../visualize/README.md).

Checkpoints store the model's state dictionary. Save with `splats.save_checkpoint(path)`, or exchange standard 3DGS PLY files with:

```bash
uv run convert pt-to-ply assets/cornell.pt
uv run convert ply-to-pt scene.ply
```

## Representation and conventions

For `N` splats and SH degree `d`, `K = (d + 1)²`:

| Tensor | Shape | Meaning |
|---|---|---|
| `mean` | N × 3 | Gaussian means |
| `rotation` | N × 4 | Gaussian rotation quaternions, `(w, x, y, z)` |
| `scale` | N × 3 | Gaussian Log standard deviations |
| `opacity` | N × 1 | Opacity logits; `alpha = opacity.sigmoid()` |
| `color` | N × K × 3 | Real SH coefficients per RGB channel; evaluation adds 0.5 |
| `normals` | N × 3 | World-space surface normals, when available |

Cameras use world-to-camera rotation and translation: `p_camera = R @ p_world + t`, looking along −Z with +Y up (blender convention). Focal lengths and principal point are in pixels, with image coordinates increasing right and down.

## Rasterizer benchmark

| Splats | Torch F/F+B (ms) | Triton F/F+B (ms) | gsplat F/F+B (ms) |
| ---: | ---: | ---: | ---: |
| 1,000 | 8.19 / 10.95 | 0.79 / 1.27 | 1.15 / 1.77 |
| 10,000 | 19.43 / 66.15 | 1.67 / 3.89 | 1.96 / 3.18 |
| 100,000 | 182.36 / 430.44 | 2.66 / 4.51 | 2.16 / 3.54 |
| 250,000 | 436.83 / 901.26 | 6.15 / 6.06 | 2.37 / 7.81 |

*All measurements are averages across [various baked scenes](../bake/README.md) at 512×512. Measured on an NVIDIA RTX 3090.*
