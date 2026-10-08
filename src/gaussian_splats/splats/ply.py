from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _values(vertex, names):
    return np.stack([vertex[name] for name in names], axis=1)


def save_ply(state: dict[str, torch.Tensor], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mean, rotation, scale, opacity, color = (
        state[k].detach().cpu().numpy() for k in ("mean", "rotation", "scale", "opacity", "color")
    )
    normals = state.get("normals", torch.zeros_like(state["mean"])).detach().cpu().numpy()

    N, D, C = color.shape
    if C != 3 or int(D**0.5) ** 2 != D:
        raise ValueError(f"Invalid color shape {color.shape}. Expected [N, (degree + 1) ** 2, 3]")

    fields = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range((D - 1) * 3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )

    values = np.concatenate(
        (
            mean,
            normals,
            color[:, 0],
            color[:, 1:].transpose(0, 2, 1).reshape(N, (D - 1) * 3),
            opacity.reshape(N, 1),
            scale,
            rotation,
        ),
        axis=1,
    )

    vertex = np.empty(N, dtype=[(field, "f4") for field in fields])
    for index, field in enumerate(fields):
        vertex[field] = values[:, index]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def load_ply(path):
    assert Path(path).suffix.lower() == ".ply", f"Only .ply files are supported. Got {Path(path)}"
    vertex = PlyData.read(str(path))["vertex"].data
    names = vertex.dtype.names or ()

    mean = _values(vertex, ["x", "y", "z"])
    normals = (
        _values(vertex, ["nx", "ny", "nz"])
        if all(name in names for name in ("nx", "ny", "nz"))
        else np.zeros_like(mean)
    )
    rotation = _values(vertex, [f"rot_{i}" for i in range(4)])
    scale = _values(vertex, [f"scale_{i}" for i in range(3)])
    opacity = vertex["opacity"][:, None]

    D = len([name for name in names if name.startswith("f_rest_")])
    dc_color = _values(vertex, [f"f_dc_{i}" for i in range(3)])
    sh_color = (
        _values(vertex, [f"f_rest_{i}" for i in range(D)])
        .reshape(len(vertex), 3, -1)
        .transpose(0, 2, 1)
        if D
        else np.empty((len(vertex), 0, 3), dtype=dc_color.dtype)
    )
    color = np.concatenate((dc_color[:, None], sh_color), axis=1)

    return {
        "mean": torch.as_tensor(mean),
        "normals": torch.as_tensor(normals),
        "rotation": torch.as_tensor(rotation),
        "scale": torch.as_tensor(scale),
        "opacity": torch.as_tensor(opacity),
        "color": torch.as_tensor(color),
    }
