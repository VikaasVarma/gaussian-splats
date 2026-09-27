from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gaussian_splats.blender import CyclesSession, Scene


class CyclesBackend:
    def __init__(
        self,
        mesh: str | Path,
        executable: str = "blender",
        samples: int = 1,
        batch_size: int = 4096,
        eps: float = 1e-4,
    ) -> None:
        assert 1 <= batch_size <= 4096, f"batch_size must be between 1 and 4096. Got {batch_size}"

        self.mesh = Path(mesh)
        self.executable = executable
        self.samples = samples
        self.batch_size = batch_size
        self.eps = eps
        self._session: CyclesSession | None = None
        self._active = False
        self.scene: Scene | None = None

    def __enter__(self) -> CyclesBackend:
        if not self.mesh.is_file():
            raise FileNotFoundError(self.mesh)

        self._session = CyclesSession(
            self.mesh, self.executable, self.samples, self.batch_size
        ).__enter__()
        self.scene = self._session.scene

        self._active = True
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if self._session is not None:
            self._session.__exit__(exception_type, exception, traceback)

        self._session = None
        self.scene = None
        self._active = False

    def query(
        self, points: torch.Tensor, normals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._active:
            raise RuntimeError("CyclesBackend must be used as a context manager")
        if self._session is None:
            raise RuntimeError("CyclesBackend session is not running")

        device, dtype = points.device, points.dtype
        points_np = points.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()

        scale = float(np.linalg.norm(np.ptp(points_np, axis=0)))
        offset = max(self.eps, scale * 1e-7)
        origins = points_np + normals_np * offset

        color, alpha, hit = self._session.query(
            origins,
            normals_np,
            np.zeros(len(points_np), dtype=np.float32),
        )

        return (
            torch.as_tensor(color, device=device, dtype=dtype),
            torch.as_tensor(alpha, device=device, dtype=dtype),
            torch.as_tensor(hit, device=device, dtype=dtype),
        )
