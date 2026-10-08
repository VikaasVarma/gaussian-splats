from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch
from tqdm.auto import tqdm

from gaussian_splats.splats.camera import PinholeCamera

from ..blender import BlenderSession
from ..scene import Scene
from .types import RayBackend, RenderBackend


class CyclesBackend(RayBackend, RenderBackend):
    def __init__(
        self,
        session: BlenderSession,
        samples: int = 1,
        batch_size: int = 1048576,
        eps: float = 1e-4,
        exposure: float = 0.0,
        device: str = "CPU",
        workers: int = 1,
    ) -> None:
        self.session = session
        self.samples = samples
        self.batch_size = batch_size
        self.eps = eps
        self.exposure = exposure
        self.device = device
        self.workers = workers

    def query(
        self,
        scene: Scene,
        origins: torch.Tensor,
        directions: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device, dtype = origins.device, origins.dtype
        origins_np = origins.detach().cpu().numpy()
        directions_np = directions.detach().cpu().numpy()

        scale = float(np.linalg.norm(np.ptp(origins_np, axis=0)))
        offset = max(self.eps, scale * 1e-7)
        ray_origins = origins_np + directions_np * offset

        # Split at batch boundaries; each dispatch thread owns a separate Blender process.
        batches = math.ceil(len(origins_np) / self.batch_size)
        chunk_size = math.ceil(batches / self.workers) * self.batch_size

        def query_chunk(start):
            stop = start + chunk_size
            context = (
                nullcontext(self.session)
                if start == 0
                else BlenderSession(self.session.mesh, self.session.executable)
            )
            with context as session:
                output = session.request(
                    {
                        "command": "query",
                        "samples": self.samples,
                        "batch_size": self.batch_size,
                        "device": self.device,
                    },
                    {
                        "origins": ray_origins[start:stop],
                        "directions": directions_np[start:stop],
                        "footprints": np.zeros(len(origins_np[start:stop]), dtype=np.float32),
                    },
                    progress=progress,
                )
                with np.load(output) as result:
                    return tuple(result[key] for key in ("colors", "alphas", "hits"))

        with (
            tqdm(total=batches, desc="Cycles queries", unit="batch") as progress,
            ThreadPoolExecutor(max_workers=self.workers) as pool,
        ):
            results = list(pool.map(query_chunk, range(0, len(origins_np), chunk_size)))

        colors, alphas, hits = map(
            lambda chunks: torch.as_tensor(np.concatenate(chunks), device=device, dtype=dtype),
            zip(*results, strict=True),
        )
        colors = colors * 2**self.exposure

        valid = torch.isfinite(hits).all(1) & (
            torch.linalg.vector_norm(hits - origins, dim=1) <= self.eps * 2
        )
        return torch.cat((colors, alphas[:, None]), dim=1), valid

    def render(
        self,
        scene: Scene,
        camera: PinholeCamera,
        width: int,
        height: int,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.session.render(
            camera,
            width,
            height,
            renderer="cycles",
            samples=self.samples,
            exposure=float(kwargs.get("exposure", 0.0)),
        )
