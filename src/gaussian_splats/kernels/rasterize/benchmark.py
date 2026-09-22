"""End-to-end rasterizer benchmark over a continuous camera trajectory."""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import torch
import typer
from PIL import Image
from tqdm import tqdm
from triton.testing import do_bench

from ...splats.camera import PinholeCamera
from ...splats.splats import GaussianSplat
from .interface import rasterize as triton_rasterize
from .torch import rasterize as torch_rasterize

app = typer.Typer(add_completion=False)


def random_scene(num_splats: int, seed: int) -> GaussianSplat:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        splats = GaussianSplat(num_splats, sh_degree=1)
        with torch.no_grad():
            splats.mean[:, :2].normal_(std=0.5)
            splats.mean[:, 2].uniform_(2.0, 6.0)
            splats.scale.fill_(-2.0)
    return splats.cuda()


def select_splats(splats: GaussianSplat, num_splats: int | None, seed: int) -> GaussianSplat:
    if num_splats is None or num_splats >= splats.num_points:
        return splats

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        indices = torch.randperm(splats.num_points)[:num_splats]

    selected = GaussianSplat(num_splats, sh_degree=splats.sh_degree).to(splats.mean.device)
    with torch.no_grad():
        selected.mean.copy_(splats.mean[indices])
        selected.rotation.copy_(splats.rotation[indices])
        selected.scale.copy_(splats.scale[indices])
        selected.opacity.copy_(splats.opacity[indices])
        selected.color.copy_(splats.color[indices])

    return selected


def generate_trajectory(
    splats: GaussianSplat,
    image_size: tuple[int, int],
    num_frames: int,
) -> list[PinholeCamera]:
    width, height = image_size
    elevation = math.radians(45)
    half_elevation = elevation / 2
    cos_elevation = math.cos(half_elevation)
    sin_elevation = math.sin(half_elevation)
    cameras = []
    for frame in range(num_frames):
        angle = 2 * math.pi * frame / num_frames
        half_angle = angle / 2
        cos_angle = math.cos(half_angle)
        sin_angle = math.sin(half_angle)

        # Compose a y-axis orbit with a fixed downward x-axis pitch.
        rotation = splats.mean.new_tensor(
            [
                cos_angle * cos_elevation,
                -cos_angle * sin_elevation,
                sin_angle * cos_elevation,
                -sin_angle * sin_elevation,
            ]
        )
        camera = PinholeCamera(
            image_size=image_size,
            focal_length=(0.75 * width, 0.75 * width),
            principal_point=(0.5 * width, 0.5 * height),
            rotation=rotation,
        )
        cameras.append(camera.fit_to_points(splats.mean.detach()))
    return cameras


def save_gif(frames: list[torch.Tensor], path: Path) -> None:
    images = [
        Image.fromarray(frame.detach().clamp(0, 1).mul(255).byte().cpu().numpy(), "RGB")
        for frame in frames
    ]
    images[0].save(path, save_all=True, append_images=images[1:], duration=40, loop=0)


def plot_bars(values: dict[str, dict[str, float]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = list(values)
    data = {
        "backend": labels * 2,
        "operation": ["forward"] * len(labels) + ["forward + backward"] * len(labels),
        "milliseconds": [values[label]["forward"] for label in labels]
        + [values[label]["backward"] for label in labels],
    }
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=data,
        x="backend",
        y="milliseconds",
        hue="operation",
        errorbar=None,
        ax=axis,
    )
    axis.set_yscale("log")
    axis.set_ylabel("Milliseconds per frame")
    axis.set_title("Representative-view rasterization time")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_lines(records: dict[str, float], path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = {"backend": list(records), "milliseconds": list(records.values())}
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=data, x="backend", y="milliseconds", marker="o", ax=axis)
    axis.set_xlabel("Backend")
    axis.set_ylabel("Total trajectory milliseconds")
    axis.set_title("End-to-end trajectory rendering time")
    axis.set_yscale("log")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(
    *,
    checkpoint: str | Path | None,
    output_dir: str | Path,
    device: str | None,
    image_size: tuple[int, int],
    num_splats: int | None,
    frames: int,
    repeats: int,
    seed: int,
) -> None:
    output_root = Path(output_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    tqdm.write(f"Benchmarking to {output_dir}")

    assert torch.cuda.is_available(), "CUDA is required for benchmarking"

    if checkpoint is None:
        splats = random_scene(num_splats or 10_000, seed)
    else:
        splats = GaussianSplat.from_checkpoint(checkpoint).cuda()
        splats = select_splats(splats, num_splats, seed)

    trajectory = generate_trajectory(splats, image_size, frames)
    funcs = {
        "torch_eager": torch_rasterize,
        "torch_compiled": torch.compile(torch_rasterize, dynamic=True),
        "triton": triton_rasterize,
    }

    first_camera = trajectory[0]
    bars: dict[str, dict[str, float]] = {}
    for backend, func in tqdm(funcs.items(), desc="First-frame benchmark"):
        with torch.no_grad():
            forward = do_bench(lambda f=func: f(splats, first_camera), warmup=2, rep=repeats)
        backward = do_bench(
            lambda f=func: f(splats, first_camera).square().mean().backward(),
            warmup=2,
            rep=repeats,
            grad_to_none=list(splats.parameters()),
        )
        bars[backend] = {"forward": forward, "backward": backward}

    records: dict[str, float] = {}
    rendered: dict[str, list[torch.Tensor]] = {}
    for backend, func in tqdm(funcs.items(), desc="Trajectory benchmark"):
        start_time = time.perf_counter()
        with torch.no_grad():
            rendered[backend] = [func(splats, camera) for camera in trajectory]
        torch.cuda.synchronize()
        records[backend] = (time.perf_counter() - start_time) * 1_000

    torch_frames = rendered["torch_eager"]
    triton_frames = rendered["triton"]
    difference_frames = [
        (torch_frame - triton_frame).abs() * 10
        for torch_frame, triton_frame in zip(torch_frames, triton_frames, strict=True)
    ]
    save_gif(torch_frames, output_dir / "torch.gif")
    save_gif(triton_frames, output_dir / "triton.gif")
    save_gif(difference_frames, output_dir / "absolute_difference.gif")
    plot_bars(bars, output_dir / "milliseconds_bars.png")
    plot_lines(records, output_dir / "milliseconds_over_time.png")
    measurements = {"first_frame": bars, "trajectory": records}
    (output_dir / "measurements.json").write_text(json.dumps(measurements, indent=2))
    tqdm.write(f"benchmark: complete; elapsed={time.perf_counter() - start_time:.1f}s")


@app.command()
def benchmark(
    checkpoint: Annotated[str | None, typer.Option(help="Scene checkpoint path.")] = None,
    output: Annotated[str, typer.Option(help="Output directory.")] = "output/rasterize-benchmark",
    device: Annotated[
        str | None, typer.Option(help="Torch device; defaults to CUDA when available.")
    ] = None,
    width: Annotated[int, typer.Option(min=8)] = 640,
    height: Annotated[int, typer.Option(min=8)] = 480,
    num_splats: Annotated[int | None, typer.Option(min=1)] = None,
    frames: Annotated[
        int,
        typer.Option(
            "--trajectory-frames", "--frames", min=2, help="Number of camera trajectory frames."
        ),
    ] = 120,
    repeats: Annotated[int, typer.Option(min=1)] = 3,
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    """Benchmark full rasterization over a continuous camera trajectory."""
    run(
        checkpoint=checkpoint,
        output_dir=output,
        device=device,
        image_size=(width, height),
        num_splats=num_splats,
        frames=frames,
        repeats=repeats,
        seed=seed,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    app()
