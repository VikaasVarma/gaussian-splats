from pathlib import Path
from typing import Annotated

import torch
import typer

from .backend import RayQueryBackend
from .cycles import CyclesBackend
from .query import query_scene
from .sample import sample_splats

app = typer.Typer(add_completion=False, no_args_is_help=True)
bake_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(bake_app, name="bake")


def _run_bake(
    mesh: Path,
    output: Path | None,
    resolution: int | None,
    n_splats: int | None,
    sigma: float,
    backend: RayQueryBackend,
    device: str,
    eps: float,
) -> None:
    with backend:
        scene = backend.scene.to(torch.device(device))
        splats = sample_splats(scene, resolution=resolution, n_splats=n_splats, sigma=sigma)
        baked, invalid = query_scene(splats, backend, eps=eps)

    output = output or mesh.with_suffix(".ply")
    baked.save_checkpoint(output)
    typer.echo(f"Wrote {output} ({len(baked.mean):,} splats)")

    if invalid.mean.any():
        invalid_output = output.with_name(f"{output.stem}.invalid{output.suffix}")
        invalid.save_checkpoint(invalid_output)
        typer.echo(f"Wrote invalid samples to {invalid_output}")


@bake_app.command("cycles")
def bake_cycles(
    mesh: Annotated[Path, typer.Option("--mesh", exists=True, dir_okay=False)],
    blender: Annotated[str, typer.Option(help="Blender executable.")] = "blender",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    resolution: Annotated[int | None, typer.Option("--resolution", min=2)] = None,
    n_splats: Annotated[int | None, typer.Option("--n-splats", min=1)] = None,
    sigma: Annotated[float, typer.Option(min=0)] = 0.65,
    device: Annotated[
        str, typer.Option("--device", help="PyTorch device for surface sampling.")
    ] = "cpu",
    cycles_samples: Annotated[int, typer.Option("--samples", min=1)] = 1,
    eps: Annotated[float, typer.Option(min=0)] = 1e-4,
    ray_batch_size: Annotated[int, typer.Option(min=1, max=4096)] = 4096,
) -> None:
    backend = CyclesBackend(mesh, blender, cycles_samples, ray_batch_size, eps)
    _run_bake(mesh, output, resolution, n_splats, sigma, backend, device, eps)


if __name__ == "__main__":
    app()
