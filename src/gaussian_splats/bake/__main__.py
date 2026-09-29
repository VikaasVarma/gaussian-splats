from pathlib import Path
from typing import Annotated

import torch
import typer

from gaussian_splats.blender import BlenderSession, Scene

from .backend import RayQueryBackend
from .cycles import CyclesBackend
from .query import query_scene
from .sample import sample_splats
from .workbench import WorkbenchBackend

app = typer.Typer(add_completion=False, no_args_is_help=True)
bake_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(bake_app, name="bake")


def _run_bake(
    mesh: Path,
    output: Path | None,
    resolution: int | None,
    n_splats: int | None,
    sigma: float,
    scene: Scene,
    backend: RayQueryBackend,
    device: str,
    eps: float,
) -> None:
    scene = scene.to(torch.device(device))
    splats, triangle_id, barycentric = sample_splats(
        scene, resolution=resolution, n_splats=n_splats, sigma=sigma
    )
    baked, invalid = query_scene(
        scene,
        splats,
        backend,
        triangle_id,
        barycentric,
        eps=eps,
    )

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
    with BlenderSession(mesh, blender) as session:
        backend = CyclesBackend(session, cycles_samples, ray_batch_size, eps)
        _run_bake(mesh, output, resolution, n_splats, sigma, session.scene, backend, device, eps)


@bake_app.command("workbench")
def bake_workbench(
    mesh: Annotated[Path, typer.Option("--mesh", exists=True, dir_okay=False)],
    blender: Annotated[str, typer.Option(help="Blender executable.")] = "blender",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    resolution: Annotated[int | None, typer.Option("--resolution", min=2)] = None,
    n_splats: Annotated[int | None, typer.Option("--n-splats", min=1)] = None,
    sigma: Annotated[float, typer.Option(min=0)] = 0.65,
    device: Annotated[
        str, typer.Option("--device", help="PyTorch device for sampling and rendering.")
    ] = "cpu",
    eps: Annotated[float, typer.Option(min=0)] = 1e-4,
) -> None:
    with BlenderSession(mesh, blender) as session:
        scene = session.scene

    backend = WorkbenchBackend(eps)
    _run_bake(mesh, output, resolution, n_splats, sigma, scene, backend, device, eps)


if __name__ == "__main__":
    app()
