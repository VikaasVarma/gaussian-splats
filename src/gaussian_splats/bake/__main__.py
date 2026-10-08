from pathlib import Path
from typing import Annotated

import torch
import typer

from .blender import BlenderSession
from .query import query_scene
from .renderers import CyclesBackend, PhongBackend, WorkbenchBackend
from .renderers.types import RayBackend
from .sample import sample_splats
from .scene import Scene

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
    backend: RayBackend,
    device: str,
    sh_degree: int,
    view_samples: int,
    sh_smoothing_percent: float,
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
        sh_degree=sh_degree,
        view_samples=view_samples,
        sh_smoothing_percent=sh_smoothing_percent,
    )

    output = output or mesh.with_suffix(".pt")
    baked.save_checkpoint(output)
    typer.echo(f"Wrote {output} ({len(baked.mean):,} splats)")

    if invalid.mean.any():
        invalid_output = output.with_name(f"{output.stem}.invalid.pt")
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
    cycles_device: Annotated[str, typer.Option(help="Cycles backend: CPU or OPTIX.")] = "CPU",
    cycles_workers: Annotated[
        int, typer.Option(min=1, help="Concurrent Blender processes for Cycles queries.")
    ] = 1,
    eps: Annotated[float, typer.Option(min=0)] = 1e-4,
    ray_batch_size: Annotated[int, typer.Option(min=1, max=8192)] = 8192,
    sh_degree: Annotated[int, typer.Option(min=0, max=8)] = 4,
    view_samples: Annotated[int, typer.Option(min=1)] = 32,
    sh_smoothing_percent: Annotated[
        float, typer.Option(min=0, max=100, help="Blend SH toward each triangle mean, in percent.")
    ] = 0.0,
    exposure: Annotated[float, typer.Option()] = 4.0,
) -> None:
    with BlenderSession(mesh, blender) as session:
        backend = CyclesBackend(
            session, cycles_samples, ray_batch_size, eps, exposure, cycles_device, cycles_workers
        )
        _run_bake(
            mesh,
            output,
            resolution,
            n_splats,
            sigma,
            session.scene,
            backend,
            device,
            sh_degree,
            view_samples,
            sh_smoothing_percent,
        )


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
    ambient_strength: Annotated[float, typer.Option(min=0)] = 0.05,
    diffuse_strength: Annotated[float, typer.Option(min=0)] = 1.0,
    sh_degree: Annotated[int, typer.Option(min=0, max=3)] = 2,
    view_samples: Annotated[int, typer.Option(min=1)] = 32,
    sh_smoothing_percent: Annotated[
        float, typer.Option(min=0, max=100, help="Blend SH toward each triangle mean, in percent.")
    ] = 0.0,
) -> None:
    with BlenderSession(mesh, blender) as session:
        scene = session.scene

    backend = WorkbenchBackend(ambient_strength, diffuse_strength)
    _run_bake(
        mesh,
        output,
        resolution,
        n_splats,
        sigma,
        scene,
        backend,
        device,
        sh_degree,
        view_samples,
        sh_smoothing_percent,
    )


@bake_app.command("phong")
def bake_phong(
    mesh: Annotated[Path, typer.Option("--mesh", exists=True, dir_okay=False)],
    blender: Annotated[str, typer.Option(help="Blender executable.")] = "blender",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    resolution: Annotated[int | None, typer.Option("--resolution", min=2)] = None,
    n_splats: Annotated[int | None, typer.Option("--n-splats", min=1)] = None,
    sigma: Annotated[float, typer.Option(min=0)] = 0.65,
    device: Annotated[
        str, typer.Option("--device", help="PyTorch device for sampling and rendering.")
    ] = "cpu",
    ambient_strength: Annotated[float, typer.Option(min=0)] = 0.25,
    diffuse_strength: Annotated[float, typer.Option(min=0)] = 1.0,
    specular_strength: Annotated[float, typer.Option(min=0)] = 1.0,
    sh_degree: Annotated[int, typer.Option(min=0, max=3)] = 2,
    view_samples: Annotated[int, typer.Option(min=1)] = 32,
    sh_smoothing_percent: Annotated[
        float, typer.Option(min=0, max=100, help="Blend SH toward each triangle mean, in percent.")
    ] = 0.0,
) -> None:
    with BlenderSession(mesh, blender) as session:
        scene = session.scene

    backend = PhongBackend(
        ambient_strength,
        diffuse_strength,
        specular_strength,
    )
    _run_bake(
        mesh,
        output,
        resolution,
        n_splats,
        sigma,
        scene,
        backend,
        device,
        sh_degree,
        view_samples,
        sh_smoothing_percent,
    )


if __name__ == "__main__":
    app()
