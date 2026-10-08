from pathlib import Path
from typing import Annotated

import torch
import typer

from gaussian_splats.splats.splats import GaussianSplat

from .blender import BlenderSession
from .query import query_scene
from .renderers import CyclesBackend, PhongBackend, WorkbenchBackend
from .renderers.types import RayBackend
from .sample import sample_splats
from .scene import Scene

app = typer.Typer(add_completion=False, no_args_is_help=True)
bake_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(bake_app, name="bake")


@torch.no_grad()
def _run_bake(
    resolution: int | None,
    n_splats: int | None,
    sigma: float,
    scene: Scene,
    backend: RayBackend,
    device: str,
    sh_degree: int,
    view_samples: int,
    sh_smoothing_percent: float,
) -> tuple[GaussianSplat, GaussianSplat]:
    scene = scene.to(torch.device(device))
    splats, triangle_id, barycentric = sample_splats(
        scene, resolution=resolution, n_splats=n_splats, sigma=sigma
    )
    return query_scene(
        scene,
        splats,
        backend,
        triangle_id,
        barycentric,
        sh_degree=sh_degree,
        view_samples=view_samples,
        sh_smoothing_percent=sh_smoothing_percent,
    )


def _save_bake(baked: GaussianSplat, invalid: GaussianSplat, output: Path) -> None:
    baked.save_checkpoint(output)
    typer.echo(f"Wrote {output} ({len(baked.mean):,} splats)")

    if invalid.num_points:
        invalid_output = output.with_name(f"{output.stem}.invalid.pt")
        invalid.save_checkpoint(invalid_output)
        typer.echo(f"Wrote invalid samples to {invalid_output}")


@bake_app.command("cycles")
def bake_cycles(
    mesh: Annotated[Path, typer.Option("--mesh", exists=True, dir_okay=False)],
    blender: Annotated[str, typer.Option(help="Blender executable.")] = "blender",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    resolution: Annotated[int | None, typer.Option("--resolution", min=2)] = None,
    n_splats: Annotated[
        int | None,
        typer.Option(
            "--n-splats", min=1, help="Target count; grid coverage determines the final count."
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 0,
    sigma: Annotated[float, typer.Option(min=0)] = 0.6825,
    device: Annotated[
        str, typer.Option("--device", help="PyTorch device for sampling and SH fitting.")
    ] = "cpu",
    cycles_samples: Annotated[int, typer.Option("--samples", min=1)] = 32,
    cycles_device: Annotated[str, typer.Option(help="Cycles backend: CPU or OPTIX.")] = "CPU",
    cycles_workers: Annotated[
        int, typer.Option(min=1, help="Concurrent Blender processes for Cycles queries.")
    ] = 1,
    eps: Annotated[float, typer.Option(min=0)] = 1e-4,
    ray_batch_size: Annotated[int, typer.Option(min=1)] = 1048576,
    sh_degree: Annotated[int, typer.Option(min=0, max=12)] = 6,
    view_samples: Annotated[int, typer.Option(min=1)] = 128,
    sh_smoothing_percent: Annotated[
        float, typer.Option(min=0, max=100, help="Blend SH toward each triangle mean, in percent.")
    ] = 0.0,
    exposure: Annotated[float, typer.Option()] = 0.0,
) -> None:
    torch.manual_seed(seed)
    with BlenderSession(mesh, blender) as session:
        backend = CyclesBackend(
            session, cycles_samples, ray_batch_size, eps, exposure, cycles_device, cycles_workers
        )
        baked, invalid = _run_bake(
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

    _save_bake(baked, invalid, output or mesh.with_suffix(".pt"))


@bake_app.command("workbench")
def bake_workbench(
    mesh: Annotated[Path, typer.Option("--mesh", exists=True, dir_okay=False)],
    blender: Annotated[str, typer.Option(help="Blender executable.")] = "blender",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    resolution: Annotated[int | None, typer.Option("--resolution", min=2)] = None,
    n_splats: Annotated[
        int | None,
        typer.Option(
            "--n-splats", min=1, help="Target count; grid coverage determines the final count."
        ),
    ] = None,
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
    baked, invalid = _run_bake(
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

    _save_bake(baked, invalid, output or mesh.with_suffix(".pt"))


@bake_app.command("phong")
def bake_phong(
    mesh: Annotated[Path, typer.Option("--mesh", exists=True, dir_okay=False)],
    blender: Annotated[str, typer.Option(help="Blender executable.")] = "blender",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    resolution: Annotated[int | None, typer.Option("--resolution", min=2)] = None,
    n_splats: Annotated[
        int | None,
        typer.Option(
            "--n-splats", min=1, help="Target count; grid coverage determines the final count."
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 0,
    sigma: Annotated[float, typer.Option(min=0)] = 0.65,
    device: Annotated[
        str, typer.Option("--device", help="PyTorch device for sampling and rendering.")
    ] = "cpu",
    ambient_strength: Annotated[float, typer.Option(min=0)] = 0.25,
    diffuse_strength: Annotated[float, typer.Option(min=0)] = 1.0,
    specular_strength: Annotated[float, typer.Option(min=0)] = 1.0,
    sh_degree: Annotated[int, typer.Option(min=0, max=12)] = 2,
    view_samples: Annotated[int, typer.Option(min=1)] = 32,
    sh_smoothing_percent: Annotated[
        float, typer.Option(min=0, max=100, help="Blend SH toward each triangle mean, in percent.")
    ] = 0.0,
) -> None:
    torch.manual_seed(seed)
    with BlenderSession(mesh, blender) as session:
        scene = session.scene

    backend = PhongBackend(
        ambient_strength,
        diffuse_strength,
        specular_strength,
    )
    baked, invalid = _run_bake(
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

    _save_bake(baked, invalid, output or mesh.with_suffix(".pt"))


if __name__ == "__main__":
    app()
