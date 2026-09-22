"""Rasterizer test CLI."""

import os
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def test(
    interpret: Annotated[
        bool, typer.Option("--interpret", help="Run Triton kernels through the CPU interpreter.")
    ] = False,
    workers: Annotated[
        str | None,
        typer.Option(
            "--workers", help="Number of parallel pytest workers, or 'auto'. Serial by default."
        ),
    ] = None,
) -> None:
    """Run the rasterizer pytest suite."""
    if interpret:
        os.environ["TRITON_INTERPRET"] = "1"

    import pytest

    arguments = ["-v"]
    if workers is not None:
        arguments.extend(["-n", workers])
    raise typer.Exit(pytest.main(arguments))


@app.command()
def bench(
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
    from .benchmark import run

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


if __name__ == "__main__":
    app()
