"""Rasterizer test CLI."""

import os

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def test(
    interpret: bool = typer.Option(
        False, "--interpret", help="Run Triton kernels through the CPU interpreter."
    ),
    workers: str | None = typer.Option(
        None,
        "--workers",
        help="Number of parallel pytest workers, or 'auto'. Serial by default.",
    ),
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
    checkpoint: str | None = typer.Option(None, help="Scene checkpoint path."),
    output: str = typer.Option("output/rasterize-benchmark", help="Output directory."),
    device: str | None = typer.Option(None, help="Torch device; defaults to CUDA when available."),
    width: int = typer.Option(640, min=8),
    height: int = typer.Option(480, min=8),
    num_splats: int | None = typer.Option(None, min=1),
    frames: int = typer.Option(
        120, "--trajectory-frames", "--frames", min=2, help="Number of camera trajectory frames."
    ),
    repeats: int = typer.Option(3, min=1),
    seed: int = typer.Option(42),
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
