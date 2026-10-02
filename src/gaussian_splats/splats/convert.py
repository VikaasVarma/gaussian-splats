from pathlib import Path
from typing import Annotated

import typer

from .ply import load_ply, save_ply
from .splats import GaussianSplat

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("pt-to-ply")
def pt_to_ply(
    checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    output = output or checkpoint.with_suffix(".ply")
    save_ply(GaussianSplat.from_checkpoint(checkpoint).state_dict(), output)
    typer.echo(f"Wrote {output}")


@app.command("ply-to-pt")
def ply_to_pt(
    checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    output = output or checkpoint.with_suffix(".pt")
    GaussianSplat.from_tensors(**load_ply(checkpoint)).save_checkpoint(output)
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    app()
