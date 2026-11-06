"""
3DGS MCMC training: https://arxiv.org/abs/2404.09591
"""

import math
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError
from tqdm.auto import tqdm

import wandb

from ..splats.rasterize import rasterize
from ..splats.splats import GaussianSplat
from .data import make_dataloaders
from .utils import initialize_splats, noise_splats, relocate_splats, ssim

app = typer.Typer(add_completion=False, no_args_is_help=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKEND: Literal["torch", "triton", "gsplat"] = "gsplat" if torch.cuda.is_available() else "torch"
TILE_SIZE = 16


@torch.no_grad()
def validate(splats: GaussianSplat, loader: DataLoader) -> tuple[dict[str, float], list[Tensor]]:
    metrics = MetricCollection(
        {
            "l1": MeanAbsoluteError(),
            "psnr": PeakSignalNoiseRatio(data_range=1.0),
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
        }
    ).to(DEVICE)
    comparisons = []
    for camera, target in tqdm(loader, desc="Validate", unit="view", leave=False):
        camera = camera.to(DEVICE)
        target = target.to(DEVICE)

        image = rasterize(splats, camera, tile_size=TILE_SIZE, backend=BACKEND).clamp(0, 1)
        metrics.update(
            image.permute(2, 0, 1).contiguous()[None],
            target.permute(2, 0, 1).contiguous()[None],
        )
        rendered = (image * 255).round().byte().cpu()
        reference = (target.clamp(0, 1) * 255).round().byte().cpu()
        comparisons.append(torch.cat((rendered, reference), dim=1))

    return {name: value.item() for name, value in metrics.compute().items()}, comparisons


def train(
    data_path: Path,
    # Training schedule
    seed: int = 1337,
    n_steps: int = 30_000,
    val_every_n_steps: int = 1_000,
    held_out_views: int = 1,
    warmup: int = 500,
    relocation_interval: int = 100,
    # Splats
    initial_splats: int = 100_000,
    initial_sh_degree: int = 3,
    max_splats: int = 1_000_000,
    # Loss
    lambda_dssim: float = 0.2,
    # Learning rates
    position_lr: float = 1.6e-4,
    position_lr_final: float = 1.6e-6,
    feature_lr: float = 2.5e-3,
    opacity_lr: float = 5e-2,
    scale_lr: float = 5e-3,
    rotation_lr: float = 1e-3,
    noise_lr: float = 5e5,
    log: bool = False,
    log_project: str = "gaussian-splat",
    log_group: str | None = None,
    # Regularization
    opacity_reg: float = 1e-2,
    scale_reg: float = 1e-2,
) -> GaussianSplat:
    """Randomly initialize and train on the expected torch-saved view dataset."""
    if log:
        run = wandb.init(project=log_project, group=log_group, save_code=True)
        run.log_code(include_fn=lambda path: Path(path).suffix == ".py")

    torch.manual_seed(seed)
    train_loader, validation_loader = make_dataloaders(data_path, DEVICE, held_out_views)
    train_views = train_loader.dataset.views
    splats = initialize_splats(train_views, initial_splats, initial_sh_degree).to(DEVICE)

    camera_positions = torch.stack([camera.position for camera, _ in train_views])
    scene_center = camera_positions.mean(0)
    spatial_lr_scale = 1.1 * torch.linalg.vector_norm(camera_positions - scene_center, dim=-1).max()
    initial_position_lr = position_lr * spatial_lr_scale.item()
    final_position_lr = position_lr_final * spatial_lr_scale.item()
    groups = [
        {"params": [splats.mean], "name": "mean", "lr": initial_position_lr},
        {"params": [splats.rotation], "name": "rotation", "lr": rotation_lr},
        {"params": [splats.scale], "name": "scale", "lr": scale_lr},
        {"params": [splats.opacity], "name": "opacity", "lr": opacity_lr},
        {"params": [splats.color], "name": "color", "lr": feature_lr},
    ]
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    step = 0

    epochs = math.ceil(n_steps / len(train_loader))
    with tqdm(total=n_steps, desc="Train", unit="step") as progress:
        for _ in range(epochs):
            for camera, target in train_loader:
                camera = camera.to(DEVICE)
                target = target.to(DEVICE)

                # Update position learning rate
                fraction = min((step + 1) / n_steps, 1)
                position_lr = (
                    initial_position_lr * (final_position_lr / initial_position_lr) ** fraction
                )
                optimizer.param_groups[0]["lr"] = position_lr

                # Compute loss
                image = rasterize(splats, camera, tile_size=TILE_SIZE, backend=BACKEND).clamp(0, 1)
                l1_loss = (image - target).abs().mean()
                dssim_loss = 1 - ssim(image, target)
                opacity_loss = opacity_reg * splats.opacity.sigmoid().mean()
                scale_loss = scale_reg * splats.scale.exp().mean()
                loss = (
                    (1 - lambda_dssim) * l1_loss
                    + lambda_dssim * dssim_loss
                    + opacity_loss
                    + scale_loss
                )

                # Update
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Noise positions
                noise_splats(splats, noise_lr, position_lr)
                if step >= warmup and step % relocation_interval == 0:
                    relocate_splats(splats, optimizer, max_splats)

                step += 1
                progress.update()
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l1=f"{l1_loss.item():.4f}",
                    splats=f"{splats.num_points:,}",
                )

                if log:
                    logs = {
                        f"train/{key}": value
                        for key, value in {
                            "loss": loss.item(),
                            "l1": l1_loss.item(),
                            "dssim": dssim_loss.item(),
                            "lr": position_lr,
                            "opacity_reg": opacity_loss.item(),
                            "scale_reg": scale_loss.item(),
                            "num_splats": splats.num_points,
                        }.items()
                    }

                # Validate
                if step % val_every_n_steps == 0 or step == n_steps:
                    validation_metrics, comparisons = validate(splats, validation_loader)

                    if log:
                        val_logs = {
                            **{f"val/{key}": value for key, value in validation_metrics.items()},
                            "val/comparisons": [
                                wandb.Image(image.permute(2, 0, 1)) for image in comparisons
                            ],
                        }
                        logs.update(val_logs)

                if log:
                    run.log({**logs, "step": step})

                if step == n_steps:
                    break
    if log:
        run.finish()

    return splats


@app.command()
def cli(
    data_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help=(
                "Torch-saved camera/image records. Final held-out views are used for validation."
            ),
        ),
    ],
    #
    seed: int = 1337,
    n_steps: Annotated[int, typer.Option(min=1)] = 30_000,
    val_every_n_steps: Annotated[int, typer.Option(min=1)] = 1_000,
    held_out_views: Annotated[int, typer.Option(min=1)] = 1,
    warmup: Annotated[int, typer.Option(min=0)] = 500,
    relocation_interval: Annotated[int, typer.Option(min=1)] = 100,
    #
    initial_splats: Annotated[int, typer.Option(min=1)] = 100_000,
    initial_sh_degree: Annotated[int, typer.Option(min=0, max=12)] = 3,
    max_splats: Annotated[int, typer.Option(min=1)] = 1_000_000,
    #
    lambda_dssim: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
    #
    position_lr: Annotated[float, typer.Option(min=0)] = 1.6e-4,
    position_lr_final: Annotated[float, typer.Option(min=0)] = 1.6e-6,
    feature_lr: Annotated[float, typer.Option(min=0)] = 2.5e-3,
    opacity_lr: Annotated[float, typer.Option(min=0)] = 5e-2,
    scale_lr: Annotated[float, typer.Option(min=0)] = 5e-3,
    rotation_lr: Annotated[float, typer.Option(min=0)] = 1e-3,
    noise_lr: Annotated[float, typer.Option(min=0)] = 5e5,
    log: Annotated[bool, typer.Option("--log")] = False,
    log_project: Annotated[str, typer.Option("--log-project")] = "gaussian-splat",
    log_group: Annotated[str | None, typer.Option("--log-group")] = None,
    #
    opacity_reg: Annotated[float, typer.Option(min=0)] = 1e-2,
    scale_reg: Annotated[float, typer.Option(min=0)] = 1e-2,
) -> None:
    """Train from random initialization and save beside the input file."""
    result = train(
        data_path,
        # Training schedule
        seed=seed,
        n_steps=n_steps,
        val_every_n_steps=val_every_n_steps,
        held_out_views=held_out_views,
        warmup=warmup,
        relocation_interval=relocation_interval,
        # Splats
        initial_splats=initial_splats,
        initial_sh_degree=initial_sh_degree,
        max_splats=max_splats,
        # Loss
        lambda_dssim=lambda_dssim,
        # Learning rates
        position_lr=position_lr,
        position_lr_final=position_lr_final,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scale_lr=scale_lr,
        rotation_lr=rotation_lr,
        noise_lr=noise_lr,
        log=log,
        log_project=log_project,
        log_group=log_group,
        # Regularization
        opacity_reg=opacity_reg,
        scale_reg=scale_reg,
    )
    output = data_path.with_name(f"{data_path.stem}.pt")
    result.save_checkpoint(output)
    typer.echo(f"Wrote {output} ({result.num_points:,} splats)")


if __name__ == "__main__":
    app()
