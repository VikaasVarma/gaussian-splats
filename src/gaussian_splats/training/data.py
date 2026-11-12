from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..splats.camera import Camera, PinholeCamera


class ViewDataset(Dataset[tuple[Camera, Tensor]]):
    def __init__(self, views: list[tuple[Camera, Tensor]]):
        self.views = []
        for camera, image in views:
            image = image.detach().cpu()
            image = image.float().div_(255) if image.dtype == torch.uint8 else image.float()
            self.views.append((deepcopy(camera), image))

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, index: int) -> tuple[Camera, Tensor]:
        return self.views[index]


def _single_view(batch: list[tuple[Camera, Tensor]]) -> tuple[Camera, Tensor]:
    return batch[0]


def _make_dataloader(
    views: list[tuple[Camera, Tensor]], *, shuffle: bool, device: torch.device
) -> DataLoader[tuple[Camera, Tensor]]:
    return DataLoader(
        ViewDataset(views),
        batch_size=1,
        shuffle=shuffle,
        collate_fn=_single_view,
        pin_memory=device.type == "cuda",
    )


def load_views(path: Path) -> list[tuple[PinholeCamera, Tensor]]:
    """Load a torch-saved list of pinhole camera records and HWC RGB tensors."""
    records: list[dict[str, Any]] = torch.load(path, map_location="cpu", weights_only=True)
    return [
        (
            PinholeCamera(
                rotation=record["camera"]["rotation"],
                translation=record["camera"]["translation"],
                image_size=tuple(record["camera"]["image_size"]),
                focal_length=tuple(record["camera"]["focal_length"]),
                principal_point=tuple(record["camera"]["principal_point"]),
            ),
            record["image"],
        )
        for record in records
    ]


def make_dataloaders(
    path: Path, device: torch.device, held_out_views: int = 1
) -> tuple[DataLoader[tuple[Camera, Tensor]], DataLoader[tuple[Camera, Tensor]]]:
    views = load_views(path)
    assert held_out_views > 0 and held_out_views < len(views), (
        f"Held out views must be between 1 and {len(views)}. Got {held_out_views}."
    )

    train_views, validation_view = views[:-held_out_views], views[-held_out_views:]
    train_loader = _make_dataloader(train_views, shuffle=True, device=device)
    validation_loader = _make_dataloader(validation_view, shuffle=False, device=device)
    return train_loader, validation_loader
