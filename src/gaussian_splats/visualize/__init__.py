import os
from dataclasses import dataclass, field

import torch


@dataclass
class Frame:
    image: torch.Tensor
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


def default_device():
    return os.getenv("VISUALIZE_DEVICE") or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
