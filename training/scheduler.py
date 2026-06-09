from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    LambdaLR,
    SequentialLR,
)


def get_scheduler(
    optimizer: Optimizer,
    *,
    warmup_epochs: int,
    total_epochs: int,
    cosine_eta_min: float = 1e-6,
) -> LRScheduler:
    cosine_epochs = max(1, total_epochs - warmup_epochs)

    if warmup_epochs <= 0:
        return CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=cosine_eta_min,
        )

    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / float(warmup_epochs)),
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=cosine_eta_min)

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
