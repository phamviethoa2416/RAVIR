from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LRScheduler,
    LambdaLR,
    SequentialLR,
)


def get_scheduler(
    optimizer: Optimizer,
    *,
    warmup_epochs: int,
    cosine_t0: int,
    cosine_t_mult: int,
    cosine_eta_min: float = 1e-6,
) -> LRScheduler:

    if warmup_epochs <= 0:
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_t0,
            T_mult=cosine_t_mult,
            eta_min=cosine_eta_min,
        )

    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: float(min(1.0, (epoch + 1) / float(warmup_epochs))),
    )
    cosine = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cosine_t0,
        T_mult=cosine_t_mult,
        eta_min=cosine_eta_min,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
