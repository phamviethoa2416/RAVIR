from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopoProjector(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=embedding_dim,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.projector(x)
        return F.normalize(e, dim=1, eps=1e-6)
