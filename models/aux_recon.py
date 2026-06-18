from __future__ import annotations

import torch
import torch.nn as nn


class AuxReconHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
