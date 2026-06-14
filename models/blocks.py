from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden,
                out_channels=channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x) + x * self.sSE(x)


class ResidualModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_scse: bool = False,
        scse_reduction: int = 16,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.scse: nn.Module = (
            SCSEModule(out_channels, reduction=scse_reduction)
            if use_scse
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)

        out += identity
        out = self.relu(out)
        out = self.scse(out)
        return out


class AttentionModule(nn.Module):
    def __init__(
        self, gate_channels: int, skip_channels: int, inter_channels: int | None = None
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = max(skip_channels // 4, 8)
        if inter_channels >= 8 and inter_channels % 8 != 0:
            inter_channels = ((inter_channels + 7) // 8) * 8

        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, inter_channels), inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, inter_channels), inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        x = self.W_skip(skip)

        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        attention = self.psi(self.relu(g + x))
        return skip * attention
