from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(self.bn2(self.conv2(out)))
        return self.relu(out + identity)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class AttentionGate(nn.Module):
    """Attention Gate (Oktay et al., 2018) — learns to suppress irrelevant
    spatial regions in the encoder skip connection using the decoder gating
    signal."""

    def __init__(
            self,
            gate_channels: int,
            skip_channels: int,
            inter_channels: int | None = None,
    ):
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2

        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
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


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — captures multi-scale context through
    parallel dilated convolutions and a global average pooling branch."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dilations: list[int] | None = None,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 6, 12, 18]

        branches: list[nn.Module] = []
        for d in dilations:
            if d == 1:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ))
            else:
                branches.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels, 3,
                        padding=d, dilation=d, bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ))
        self.branches = nn.ModuleList(branches)

        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.gap_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.gap_bn = nn.BatchNorm2d(out_channels)
        self.gap_relu = nn.ReLU(inplace=True)

        n_features = len(dilations) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * n_features, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        gap = self.gap_conv(self.gap_pool(x))
        gap = F.interpolate(gap, size=x.shape[2:], mode="bilinear", align_corners=False)
        gap = self.gap_relu(self.gap_bn(gap))
        features.append(gap)
        return self.project(torch.cat(features, dim=1))
