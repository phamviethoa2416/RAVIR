from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock, CBAMBlock


class DecoderStage(nn.Module):
    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            dropout_rate: float = 0.0,
            use_attention: bool = True,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2, bias=False,
        )
        self.bn_up = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)

        self.residual = ResidualBlock(
            in_channels // 2 + skip_channels, out_channels, dropout_rate,
        )
        self.attention = CBAMBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_up(self.up(x)))

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        x = self.residual(x)
        x = self.attention(x)
        return x


class FeatureDecoder(nn.Module):
    def __init__(
            self,
            skip_channels: list[int],
            bottleneck_channels: int,
            dropout_rate: float = 0.0,
            use_attention: bool = True,
    ):
        super().__init__()
        assert len(skip_channels) == 4, f"Expected 4 skip channels, got {len(skip_channels)}"

        s1, s2, s3, s4 = skip_channels

        # Decoder stages (deepest → shallowest)
        self.dec4 = DecoderStage(bottleneck_channels, s4, s4, dropout_rate, use_attention)
        self.dec3 = DecoderStage(s4, s3, s3, dropout_rate, use_attention)
        self.dec2 = DecoderStage(s3, s2, s2, dropout_rate, use_attention)
        self.dec1 = DecoderStage(s2, s1, s1, dropout_rate, use_attention)

        self.output_channels = s1

    def forward(
            self, skips: list[torch.Tensor], bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        skip1, skip2, skip3, skip4 = skips

        x = self.dec4(bottleneck, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return x
