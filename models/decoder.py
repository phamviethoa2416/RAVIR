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
        self.up_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.residual = ResidualBlock(
            in_channels // 2 + skip_channels, out_channels, dropout_rate,
        )
        self.attention = CBAMBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.up_conv(x)
        x = torch.cat([skip, x], dim=1)
        x = self.residual(x)
        x = self.attention(x)
        return x


class FeatureDecoder(nn.Module):
    def __init__(
            self,
            channels: list[int] | None = None,
            dropout_rate: float = 0.0,
            use_attention: bool = True,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512, 1024]
        assert len(channels) == 5, "channels list must have exactly 5 entries"
        c1, c2, c3, c4, c5 = channels

        self.dec4 = DecoderStage(c5, c4, c4, dropout_rate, use_attention)
        self.dec3 = DecoderStage(c4, c3, c3, dropout_rate, use_attention)
        self.dec2 = DecoderStage(c3, c2, c2, dropout_rate, use_attention)
        self.dec1 = DecoderStage(c2, c1, c1, dropout_rate, use_attention)

    def forward(
            self, skips: list[torch.Tensor], bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        skip1, skip2, skip3, skip4 = skips

        x = self.dec4(bottleneck, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return x