import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock, CBAMBlock, AttentionGate


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
        up_channels = in_channels // 2
        self.up_conv = nn.Conv2d(in_channels, up_channels, kernel_size=1)

        self.attention_gate = (
            AttentionGate(gate_channels=up_channels, skip_channels=skip_channels)
            if use_attention else None
        )

        self.residual = ResidualBlock(
            up_channels + skip_channels, out_channels, dropout_rate,
        )
        self.cbam = CBAMBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.up_conv(x)

        if self.attention_gate is not None:
            skip = self.attention_gate(gate=x, skip=skip)

        x = torch.cat([skip, x], dim=1)
        x = self.residual(x)
        x = self.cbam(x)
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
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip1, skip2, skip3, skip4 = skips

        d4 = self.dec4(bottleneck, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        return d1, [d4, d3, d2]
