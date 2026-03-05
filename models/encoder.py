import torch
import torch.nn as nn

from .blocks import ResidualBlock, CBAMBlock


class EncoderStage(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout_rate: float = 0.0,
            use_attention: bool = True,
    ):
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels, dropout_rate)
        self.attention = CBAMBlock(out_channels) if use_attention else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.residual(x)
        features = self.attention(features)
        downsampled = self.pool(features)
        return features, downsampled


class FeatureEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            channels: list[int] | None = None,
            dropout_rate: float = 0.0,
            use_attention: bool = True,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512, 1024]
        assert len(channels) == 5, "channels list must have exactly 5 entries"
        c1, c2, c3, c4, c5 = channels

        self.stage1 = EncoderStage(in_channels, c1, dropout_rate, use_attention)
        self.stage2 = EncoderStage(c1, c2, dropout_rate, use_attention)
        self.stage3 = EncoderStage(c2, c3, dropout_rate, use_attention)
        self.stage4 = EncoderStage(c3, c4, dropout_rate, use_attention)

        self.bottleneck = ResidualBlock(c4, c5, dropout_rate)
        self.bottleneck_att = CBAMBlock(c5) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skip1, x = self.stage1(x)
        skip2, x = self.stage2(x)
        skip3, x = self.stage3(x)
        skip4, x = self.stage4(x)

        x = self.bottleneck(x)
        x = self.bottleneck_att(x)

        return [skip1, skip2, skip3, skip4], x
