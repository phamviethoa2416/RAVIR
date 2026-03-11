from __future__ import annotations

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
    """Original handcrafted encoder used in RAVIRNet."""

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


class SMPFeatureEncoder(nn.Module):
    """Encoder wrapper using segmentation_models_pytorch + timm backbones.

    This exposes the same interface as FeatureEncoder:
      forward(x) -> (skips[4], bottleneck)
    and records the channel progression so that the decoder can be built
    to match the pretrained backbone.
    """

    def __init__(
            self,
            encoder_name: str,
            in_channels: int = 3,
            depth: int = 5,
            weights: str | None = "imagenet",
    ):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required for SMPFeatureEncoder. "
                "Install it with `pip install segmentation-models-pytorch`."
            ) from exc

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=depth,
            weights=weights,
        )

        # smp encoders expose out_channels for each stage (len = depth + 1).
        enc_channels = list(self.encoder.out_channels)
        if len(enc_channels) < 5:
            raise ValueError(
                f"Encoder '{encoder_name}' (depth={depth}) exposes only "
                f"{len(enc_channels)} feature levels, but at least 5 are required "
                "to provide 4 skip connections + bottleneck.",
            )

        # Use last 5 feature levels as [c1, c2, c3, c4, c5]
        self.channels: list[int] = enc_channels[-5:]

    @property
    def out_channels(self) -> list[int]:
        """Channel progression [c1, c2, c3, c4, c5] matching skips + bottleneck."""
        return self.channels

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        # smp encoder returns a list of feature maps for each stage (len = depth + 1)
        features = self.encoder(x)
        if len(features) < 5:
            raise RuntimeError(
                f"Encoder produced only {len(features)} feature maps; expected at least 5.",
            )

        # Take last 5 feature maps: smallest spatial size is bottleneck
        f1, f2, f3, f4, bottleneck = features[-5:]
        return [f1, f2, f3, f4], bottleneck