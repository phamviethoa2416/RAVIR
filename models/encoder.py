from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        depth: int = 5,
        weights: str | None = "imagenet",
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=depth,
            weights=weights,
        )

        self.out_channels: list[int] = list(self.encoder.out_channels)

    @property
    def skip_channels(self) -> list[int]:
        return self.out_channels[:-1]

    @property
    def bottleneck_channel(self) -> int:
        return self.out_channels[-1]

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        features = self.encoder(x)
        skips = list(features[:-1])
        bottleneck = features[-1]
        return skips, bottleneck
