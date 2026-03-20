from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    def __init__(
            self,
            encoder_name: str = "efficientnet-b4",
            in_channels: int = 3,
            depth: int = 5,
            weights: str | None = "imagenet",
            freeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=depth,
            weights=weights,
        )

        self._out_channels: list[int] = list(self.encoder.out_channels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @property
    def skip_channels(self) -> list[int]:
        return self._out_channels[:-1]

    @property
    def bottleneck_channels(self) -> int:
        return self._out_channels[-1]

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        features = self.encoder(x)
        skips = list(features[:-1])
        bottleneck = features[-1]
        return skips, bottleneck
