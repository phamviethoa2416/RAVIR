from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class PretrainedEncoder(nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            in_channels: int = 1,
            weights: str | None = "imagenet",
            depth: int = 5,
            freeze_encoder: bool = False,
    ):
        super().__init__()
        self.backbone = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=depth,
            weights=weights,
        )
        self._out_channels: tuple[int, ...] = self.backbone.out_channels

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def skip_channels(self) -> list[int]:
        return list(self._out_channels[1:-1])

    @property
    def bottleneck_channels(self) -> int:
        return self._out_channels[-1]

    @property
    def out_channels(self) -> tuple[int, ...]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        features = self.backbone(x)
        skips = list(features[1:-1])  # 4 tensors
        bottleneck = features[-1]  # 1 tensor
        return skips, bottleneck
