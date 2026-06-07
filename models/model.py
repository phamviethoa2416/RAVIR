from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from .encoder import Encoder
from .refinement import RecursiveRefinement


class Model(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        num_classes: int = 3,
        encoder_weights: str | None = "imagenet",
        dropout: float = 0.1,
        refinement_iterations: int = 2,
        refinement_base_channels: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels

        self.encoder = Encoder(
            encoder_name=encoder_name,
            in_channels=in_channels,
            weights=encoder_weights,
            depth=5,
        )

        self.decoder = Decoder(
            skip_channels=self.encoder.skip_channels,
            bottleneck_channel=self.encoder.bottleneck_channel,
            num_classes=num_classes,
            dropout=dropout,
        )

        decoder_final_channel = self.decoder.final_conv.in_channels
        self.vessel_head = nn.Conv2d(decoder_final_channel, 1, kernel_size=1)
        self.refinement = RecursiveRefinement(
            num_iterations=refinement_iterations,
            base_channels=refinement_base_channels,
        )

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        input_size = x.shape[2:]

        skips, bottleneck = self.encoder(x)

        segmentation, ds, decoder_output = self.decoder(skips, bottleneck)

        vessel = self.vessel_head(decoder_output)

        if segmentation.shape[2:] != input_size:
            segmentation = F.interpolate(
                segmentation,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        if vessel.shape[2:] != input_size:
            vessel = F.interpolate(
                vessel,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "segmentation": segmentation,
            "ds": ds,
            "vessel": vessel,
            "refinement": self.refinement(segmentation),
        }

        return outputs
