from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import FeatureDecoder
from .encoder import FeatureEncoder


class RAVIRNet(nn.Module):
    def __init__(
            self,
            encoder_name: str = "efficientnet-b4",
            in_channels: int = 3,
            num_classes: int = 3,
            encoder_weights: str | None = "imagenet",
            freeze_encoder: bool = False,
            use_deep_supervision: bool = True,
            dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FeatureEncoder(
            encoder_name=encoder_name,
            in_channels=in_channels,
            weights=encoder_weights,
            depth=5,
            freeze_encoder=freeze_encoder,
        )

        self.decoder = FeatureDecoder(
            skip_channels=self.encoder.skip_channels,
            bottleneck_channels=self.encoder.bottleneck_channels,
            num_classes=num_classes,
            use_deep_supervision=use_deep_supervision,
            dropout_rate=dropout_rate,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        input_size = x.shape[2:]

        skips, bottleneck = self.encoder(x)

        seg_out, ds_outputs = self.decoder(skips, bottleneck)

        if seg_out.shape[2:] != input_size:
            seg_out = nn.functional.interpolate(
                seg_out,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "seg": seg_out,
            "ds": ds_outputs,
        }

        return outputs
