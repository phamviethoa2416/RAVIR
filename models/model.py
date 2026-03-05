from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PretrainedEncoder
from .decoder import FeatureDecoder


class RAVIRNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 3,
            encoder_name: str = "resnet34",
            encoder_weights: str | None = "imagenet",
            dropout_rate: float = 0.1,
            use_attention: bool = True,
            freeze_encoder: bool = False,
    ):
        super().__init__()

        # ── Pretrained encoder ────────────────────────────────────────
        self.encoder = PretrainedEncoder(
            encoder_name=encoder_name,
            in_channels=in_channels,
            weights=encoder_weights,
            depth=5,
            freeze_encoder=freeze_encoder,
        )

        # ── Decoder (channel sizes driven by encoder) ─────────────────
        self.decoder = FeatureDecoder(
            skip_channels=self.encoder.skip_channels,
            bottleneck_channels=self.encoder.bottleneck_channels,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
        )

        dec_out = self.decoder.output_channels

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dec_out, dec_out, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dec_out),
            nn.ReLU(inplace=True),
        )

        # ── Head 1: semantic segmentation (bg / artery / vein) ────────
        self.seg_head = nn.Conv2d(dec_out, num_classes, kernel_size=1)

        # ── Head 2: binary vessel probability ─────────────────────────
        self.vessel_prob_head = nn.Conv2d(dec_out, 1, kernel_size=1)

        self._init_heads()

    def _init_heads(self):
        for module in [self.decoder, self.final_up, self.seg_head, self.vessel_prob_head]:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[2:]

        skips, bottleneck = self.encoder(x)
        features = self.decoder(skips, bottleneck)
        features = self.final_up(features)

        if features.shape[2:] != input_size:
            features = F.interpolate(features, size=input_size, mode="bilinear", align_corners=False)

        return {
            "segmentation": self.seg_head(features),
            "vessel_prob": self.vessel_prob_head(features),
        }
