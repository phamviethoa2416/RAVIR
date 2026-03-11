import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import FeatureEncoder, SMPFeatureEncoder
from .decoder import FeatureDecoder
from .blocks import ASPP


class RAVIRNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 3,
            channels: list[int] | None = None,
            dropout_rate: float = 0.1,
            use_attention: bool = True,
            use_deep_supervision: bool = True,
            # Encoder configuration
            encoder_type: str = "custom",          # "custom" | "smp"
            encoder_name: str | None = None,       # e.g. "timm-resnet34"
            encoder_weights: str | None = "imagenet",
            encoder_depth: int = 5,
    ):
        """
        encoder_type:
          - "custom": use handcrafted FeatureEncoder (default, backwards compatible)
          - "smp"   : use SMPFeatureEncoder with a timm backbone
        """
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512, 1024]

        self.use_deep_supervision = use_deep_supervision

        # ── Encoder selection ────────────────────────────────────────────
        if encoder_type.lower() == "smp":
            if encoder_name is None:
                encoder_name = "timm-resnet34"
            smp_encoder = SMPFeatureEncoder(
                encoder_name=encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )
            self.encoder = smp_encoder
            channels = smp_encoder.out_channels
        else:
            self.encoder = FeatureEncoder(
                in_channels=in_channels,
                channels=channels,
                dropout_rate=dropout_rate,
                use_attention=use_attention,
            )

        self.aspp = ASPP(channels[-1], channels[-1])

        self.decoder = FeatureDecoder(
            channels=channels,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
        )

        c1 = channels[0]

        # Head 1: semantic segmentation (bg / artery / vein)
        self.seg_head = nn.Conv2d(c1, num_classes, kernel_size=1)

        # Head 2: binary vessel probability
        self.vessel_prob_head = nn.Conv2d(c1, 1, kernel_size=1)

        # Deep supervision: auxiliary segmentation heads at decoder stages 4/3/2
        if use_deep_supervision:
            self.ds_seg_heads = nn.ModuleList([
                nn.Conv2d(channels[3], num_classes, kernel_size=1),  # dec4 output
                nn.Conv2d(channels[2], num_classes, kernel_size=1),  # dec3 output
                nn.Conv2d(channels[1], num_classes, kernel_size=1),  # dec2 output
            ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[2:]

        skips, bottleneck = self.encoder(x)
        bottleneck = self.aspp(bottleneck)
        features, intermediates = self.decoder(skips, bottleneck)

        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "segmentation": self.seg_head(features),
            "vessel_prob": self.vessel_prob_head(features),
        }

        if self.use_deep_supervision and self.training:
            result["deep_supervision"] = [
                F.interpolate(
                    head(feat), size=input_size,
                    mode="bilinear", align_corners=False,
                )
                for head, feat in zip(self.ds_seg_heads, intermediates)
            ]

        return result
