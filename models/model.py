import torch
import torch.nn as nn

from .encoder import FeatureEncoder
from .decoder import FeatureDecoder


class RAVIRNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 3,
            channels: list[int] | None = None,
            dropout_rate: float = 0.1,
            use_attention: bool = True,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512, 1024]

        self.encoder = FeatureEncoder(
            in_channels=in_channels,
            channels=channels,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
        )
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        skips, bottleneck = self.encoder(x)
        features = self.decoder(skips, bottleneck)

        return {
            "segmentation": self.seg_head(features),
            "vessel_prob": self.vessel_prob_head(features),
        }
