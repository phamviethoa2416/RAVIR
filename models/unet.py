import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        downsampled = self.pool(features)
        return features, downsampled

class Decoder(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.up_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        self.conv = ConvBlock(
            in_channels=in_channels // 2 + skip_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.up_conv(x)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 3,
            channels: list[int] = None,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512, 1024]
        assert len(channels) == 5, "Number of channels must be 5"
        c1, c2, c3, c4, c5 = channels

        self.enc1 = Encoder(in_channels, c1, dropout_rate)
        self.enc2 = Encoder(c1, c2, dropout_rate)
        self.enc3 = Encoder(c2, c3, dropout_rate)
        self.enc4 = Encoder(c3, c4, dropout_rate)

        self.bottleneck = ConvBlock(c4, c5, dropout_rate)

        self.dec4 = Decoder(c5, c4, c4, dropout_rate)
        self.dec3 = Decoder(c4, c3, c3, dropout_rate)
        self.dec2 = Decoder(c3, c2, c2, dropout_rate)
        self.dec1 = Decoder(c2, c1, c1, dropout_rate)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return self.head(x)