from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    for group in (max_groups, 4, 2, 1):
        if num_channels % group == 0:
            return nn.GroupNorm(group, num_channels)
    return nn.GroupNorm(1, num_channels)


class RRConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RRSubnet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        final_zero_init: bool = True,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = RRConvBlock(in_channels=in_channels, out_channels=c1)
        self.enc2 = RRConvBlock(in_channels=c1, out_channels=c2)
        self.enc3 = RRConvBlock(in_channels=c2, out_channels=c3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = RRConvBlock(in_channels=c3, out_channels=c4)

        self.up3 = nn.ConvTranspose2d(
            in_channels=c4,
            out_channels=c3,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = RRConvBlock(in_channels=c4, out_channels=c3)

        self.up2 = nn.ConvTranspose2d(
            in_channels=c3,
            out_channels=c2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = RRConvBlock(in_channels=c3, out_channels=c2)

        self.up1 = nn.ConvTranspose2d(
            in_channels=c2,
            out_channels=c1,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = RRConvBlock(in_channels=c2, out_channels=c1)

        self.head = nn.Conv2d(
            in_channels=c1,
            out_channels=out_channels,
            kernel_size=1,
        )

        if final_zero_init:
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        pad_h = (-H) % 8
        pad_w = (-W) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.dec3(torch.cat([self.up3(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        out = self.head(dec1)

        if pad_h or pad_w:
            out = out[..., :H, :W]

        return out


class RecursiveRefinement(nn.Module):
    def __init__(
        self,
        num_iterations: int = 2,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        self.num_iterations = num_iterations

        self.subnet = RRSubnet(
            in_channels=3,
            out_channels=2,
            base_channels=base_channels,
            final_zero_init=True,
        )

    def forward(self, base: torch.Tensor) -> list[torch.Tensor]:
        refined: list[torch.Tensor] = []

        bg_logits = base[:, 0:1].detach()
        vessel_logits = base[:, 1:3].max(dim=1, keepdim=True).values.detach()

        base_probs = base.softmax(dim=1)
        vessel_prob = (base_probs[:, 1:2] + base_probs[:, 2:3]).clamp(1e-6, 1.0 - 1e-6)

        current = base[:, 1:3]

        for _ in range(self.num_iterations):
            av_probs = current.softmax(dim=1)

            rr_input = torch.cat(
                [
                    av_probs[:, 0:1],  # artery probs
                    av_probs[:, 1:2],  # vein probs
                    vessel_prob,
                ],
                dim=1,
            )

            delta = self.subnet(rr_input)
            current = current + delta

            m = current.max(dim=1, keepdim=True).values
            artery_logit = vessel_logits + (current[:, 0:1] - m)
            vein_logit = vessel_logits + (current[:, 1:2] - m)

            current_logits = torch.cat(
                [bg_logits, artery_logit, vein_logit],
                dim=1,
            )

            refined.append(current_logits)

        return refined
