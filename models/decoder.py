from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock, AttentionGate


class DecoderStage(nn.Module):
    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
        )
        self.attention_gate = AttentionGate(
            gate_channels=in_channels,
            skip_channels=skip_channels,
        )

        self.conv_block = ResidualBlock(
            in_channels + skip_channels,
            out_channels,
            dropout_rate,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        skip = self.attention_gate(gate=x, skip=skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)

        return x


class FeatureDecoder(nn.Module):
    def __init__(
            self,
            skip_channels: list[int],
            bottleneck_channels: int,
            num_classes: int = 3,
            use_deep_supervision: bool = True,
            dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.use_deep_supervision = use_deep_supervision
        self.num_stages = len(skip_channels)

        reversed_skips = list(reversed(skip_channels))

        stages: list[DecoderStage] = []
        in_ch = bottleneck_channels
        for i, sk_ch in enumerate(reversed_skips):
            out_ch = sk_ch
            drop = dropout_rate if i < 2 else 0.0
            stages.append(DecoderStage(in_ch, sk_ch, out_ch, drop))
            in_ch = out_ch

        self.stages = nn.ModuleList(stages)

        self.final_conv = nn.Conv2d(reversed_skips[-1], num_classes, kernel_size=1)

        if use_deep_supervision:
            ds_heads: list[nn.Module] = []
            for i in range(self.num_stages - 1):
                ds_heads.append(
                    nn.Conv2d(reversed_skips[i], num_classes, kernel_size=1)
                )
            self.ds_heads = nn.ModuleList(ds_heads)

    def forward(
            self,
            skips: list[torch.Tensor],
            bottleneck: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        reversed_skips = list(reversed(skips))

        target_size = skips[0].shape[2:]

        x = bottleneck
        ds_outputs: list[torch.Tensor] = []

        for i, stage in enumerate(self.stages):
            x = stage(x, reversed_skips[i])

            if self.use_deep_supervision and i < self.num_stages - 1:
                ds_logits = self.ds_heads[i](x)
                if ds_logits.shape[2:] != target_size:
                    ds_logits = F.interpolate(
                        ds_logits,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                ds_outputs.append(ds_logits)

        seg_out = self.final_conv(x)

        if seg_out.shape[2:] != target_size:
            seg_out = F.interpolate(
                seg_out,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        return seg_out, ds_outputs
