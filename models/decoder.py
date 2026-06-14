from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import AttentionModule, ResidualModule


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channel: int,
        skip_channel: int,
        out_channel: int,
        dropout: float = 0.0,
        use_scse: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        self.attention = (
            AttentionModule(gate_channels=in_channel, skip_channels=skip_channel)
            if use_attention
            else None
        )
        self.conv_block = ResidualModule(
            in_channels=in_channel + skip_channel,
            out_channels=out_channel,
            dropout=dropout,
            use_scse=use_scse,
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

        if self.attention is not None:
            skip = self.attention(gate=x, skip=skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        skip_channels: list[int],
        bottleneck_channel: int,
        num_classes: int = 3,
        dropout: float = 0.1,
        use_scse: bool = False,
        use_attention: bool = False,
        use_deep_supervision: bool = True,
    ):
        super().__init__()
        self.num_stages = len(skip_channels)
        self.use_scse = use_scse
        self.use_attention = use_attention
        self.use_deep_supervision = use_deep_supervision

        reversed_skips = list(reversed(skip_channels))

        min_dec_ch = 32

        stages: list[DecoderStage] = []
        stage_out_channels: list[int] = []
        in_channel = bottleneck_channel
        for i, skip_channel in enumerate(reversed_skips):
            out_channel = max(in_channel // 2, min_dec_ch)
            stages.append(
                DecoderStage(
                    in_channel=in_channel,
                    skip_channel=skip_channel,
                    out_channel=out_channel,
                    dropout=dropout,
                    use_scse=use_scse,
                    use_attention=use_attention,
                )
            )
            stage_out_channels.append(out_channel)
            in_channel = out_channel

        self.stages = nn.ModuleList(stages)
        self.stage_out_channels = stage_out_channels

        self.final_conv = nn.Conv2d(
            in_channels=stage_out_channels[-1],
            out_channels=num_classes,
            kernel_size=1,
        )

        if self.use_deep_supervision:
            ds_heads: list[nn.Module] = []
            for i in range(self.num_stages - 1):
                ds_heads.append(
                    nn.Conv2d(
                        in_channels=stage_out_channels[i],
                        out_channels=num_classes,
                        kernel_size=1,
                    )
                )
            self.ds_heads = nn.ModuleList(ds_heads)

    def forward(
        self,
        skips: list[torch.Tensor],
        bottleneck: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        reversed_skips = list(reversed(skips))

        target_size = skips[0].shape[2:]

        x = bottleneck
        ds_outputs: list[torch.Tensor] = []
        stage_features: list[torch.Tensor] = []

        for i, stage in enumerate(self.stages):
            x = stage(x, reversed_skips[i])
            stage_features.append(x)

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

        segmentation = self.final_conv(x)

        if segmentation.shape[2:] != target_size:
            segmentation = F.interpolate(
                segmentation,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        return segmentation, ds_outputs[::-1], stage_features
