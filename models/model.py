from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from .encoder import Encoder
from .projector import TopoProjector
from .refinement import RecursiveRefinement


class RAVIRNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        num_classes: int = 3,
        encoder_weights: str | None = "imagenet",
        dropout: float = 0.1,
        use_scse: bool = False,
        use_attention: bool = False,
        use_deep_supervision: bool = True,
        use_contrastive: bool = False,
        cgt_projector_stage_idx: int | None = 2,
        cgt_embedding_dim: int = 64,
        cgt_hidden_dim: int = 128,
        use_refinement: bool = False,
        refinement_iterations: int = 2,
        refinement_base_channels: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.use_scse = use_scse
        self.use_attention = use_attention
        self.use_deep_supervision = use_deep_supervision
        self.use_refinement = use_refinement
        self.use_contrastive = use_contrastive
        self.cgt_projector_stage_idx = cgt_projector_stage_idx
        self.embedding_dim = cgt_embedding_dim
        self.hidden_dim = cgt_hidden_dim

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
            use_scse=use_scse,
            use_attention=use_attention,
            use_deep_supervision=use_deep_supervision,
        )

        if self.use_contrastive and cgt_projector_stage_idx is not None:
            if not (
                0 <= cgt_projector_stage_idx < len(self.decoder.stage_out_channels)
            ):
                raise ValueError("Stage index out of range")
            stage_channels = self.decoder.stage_out_channels[cgt_projector_stage_idx]
            self.projector: nn.Module | None = TopoProjector(
                in_channels=stage_channels,
                embedding_dim=cgt_embedding_dim,
                hidden_dim=cgt_hidden_dim,
            )
        else:
            self.projector = None

        self.refinement = (
            RecursiveRefinement(
                num_iterations=refinement_iterations,
                base_channels=refinement_base_channels,
            )
            if self.use_refinement
            else None
        )

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        input_size = x.shape[2:]

        skips, bottleneck = self.encoder(x)

        segmentation, ds, stage_features = self.decoder(skips, bottleneck)

        if segmentation.shape[2:] != input_size:
            segmentation = F.interpolate(
                segmentation,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        outputs: dict[str, torch.Tensor | list[torch.Tensor] | None] = {
            "segmentation": segmentation,
            "ds": ds,
            "embedding": (
                self.projector(stage_features[self.cgt_projector_stage_idx])
                if self.projector is not None
                and self.cgt_projector_stage_idx is not None
                else None
            ),
            "refinement": (
                self.refinement(segmentation) if self.refinement is not None else []
            ),
        }

        return outputs
