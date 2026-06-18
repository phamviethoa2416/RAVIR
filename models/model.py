from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aux_recon import AuxReconHead
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
        # Contrastive Learning ───────────────────
        cgt_projector_stage_idx: int | None = 2,
        cgt_embedding_dim: int = 64,
        cgt_hidden_dim: int = 128,
        # Recursive Refinement ───────────────────
        use_refinement: bool = False,
        refinement_iterations: int = 2,
        refinement_base_channels: int = 32,
        # Auxiliary skeleton reconstruction
        use_skeleton_recon: bool = False,
        aux_mid_channels: int = 64,
        # Auxiliary Frangi reconstruction
        use_frangi_recon: bool = False,
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

        decoder_final_channels = self.decoder.final_conv.in_channels

        # Projector head for Contrastive Learning ────────────
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

        # Recursive Refinement ────────────
        self.refinement = (
            RecursiveRefinement(
                num_iterations=refinement_iterations,
                base_channels=refinement_base_channels,
            )
            if self.use_refinement
            else None
        )

        # Auxiliary skeleton reconstruction head ─────────────────
        self.use_skeleton_recon = use_skeleton_recon
        if self.use_skeleton_recon:
            skeleton_out_channels = 2
            self.skeleton_head: AuxReconHead | None = AuxReconHead(
                in_channels=decoder_final_channels,
                out_channels=skeleton_out_channels,
                mid_channels=aux_mid_channels,
            )
        else:
            self.skeleton_head = None

        # Auxiliary Frangi reconstruction head ─────────────────
        self.use_frangi_recon = use_frangi_recon
        if self.use_frangi_recon:
            self.frangi_head: AuxReconHead | None = AuxReconHead(
                in_channels=decoder_final_channels,
                out_channels=1,
                mid_channels=aux_mid_channels,
            )
        else:
            self.frangi_head = None

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        input_size = x.shape[2:]

        skips, bottleneck = self.encoder(x)

        segmentation, ds, decoder_output, stage_features = self.decoder(
            skips, bottleneck
        )

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
            "skeleton_recon_logits": None,
            "frangi_recon_logits": None,
        }

        if self.skeleton_head is not None:
            skel_logits = self.skeleton_head(decoder_output)
            if skel_logits.shape[2:] != input_size:
                skel_logits = nn.functional.interpolate(
                    skel_logits,
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            outputs["skeleton_recon_logits"] = skel_logits

        if self.frangi_head is not None:
            frangi_logits = self.frangi_head(decoder_output)
            if frangi_logits.shape[2:] != input_size:
                frangi_logits = nn.functional.interpolate(
                    frangi_logits,
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            outputs["frangi_recon_logits"] = frangi_logits

        return outputs
