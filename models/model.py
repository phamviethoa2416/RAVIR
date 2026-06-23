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
        # Frangi auxiliary head
        use_frangi_recon: bool = False,
        aux_mid_channels: int = 64,
        # Contrastive Learning ───────────────────
        cl_projector_stage_idx: int | None = None,
        cl_embedding_dim: int = 64,
        cl_hidden_dim: int = 128,
        # Recursive Refinement ───────────────────
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

        self.embedding_dim = cl_embedding_dim
        self.hidden_dim = cl_hidden_dim

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
        self.cl_projector_stage_idx = cl_projector_stage_idx
        if cl_projector_stage_idx is not None:
            stage_channels = self.infer_stage_channels(
                cl_projector_stage_idx, in_channels
            )
            self.projector: nn.Module | None = TopoProjector(
                in_channels=stage_channels,
                embedding_dim=cl_embedding_dim,
                hidden_dim=cl_hidden_dim,
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

    @torch.no_grad()
    def infer_stage_channels(self, stage_idx: int, in_channels: int) -> int:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        dummy = torch.zeros(1, in_channels, 64, 64, device=device)
        skips, bottleneck = self.encoder(dummy)
        *_, stage_features = self.decoder(skips, bottleneck)
        if not (0 <= stage_idx < len(stage_features)):
            raise ValueError(
                f"Contrastive Learning projection index {stage_idx} out of range"
            )
        channels = stage_features[stage_idx].shape[1]
        if was_training:
            self.train()
        return channels

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
            "embedding": None,
            "refinement": (
                self.refinement(segmentation) if self.refinement is not None else []
            ),
            "frangi_recon_logits": None,
        }

        if self.projector is not None and self.cl_projector_stage_idx is not None:
            stage_feat = stage_features[self.cl_projector_stage_idx]
            embedding = self.projector(stage_feat)
            assert (
                input_size[0] % embedding.shape[-2] == 0
                and input_size[1] % embedding.shape[-1] == 0
            ), f"Embedding {tuple(embedding.shape[-2:])} does not evenly divide input"
            outputs["embedding"] = embedding

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
