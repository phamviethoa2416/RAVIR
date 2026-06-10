from __future__ import annotations

import torch
import torch.nn as nn

from .cldice import ClDiceLoss
from .dice import DiceLoss


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        skeleton_weight: float = 1.0,
        ds_weight: float = 0.4,
        ds_decay: float = 0.8,
        class_weights: torch.Tensor | None = None,
        cldice_num_iterations: int = 10,
        refinement_mode: str = "uniform",
        refinement_iteration_weight: float = 1.0,
        refinement_base_segmentation_weight: float = 1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.skeleton_weight = skeleton_weight
        self.ds_weight = ds_weight
        self.ds_decay = ds_decay

        if refinement_mode not in ("uniform", "linear", "last_only"):
            raise ValueError(
                f"Refinement mode must be one of {['uniform', 'linear', 'last_only']}, "
                f"but got {refinement_mode}"
            )

        self.refinement_mode = refinement_mode
        self.refinement_iteration_weight = refinement_iteration_weight
        self.refinement_base_segmentation_weight = refinement_base_segmentation_weight

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
        )
        self.dice_loss = DiceLoss(num_classes=num_classes)

        self.skeleton_loss = ClDiceLoss(
            num_classes=num_classes,
            num_iterations=cldice_num_iterations,
        )

    def segmentation_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.ce_weight * self.ce_loss(
            logits, targets.long()
        ) + self.dice_weight * self.dice_loss(logits, targets)

    def refinement_iteration_weights(self, K: int) -> list[float]:
        if self.refinement_mode == "uniform":
            w = [1.0] * K
        elif self.refinement_mode == "linear":
            w = [(k + 1) / float(K) for k in range(K)]
        else:
            w = [0.0] * K
            w[-1] = 1.0
        w[-1] = w[-1] * self.refinement_iteration_weight
        return w

    def forward(
        self,
        outputs: dict[str, torch.Tensor | list[torch.Tensor] | None],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        segmentation_logits = outputs["segmentation"]
        ds_logits: list[torch.Tensor] = outputs.get("ds", []) or []
        refinement = outputs.get("refinement") or []
        has_refinement = isinstance(refinement, list) and len(refinement) > 0

        base_segmentation_loss = self.segmentation_loss(segmentation_logits, targets)
        base_weight = (
            self.refinement_base_segmentation_weight if has_refinement else 1.0
        )
        total = base_weight * base_segmentation_loss

        details: dict[str, float] = {
            "segmentation": base_segmentation_loss.item(),
        }

        topology_logits = segmentation_logits
        if has_refinement:
            K = len(refinement)
            refinement_total = torch.tensor(0.0, device=segmentation_logits.device)
            for k, (refinement_logits, wk) in enumerate(
                zip(refinement, self.refinement_iteration_weights(K))
            ):
                if wk == 0.0:
                    continue
                refinement_loss_k = self.segmentation_loss(refinement_logits, targets)
                refinement_total = refinement_total + refinement_loss_k * wk
                details[f"refinement_{k}"] = refinement_loss_k.item()
            total = total + refinement_total
            details["refinement_total"] = refinement_total.item()
            topology_logits = refinement[-1]

        if self.skeleton_weight > 0:
            loss_skeleton = self.skeleton_loss(topology_logits, targets)
            total = total + self.skeleton_weight * loss_skeleton
            details["cldice"] = loss_skeleton.item()

        if ds_logits:
            ds_total = torch.tensor(0.0, device=segmentation_logits.device)
            for i, ds_logit in enumerate(ds_logits):
                ds_total = ds_total + self.ds_weight * (
                    self.ds_decay**i
                ) * self.segmentation_loss(ds_logit, targets)
            total = total + ds_total
            details["ds"] = ds_total.item()

        details["total"] = total.item()
        return total, details
