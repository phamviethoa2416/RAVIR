from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = (
            F.one_hot(targets.long(), self.num_classes).permute(0, 3, 1, 2).float()
        )
        intersection = (probs * one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice[:, 1:].mean()


class CombinedLoss(nn.Module):
    def __init__(
            self,
            num_classes: int = 3,
            ce_weight: float = 1.0,
            dice_weight: float = 1.0,
            label_smoothing: float = 0.0,
            class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(num_classes=num_classes)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets.long())
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class SoftSkeletonRecallLoss(nn.Module):
    def __init__(
            self,
            num_classes: int = 3,
            smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)

        with torch.no_grad():
            skel_onehot = (
                F.one_hot(skeleton.long(), self.num_classes).permute(0, 3, 1, 2).float()
            )

            skel_foreground = skel_onehot[:, 1:]
            sum_skel = skel_foreground.sum(dim=(2, 3))

        probs_fg = probs[:, 1:]

        intersection = (probs_fg * skel_foreground).sum(dim=(2, 3))

        recall = (intersection + self.smooth) / torch.clip(
            sum_skel + self.smooth,
            min=1e-8,
        )

        return 1 - recall.mean()


class VesselSegmentationLoss(nn.Module):
    def __init__(
            self,
            num_classes: int = 3,
            ce_weight: float = 1.0,
            dice_weight: float = 1.0,
            skeleton_weight: float = 1.0,
            ds_weight: float = 0.4,
            ds_decay: float = 0.8,
            label_smoothing: float = 0.0,
            class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        self.skeleton_weight = skeleton_weight
        self.ds_weight = ds_weight
        self.ds_decay = ds_decay

        self.combined_loss = CombinedLoss(
            num_classes=num_classes,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        )

        self.skeleton_loss = SoftSkeletonRecallLoss(num_classes=num_classes)

    def forward(
            self,
            outputs: dict[str, torch.Tensor | list[torch.Tensor]],
            targets: torch.Tensor,
            skeleton: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        seg_logits = outputs["seg"]
        ds_logits_list: list[torch.Tensor] = outputs.get("ds", [])

        loss_seg = self.combined_loss(seg_logits, targets)
        total = loss_seg

        details: dict[str, float] = {
            "seg": loss_seg.item(),
        }

        if skeleton is not None and self.skeleton_weight > 0:
            loss_skel = self.skeleton_loss(seg_logits, skeleton)
            total = total + self.skeleton_weight * loss_skel
            details["skel"] = loss_skel.item()

        if ds_logits_list:
            ds_total = torch.tensor(0.0, device=seg_logits.device)

            for i, ds_logits in enumerate(ds_logits_list):
                w = self.ds_weight * (self.ds_decay ** i)
                ds_loss_i = self.combined_loss(ds_logits, targets)
                ds_total = ds_total + w * ds_loss_i

            total = total + ds_total
            details["ds"] = ds_total.item()

        details["total"] = total.item()
        return total, details
