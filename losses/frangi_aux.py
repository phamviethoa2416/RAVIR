from __future__ import annotations

import torch
import torch.nn.functional as F

POS_WEIGHT_CLAMP = 50.0


def binary_vessel_target(segmentation_targets: torch.Tensor) -> torch.Tensor:
    if segmentation_targets.dim() == 4:
        segmentation_targets = segmentation_targets.squeeze(1)
    if segmentation_targets.dim() != 3:
        raise ValueError(f"Segmentation targets must be (B, H, W) or (B, 1, H, W)")
    return (segmentation_targets > 0).to(dtype=torch.float32).unsqueeze(1)


def dynamic_pos_weight(
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    positive = target.sum(dim=(0, 2, 3))
    total = float(target.shape[0] * target.shape[2] * target.shape[3])
    negative = total - positive
    ratio = negative / (positive + eps)
    return ratio.clamp(max=POS_WEIGHT_CLAMP)


def frangi_aux_loss(
    preds: torch.Tensor,
    frangi_target: torch.Tensor,
    segmentation_targets: torch.Tensor,
    *,
    vessel_weight: float = 1.0,
    frangi_weight: float = 0.5,
    frangi_on_vessel_only: bool = True,
    frangi_loss_type: str = "mse",
    smooth_l1_beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if frangi_target.dim() == 3:
        frangi_target = frangi_target.unsqueeze(1)
    if preds.shape != frangi_target.shape:
        raise ValueError(f"Shape mismatch between preds and frangi target. ")

    vessel = binary_vessel_target(segmentation_targets).to(preds.device)
    if vessel.shape != preds.shape:
        raise ValueError(f"Shape mismatch between preds and vessel target. ")

    frangi_t = frangi_target.to(preds.dtype)
    pos_weight = dynamic_pos_weight(vessel).to(preds.dtype).view(1, 1, 1, 1)
    loss_vessel = F.binary_cross_entropy_with_logits(
        preds,
        vessel.to(preds.dtype),
        pos_weight=pos_weight,
        reduction="mean",
    )

    pred = preds.sigmoid()
    if frangi_on_vessel_only:
        mask = vessel
        denom = mask.sum().clamp(min=1.0)
        if frangi_loss_type == "mse":
            loss_frangi = ((pred - frangi_t).pow(2) * mask).sum() / denom
        elif frangi_loss_type == "l1":
            loss_frangi = ((pred - frangi_t).abs() * mask).sum() / denom
        elif frangi_loss_type == "smooth_l1":
            loss_frangi = (
                F.smooth_l1_loss(
                    pred * mask,
                    frangi_t * mask,
                    reduction="sum",
                    beta=float(smooth_l1_beta),
                )
                / denom
            )
        else:
            raise ValueError("Frangi loss type not supported.")
    else:
        if frangi_loss_type == "mse":
            loss_frangi = F.mse_loss(pred, frangi_t, reduction="mean")
        elif frangi_loss_type == "l1":
            loss_frangi = F.l1_loss(pred, frangi_t, reduction="mean")
        elif frangi_loss_type == "smooth_l1":
            loss_frangi = F.smooth_l1_loss(
                pred, frangi_t, reduction="mean", beta=float(smooth_l1_beta)
            )
        else:
            raise ValueError("Frangi loss type not supported.")

    total = vessel_weight * loss_vessel + frangi_weight * loss_frangi
    return total, {
        "vessel": loss_vessel,
        "frangi": loss_frangi,
    }
