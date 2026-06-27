from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_vessel_target(segmentation_targets: torch.Tensor) -> torch.Tensor:
    if segmentation_targets.dim() == 4:
        segmentation_targets = segmentation_targets.squeeze(1)
    if segmentation_targets.dim() != 3:
        raise ValueError(f"Segmentation targets must be (B, H, W) or (B, 1, H, W)")
    return (segmentation_targets > 0).to(dtype=torch.float32).unsqueeze(1)


def frangi_aux_loss(
    preds: torch.Tensor,
    frangi_target: torch.Tensor,
    segmentation_targets: torch.Tensor,
    *,
    frangi_loss_type: str = "mse",
    smooth_l1_beta: float = 0.1,
) -> torch.Tensor:
    if frangi_target.dim() == 3:
        frangi_target = frangi_target.unsqueeze(1)
    if preds.shape != frangi_target.shape:
        raise ValueError(f"Shape mismatch between preds and frangi target. ")

    frangi_t = frangi_target.to(preds.dtype)
    pred = preds.sigmoid()

    mask = binary_vessel_target(segmentation_targets).to(preds.device)
    if mask.shape != preds.shape:
        raise ValueError(f"Shape mismatch between preds and vessel mask. ")
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

    return loss_frangi
