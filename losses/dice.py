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
