from __future__ import annotations

import torch

class BinarySegmentationMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.tp: float = 0
        self.fp: float = 0
        self.fn: float = 0
        self.tn: float = 0

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

@torch.no_grad()
def update(self, logits: torch.Tensor, targets: torch.Tensor):
    preds = (torch.sigmoid(logits) > self.threshold).float()
    targets = targets.float()

    self.tp += (preds * targets).sum().item()
    self.fp += (preds * (1 - targets)).sum().item()
    self.fn += ((1 - preds) * targets).sum().item()
    self.tn += ((1 - preds) * (1 - targets)).sum().item

def compute(self) -> dict[str, float]:
    eps = 1e-7
    dice = (2.0 * self.tp + eps) / (2.0 * self.tp + self.fp + self.fn + eps)
    iou = (self.tp + eps) / (self.tp + self.fp + self.fn + eps)
    sensitivity = (self.tp + eps) / (self.tp + self.fn + eps)
    specificity = (self.tn + eps) / (self.tn + self.fp + eps)
    precision = (self.tp + eps) / (self.tp + self.fp + eps)

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
    }