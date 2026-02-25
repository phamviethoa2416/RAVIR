import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)

        one_hot = (
            F.one_hot(targets.long(), self.num_classes)
             .permute(0, 3, 1, 2)
             .float()
        )

        intersection = (probs * one_hot).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = (1.0 - dice).mean()

        return dice_loss

class CombinedLoss(nn.Module):
    def __init__(
            self,
            num_classes: int = 3,
            ce_weight: float = 1.0,
            dice_weight: float = 1.0,
            smooth: float = 1.0,
            label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.dice_loss = DiceLoss(num_classes=num_classes, smooth=smooth)
        self.label_smoothing = label_smoothing

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            class_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if class_weights is not None:
            ce = nn.CrossEntropyLoss(
                weight=class_weights.to(logits.device),
                label_smoothing=self.label_smoothing,
            )(logits, targets.long())
        else:
            ce = self.ce_loss(logits, targets.long())

        dice = self.dice_loss(logits, targets)

        return self.ce_weight * ce + self.dice_weight * dice
