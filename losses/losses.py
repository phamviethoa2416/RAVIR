import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        one_hot = (
            F.one_hot(targets.long(), self.num_classes)
            .permute(0, 3, 1, 2).float()
        )
        intersection = (probs * one_hot).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice[:, 1:]).mean()


class TverskyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        one_hot = (
            F.one_hot(targets.long(), self.num_classes)
            .permute(0, 3, 1, 2).float()
        )
        tp = (probs * one_hot).sum(dim=(2, 3))
        fp = (probs * (1 - one_hot)).sum(dim=(2, 3))
        fn = ((1 - probs) * one_hot).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1 - tversky[:, 1:].mean()

class BinaryDiceBCELoss(nn.Module):
    def __init__(
            self,
            dice_weight: float = 0.5,
            bce_weight: float = 0.5,
            smooth: float = 1e-5,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce


class BinaryTverskyLoss(nn.Module):
    """Binary Tversky loss operating on logits (sigmoid inside)."""

    def __init__(
            self,
            alpha: float = 0.3,
            beta: float = 0.7,
            smooth: float = 1e-5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky.mean()

class BinarySkeletonRecallLoss(nn.Module):
    """Soft Skeleton Recall (Kirchhoff et al., ECCV 2024).

    Computes soft recall of the prediction on a precomputed *tubed skeleton*
    of the ground truth.  Operates on raw logits (sigmoid applied inside).
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(
            self,
            logits: torch.Tensor,
            skeleton: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        skeleton = skeleton.float()

        inter = (probs * skeleton).sum(dim=(2, 3))
        skel_sum = skeleton.sum(dim=(2, 3))

        recall = (inter + self.smooth) / (skel_sum.clamp(min=self.smooth) + self.smooth)
        return 1.0 - recall.mean()


class BinaryDiceBCESkelRecallLoss(nn.Module):
    """Dice + BCE + Skeleton Recall for binary vessel segmentation.

    L = w_dice * L_dice + w_bce * L_bce + w_skel * L_skeleton_recall

    Reference: "Skeleton Recall Loss for Connectivity Conserving and
    Resource Efficient Segmentation of Thin Tubular Structures"
    (Kirchhoff et al., ECCV 2024)
    """

    def __init__(
            self,
            weight_dice: float = 1.0,
            weight_bce: float = 1.0,
            weight_skel: float = 1.0,
            smooth: float = 1e-5,
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.weight_skel = weight_skel
        self.smooth = smooth

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            skeleton: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice.mean()

        total = self.weight_dice * dice_loss + self.weight_bce * bce

        if skeleton is not None:
            skeleton = skeleton.float()
            skel_inter = (probs * skeleton).sum(dim=(2, 3))
            skel_sum = skeleton.sum(dim=(2, 3))
            skel_recall = (skel_inter + self.smooth) / (
                skel_sum.clamp(min=self.smooth) + self.smooth
            )
            total = total + self.weight_skel * (1.0 - skel_recall.mean())

        return total


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice loss for semantic segmentation."""

    def __init__(
        self,
        num_classes: int = 3,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-5,
        label_smoothing: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        self.dice_loss = DiceLoss(num_classes=num_classes, smooth=smooth)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self._build_ce()

    def _build_ce(self):
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets.long())
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class TverskyCELoss(nn.Module):
    """Tversky + CrossEntropy loss for semantic segmentation."""

    def __init__(
        self,
        num_classes: int = 3,
        tversky_weight: float = 0.7,
        ce_weight: float = 0.3,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        smooth: float = 1e-5,
        label_smoothing: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.ce_weight = ce_weight
        self.label_smoothing = label_smoothing
        self.tversky_loss = TverskyLoss(num_classes, tversky_alpha, tversky_beta, smooth)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self._build_ce()

    def _build_ce(self):
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets.long())
        tversky = self.tversky_loss(logits, targets)
        return self.tversky_weight * tversky + self.ce_weight * ce


# ── Auxiliary head loss ───────────────────────────────────────────────────────

class VesselProbLoss(nn.Module):
    def __init__(self, pos_weight_value: float = 3.0):
        super().__init__()
        self.register_buffer(
            "pos_weight", torch.tensor([pos_weight_value]),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight,
        )


# ── Multi-head loss (seg + vessel_prob) ───────────────────────────────────────


class MultiHeadLoss(nn.Module):
    def __init__(
        self,
        seg_criterion: nn.Module,
        vessel_prob_weight: float = 0.5,
        vessel_prob_pos_weight: float = 3.0,
    ):
        super().__init__()
        self.seg_criterion = seg_criterion
        self.vessel_prob_loss = VesselProbLoss(vessel_prob_pos_weight)
        self.vessel_prob_weight = vessel_prob_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        seg = self.seg_criterion(outputs["segmentation"], targets["mask"])
        vp = self.vessel_prob_loss(outputs["vessel_prob"], targets["vessel_prob"])

        total = seg + self.vessel_prob_weight * vp

        details = {
            "seg_loss": seg.item(),
            "vessel_prob_loss": vp.item(),
        }
        return total, details