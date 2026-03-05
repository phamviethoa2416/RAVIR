import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1.0):
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
    def __init__(self, num_classes: int = 3, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
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
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky[:, 1:].mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=3, ce_weight=1.0, dice_weight=1.0, smooth=1.0, label_smoothing=0.1):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.dice_loss = DiceLoss(num_classes=num_classes, smooth=smooth)
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, class_weights=None):
        if class_weights is not None:
            ce = nn.CrossEntropyLoss(weight=class_weights.to(logits.device),
                                     label_smoothing=self.label_smoothing)(logits, targets.long())
        else:
            ce = self.ce_loss(logits, targets.long())
        return self.ce_weight * ce + self.dice_weight * self.dice_loss(logits, targets)


class TverskyCELoss(nn.Module):
    def __init__(self, num_classes=3, tversky_weight=0.7, ce_weight=0.3,
                 tversky_alpha=0.3, tversky_beta=0.7, smooth=1.0, label_smoothing=0.1):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.ce_weight = ce_weight
        self.tversky_loss = TverskyLoss(num_classes, tversky_alpha, tversky_beta, smooth)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, class_weights=None):
        if class_weights is not None:
            ce = nn.CrossEntropyLoss(weight=class_weights.to(logits.device),
                                     label_smoothing=self.label_smoothing)(logits, targets.long())
        else:
            ce = self.ce_loss(logits, targets.long())
        return self.tversky_weight * self.tversky_loss(logits, targets) + self.ce_weight * ce


# ── Auxiliary head losses ─────────────────────────────────────────────────────


class VesselProbLoss(nn.Module):
    def __init__(self, pos_weight_value: float = 3.0):
        super().__init__()
        self.pos_weight_value = pos_weight_value

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight_value], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


class OrientationLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, vessel_mask: torch.Tensor) -> torch.Tensor:
        mask = vessel_mask.expand_as(pred)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return F.mse_loss(pred * mask, target * mask)


class WidthMapLoss(nn.Module):
    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vessel_mask = (targets > 0).float()
        if vessel_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return F.smooth_l1_loss(pred * vessel_mask, targets * vessel_mask)


class EndpointLoss(nn.Module):
    def __init__(self, pos_weight_value: float = 50.0):
        super().__init__()
        self.pos_weight_value = pos_weight_value

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight_value], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


class MultiHeadLoss(nn.Module):
    def __init__(
            self,
            seg_criterion: nn.Module,
            vessel_prob_weight: float = 0.5,
            orientation_weight: float = 0.3,
            width_weight: float = 0.3,
            endpoint_weight: float = 0.2,
            vessel_prob_pos_weight: float = 3.0,
            endpoint_pos_weight: float = 50.0,
    ):
        super().__init__()
        self.seg_criterion = seg_criterion
        self.vessel_prob_loss = VesselProbLoss(vessel_prob_pos_weight)
        self.orientation_loss = OrientationLoss()
        self.width_loss = WidthMapLoss()
        self.endpoint_loss = EndpointLoss(endpoint_pos_weight)
        self.vessel_prob_weight = vessel_prob_weight
        self.orientation_weight = orientation_weight
        self.width_weight = width_weight
        self.endpoint_weight = endpoint_weight

    def forward(self, outputs, targets, class_weights=None):
        seg = self.seg_criterion(outputs["segmentation"], targets["mask"], class_weights=class_weights)
        vp = self.vessel_prob_loss(outputs["vessel_prob"], targets["vessel_prob"])
        ori = self.orientation_loss(outputs["orientation"], targets["orientation"], targets["vessel_prob"])
        w = self.width_loss(outputs["width"], targets["width"])
        ep = self.endpoint_loss(outputs["endpoint"], targets["endpoint"])

        total = (seg
                 + self.vessel_prob_weight * vp
                 + self.orientation_weight * ori
                 + self.width_weight * w
                 + self.endpoint_weight * ep)

        details = {
            "seg_loss": seg.item(),
            "vessel_prob_loss": vp.item(),
            "orientation_loss": ori.item(),
            "width_loss": w.item(),
            "endpoint_loss": ep.item(),
        }
        return total, details
