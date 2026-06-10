from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int = 3,
        class_names: list[str] | None = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [
            "background",
            "artery",
            "vein",
        ]

        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.per_image_dice: list[dict[str, float]] = []

    def reset(self) -> None:
        self.confusion_matrix.fill(0)
        self.per_image_dice.clear()

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        preds_np = predictions.detach().cpu().numpy().astype(np.int64)
        targets_np = targets.detach().cpu().numpy().astype(np.int64)

        B = preds_np.shape[0]
        for b in range(B):
            p_flat = preds_np[b].ravel()
            t_flat = targets_np[b].ravel()

            valid = (
                (t_flat >= 0)
                & (t_flat < self.num_classes)
                & (p_flat >= 0)
                & (p_flat < self.num_classes)
            )
            p_valid = p_flat[valid]
            t_valid = t_flat[valid]

            indices = t_valid * self.num_classes + p_valid
            cm_flat = np.bincount(indices, minlength=self.num_classes**2)
            self.confusion_matrix += cm_flat.reshape(
                self.num_classes,
                self.num_classes,
            )

            img_dice = {}
            for cls in range(self.num_classes):
                tp = ((p_valid == cls) & (t_valid == cls)).sum()
                fp = ((p_valid == cls) & (t_valid != cls)).sum()
                fn = ((p_valid != cls) & (t_valid == cls)).sum()
                dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
                img_dice[self.class_names[cls]] = float(dice)
            self.per_image_dice.append(img_dice)

    def compute(self) -> dict[str, float]:
        eps = 1e-7
        cm = self.confusion_matrix
        total = cm.sum()
        metrics: dict[str, float] = {}

        dice_list: list[float] = []
        iou_list: list[float] = []
        sens_list: list[float] = []
        spec_list: list[float] = []
        precision_list: list[float] = []

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            tn = total - tp - fp - fn

            dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            iou = (tp + eps) / (tp + fp + fn + eps)
            sens = (tp + eps) / (tp + fn + eps)
            spec = (tn + eps) / (tn + fp + eps)
            precision = (tp + eps) / (tp + fp + eps)

            dice_list.append(dice)
            iou_list.append(iou)
            sens_list.append(sens)
            spec_list.append(spec)
            precision_list.append(precision)

            name = self.class_names[c]
            metrics[f"{name}_Dice"] = float(dice)
            metrics[f"{name}_IoU"] = float(iou)
            metrics[f"{name}_Sensitivity"] = float(sens)
            metrics[f"{name}_Specificity"] = float(spec)
            metrics[f"{name}_Precision"] = float(precision)

        metrics["Mean_Vessel_Dice"] = float(np.mean(dice_list[1:]))
        metrics["Mean_Vessel_IoU"] = float(np.mean(iou_list[1:]))
        metrics["Mean_Vessel_Sensitivity"] = float(np.mean(sens_list[1:]))
        metrics["Mean_Vessel_Specificity"] = float(np.mean(spec_list[1:]))
        metrics["Mean_Vessel_Precision"] = float(np.mean(precision_list[1:]))

        if self.per_image_dice:
            for cls in range(1, self.num_classes):
                name = self.class_names[cls]
                per_img = [d[name] for d in self.per_image_dice]
                metrics[f"{name}_Dice_per_image"] = float(np.mean(per_img))

            vessel_names = self.class_names[1:]
            per_img_vessel = [
                np.mean([d[n] for n in vessel_names]) for d in self.per_image_dice
            ]
            metrics["Mean_Vessel_Dice_per_image"] = float(np.mean(per_img_vessel))

        return metrics

    def summary(self) -> str:
        m = self.compute()
        lines = []

        for c in range(self.num_classes):
            name = self.class_names[c]
            lines.append(f"  {name}:")
            lines.append(
                f"    Dice={m[f'{name}_Dice']:.4f}  "
                f"IoU={m[f'{name}_IoU']:.4f}  "
                f"Sens={m[f'{name}_Sensitivity']:.4f}  "
                f"Spec={m[f'{name}_Specificity']:.4f}  "
                f"Precision={m[f'{name}_Precision']:.4f}"
            )

        lines.append("  ──────────────────────────")
        lines.append(
            f"  Mean Vessel Dice: {m['Mean_Vessel_Dice']:.4f}  "
            f"IoU: {m['Mean_Vessel_IoU']:.4f}"
        )
        return "\n".join(lines)
