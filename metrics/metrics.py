from __future__ import annotations

import numpy as np
import torch
from skimage.morphology import skeletonize


def cl_score(pred: np.ndarray, target: np.ndarray) -> float:
    skel = skeletonize(target.astype(bool))
    if skel.sum() == 0:
        return 1.0
    return float((skel & pred.astype(bool)).sum() / skel.sum())


def compute_cldice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0

    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    t_prec = cl_score(target, pred)
    t_sens = cl_score(pred, target)

    if t_prec + t_sens == 0:
        return 0.0

    return float(2.0 * t_prec * t_sens / (t_prec + t_sens))


class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int = 3,
        class_names: list[str] | None = None,
        evaluate_cldice: bool = True,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [
            "background",
            "artery",
            "vein",
        ]

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class names, "
                f"got {len(self.class_names)}"
            )

        self.evaluate_cldice = evaluate_cldice
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes),
            dtype=np.int64,
        )
        self.per_image_dice: list[dict[str, float]] = []

        self.cldice_scores: dict[int, list[float]] = {
            cls: [] for cls in range(1, num_classes)
        }

    def reset(self) -> None:
        self.confusion_matrix.fill(0)
        self.per_image_dice.clear()
        for c in self.cldice_scores:
            self.cldice_scores[c].clear()

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        preds_np = predictions.detach().cpu().numpy().astype(np.int64)
        targets_np = targets.detach().cpu().numpy().astype(np.int64)

        if preds_np.shape != targets_np.shape:
            raise ValueError(f"Predictions and targets must have the same shape. ")

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

            img_dice: dict[str, float] = {}
            for cls in range(self.num_classes):
                tp = ((p_valid == cls) & (t_valid == cls)).sum()
                fp = ((p_valid == cls) & (t_valid != cls)).sum()
                fn = ((p_valid != cls) & (t_valid == cls)).sum()
                dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
                img_dice[self.class_names[cls]] = float(dice)
            self.per_image_dice.append(img_dice)

            if self.evaluate_cldice:
                for cls in range(1, self.num_classes):
                    pred_mask = preds_np[b] == cls
                    target_mask = targets_np[b] == cls

                    score = compute_cldice_score(pred_mask, target_mask)
                    self.cldice_scores[cls].append(score)

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
            metrics[f"{name}_dice"] = float(dice)
            metrics[f"{name}_iou"] = float(iou)
            metrics[f"{name}_sensitivity"] = float(sens)
            metrics[f"{name}_specificity"] = float(spec)
            metrics[f"{name}_precision"] = float(precision)

        metrics["Mean_Vessel_Dice"] = float(np.mean(dice_list[1:]))
        metrics["Mean_Vessel_IoU"] = float(np.mean(iou_list[1:]))
        metrics["Mean_Vessel_Sensitivity"] = float(np.mean(sens_list[1:]))
        metrics["Mean_Vessel_Specificity"] = float(np.mean(spec_list[1:]))
        metrics["Mean_Vessel_Precision"] = float(np.mean(precision_list[1:]))

        if self.evaluate_cldice:
            cldice_per_class: list[float] = []

            for cls in range(1, self.num_classes):
                name = self.class_names[cls]
                scores = self.cldice_scores[cls]

                if scores:
                    mean_cldice = float(np.mean(scores))
                else:
                    mean_cldice = 0.0

                metrics[f"{name}_clDice"] = mean_cldice
                cldice_per_class.append(mean_cldice)

            if cldice_per_class:
                metrics["Mean_Vessel_clDice"] = float(np.mean(cldice_per_class))

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
                f"    Dice={m[f'{name}_dice']:.4f}  "
                f"IoU={m[f'{name}_iou']:.4f}  "
                f"Sens={m[f'{name}_sensitivity']:.4f}  "
                f"Spec={m[f'{name}_specificity']:.4f}  "
                f"Precision={m[f'{name}_precision']:.4f}"
            )

            if f"{name}_clDice" in m:
                lines.append(f"    clDice={m[f'{name}_clDice']:.4f}")

        lines.append("  ──────────────────────────")
        lines.append(
            f"  Mean Vessel Dice: {m['Mean_Vessel_Dice']:.4f}  "
            f"IoU: {m['Mean_Vessel_IoU']:.4f}"
        )
        if "Mean_Vessel_clDice" in m:
            lines.append(f"  Mean Vessel clDice: {m['Mean_Vessel_clDice']:.4f}")

        return "\n".join(lines)
