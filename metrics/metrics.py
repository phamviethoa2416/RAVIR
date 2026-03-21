from __future__ import annotations

import numpy as np
import torch
from skimage.morphology import skeletonize


def _cl_score(pred: np.ndarray, target: np.ndarray) -> float:
    skel_target = skeletonize(target.astype(bool))
    if skel_target.sum() == 0:
        return 1.0
    return float((skel_target & pred.astype(bool)).sum() / skel_target.sum())


def compute_cldice(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    t_prec = _cl_score(target, pred)
    t_sens = _cl_score(pred, target)

    if t_prec + t_sens == 0:
        return 0.0
    return float(2.0 * t_prec * t_sens / (t_prec + t_sens))


class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int = 3,
        class_names: list[str] | None = None,
        track_per_image: bool = False,
        compute_cldice: bool = True,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or ["background", "artery", "vein"]
        self.track_per_image = track_per_image
        self.compute_cldice_flag = compute_cldice

        self.confusion_matrix = np.zeros(
            (num_classes, num_classes),
            dtype=np.int64,
        )
        self._per_image_dice: list[dict[str, float]] = []

        self._cldice_preds: dict[int, list[np.ndarray]] = {
            c: [] for c in range(1, num_classes)
        }
        self._cldice_targets: dict[int, list[np.ndarray]] = {
            c: [] for c in range(1, num_classes)
        }

    def reset(self) -> None:
        self.confusion_matrix.fill(0)
        self._per_image_dice.clear()
        for c in self._cldice_preds:
            self._cldice_preds[c].clear()
            self._cldice_targets[c].clear()

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

            if self.track_per_image:
                img_dice = {}
                for c in range(self.num_classes):
                    tp = ((p_valid == c) & (t_valid == c)).sum()
                    fp = ((p_valid == c) & (t_valid != c)).sum()
                    fn = ((p_valid != c) & (t_valid == c)).sum()
                    dice = (
                        (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
                    )
                    img_dice[self.class_names[c]] = float(dice)
                self._per_image_dice.append(img_dice)

            if self.compute_cldice_flag:
                for c in range(1, self.num_classes):
                    self._cldice_preds[c].append(
                        (preds_np[b] == c).astype(np.uint8),
                    )
                    self._cldice_targets[c].append(
                        (targets_np[b] == c).astype(np.uint8),
                    )

    def compute(self) -> dict[str, float]:
        eps = 1e-7
        cm = self.confusion_matrix
        total = cm.sum()
        metrics: dict[str, float] = {}

        dice_list: list[float] = []
        iou_list: list[float] = []
        sens_list: list[float] = []
        spec_list: list[float] = []
        prec_list: list[float] = []

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            tn = total - tp - fp - fn

            dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            iou = (tp + eps) / (tp + fp + fn + eps)
            sens = (tp + eps) / (tp + fn + eps)
            spec = (tn + eps) / (tn + fp + eps)
            prec = (tp + eps) / (tp + fp + eps)

            dice_list.append(dice)
            iou_list.append(iou)
            sens_list.append(sens)
            spec_list.append(spec)
            prec_list.append(prec)

            name = self.class_names[c]
            metrics[f"{name}_Dice"] = float(dice)
            metrics[f"{name}_IoU"] = float(iou)
            metrics[f"{name}_Sensitivity"] = float(sens)
            metrics[f"{name}_Specificity"] = float(spec)
            metrics[f"{name}_Precision"] = float(prec)

        metrics["Mean_Vessel_Dice"] = float(np.mean(dice_list[1:]))
        metrics["Mean_Vessel_IoU"] = float(np.mean(iou_list[1:]))
        metrics["Mean_Vessel_Sensitivity"] = float(np.mean(sens_list[1:]))
        metrics["Mean_Vessel_Specificity"] = float(np.mean(spec_list[1:]))
        metrics["Mean_Vessel_Precision"] = float(np.mean(prec_list[1:]))

        metrics["Mean_All_Dice"] = float(np.mean(dice_list))
        metrics["Mean_All_IoU"] = float(np.mean(iou_list))

        if self.compute_cldice_flag and self._cldice_preds[1]:
            cldice_per_class: list[float] = []

            for c in range(1, self.num_classes):
                scores = [
                    compute_cldice(p, t)
                    for p, t in zip(
                        self._cldice_preds[c],
                        self._cldice_targets[c],
                    )
                ]
                mean_cldice = float(np.mean(scores))
                name = self.class_names[c]
                metrics[f"{name}_clDice"] = mean_cldice
                cldice_per_class.append(mean_cldice)

            metrics["Mean_Vessel_clDice"] = float(np.mean(cldice_per_class))

        return metrics

    @property
    def per_image_dice(self) -> list[dict[str, float]]:
        return self._per_image_dice

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
                f"Prec={m[f'{name}_Precision']:.4f}"
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
