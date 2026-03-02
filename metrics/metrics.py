import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int = 3, class_names: list = None):
        self.num_classes = num_classes
        self.class_names = class_names or ["background", "artery", "vein"]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        preds_np = predictions.cpu().numpy().flatten().astype(int)
        targets_np = targets.cpu().numpy().flatten().astype(int)

        mask = (targets_np >= 0) & (targets_np < self.num_classes) & \
               (preds_np >= 0) & (preds_np < self.num_classes)

        np.add.at(
            self.confusion_matrix,
            (targets_np[mask], preds_np[mask]),
            1
        )

    def compute(self) -> dict:
        eps = 1e-7
        cm = self.confusion_matrix
        metrics = {}

        dice_list = []
        iou_list = []
        sensitivity_list = []
        specificity_list = []
        precision_list = []

        total = cm.sum()

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            tn = total - tp - fp - fn

            dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)  # F1
            iou = (tp + eps) / (tp + fp + fn + eps)
            sensitivity = (tp + eps) / (tp + fn + eps)  # Recall
            specificity = (tn + eps) / (tn + fp + eps)
            precision = (tp + eps) / (tp + fp + eps)

            dice_list.append(dice)
            iou_list.append(iou)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
            precision_list.append(precision)

            name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            metrics[f"{name}_Dice"] = float(dice)
            metrics[f"{name}_IoU"] = float(iou)
            metrics[f"{name}_Sensitivity"] = float(sensitivity)
            metrics[f"{name}_Specificity"] = float(specificity)
            metrics[f"{name}_Precision"] = float(precision)

        metrics["avg_f1_artery_vein"] = float(np.mean(dice_list[1:]))
        metrics["Mean_Vessel_Dice"] = float(np.mean(dice_list[1:]))
        metrics["Mean_Vessel_IoU"] = float(np.mean(iou_list[1:]))
        metrics["Mean_Vessel_Sensitivity"] = float(np.mean(sensitivity_list[1:]))
        metrics["Mean_Vessel_Specificity"] = float(np.mean(specificity_list[1:]))
        metrics["Mean_Vessel_Precision"] = float(np.mean(precision_list[1:]))

        metrics["Mean_All_Dice"] = float(np.mean(dice_list))
        metrics["Mean_All_IoU"] = float(np.mean(iou_list))

        return metrics
