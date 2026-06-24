from __future__ import annotations

import os

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config

try:
    from skimage.morphology import skeletonize

    _HAS_SKIMAGE = True
except ImportError:
    skeletonize = None
    _HAS_SKIMAGE = False

COLOR_MAP = {
    0: [0, 0, 0],  # background — black
    1: [255, 60, 60],  # artery — red
    2: [60, 120, 255],  # vein — blue
}

_ERROR_COLORS = {
    (1, 0): [255, 200, 0],
    (2, 0): [255, 140, 0],
    (0, 1): [0, 220, 80],
    (0, 2): [0, 180, 180],
    (1, 2): [200, 0, 255],
    (2, 1): [255, 0, 180],
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_SKEL_ALPHA = 0.75
_OVERLAY_ALPHA = 0.5


def class_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    lut = np.zeros((max(COLOR_MAP.keys()) + 1, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        lut[cls] = color
    flat = mask.ravel().clip(0, len(lut) - 1)
    return lut[flat].reshape(h, w, 3)


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor[:3].cpu().numpy()
    img = img * _IMAGENET_STD[:, None, None] + _IMAGENET_MEAN[:, None, None]
    return np.clip(img, 0.0, 1.0).transpose(1, 2, 0)


def make_overlay(
    image: np.ndarray, mask: np.ndarray, alpha: float = _OVERLAY_ALPHA
) -> np.ndarray:
    mask_rgb = class_mask_to_rgb(mask).astype(np.float32) / 255.0
    vessel_pixels = (mask > 0)[..., None]
    blended = image.copy()
    blended[vessel_pixels[:, :, 0]] = (1 - alpha) * image[
        vessel_pixels[:, :, 0]
    ] + alpha * mask_rgb[vessel_pixels[:, :, 0]]
    return np.clip(blended, 0.0, 1.0)


def make_error_map(
    gt: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    h, w = gt.shape
    error_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    stats: dict[str, int] = {
        "missed_artery": 0,
        "missed_vein": 0,
        "hallucinated_artery": 0,
        "hallucinated_vein": 0,
        "artery_as_vein": 0,
        "vein_as_artery": 0,
        "correct_artery": 0,
        "correct_vein": 0,
        "correct_background": 0,
    }

    label_map = {
        (1, 0): ("missed_artery", _ERROR_COLORS[(1, 0)]),
        (2, 0): ("missed_vein", _ERROR_COLORS[(2, 0)]),
        (0, 1): ("hallucinated_artery", _ERROR_COLORS[(0, 1)]),
        (0, 2): ("hallucinated_vein", _ERROR_COLORS[(0, 2)]),
        (1, 2): ("artery_as_vein", _ERROR_COLORS[(1, 2)]),
        (2, 1): ("vein_as_artery", _ERROR_COLORS[(2, 1)]),
    }
    correct_map = {
        (0, 0): "correct_background",
        (1, 1): "correct_artery",
        (2, 2): "correct_vein",
    }

    for (g, p), (stat_key, color) in label_map.items():
        mask = (gt == g) & (pred == p)
        error_rgb[mask] = color
        stats[stat_key] = int(mask.sum())

    for (g, p), stat_key in correct_map.items():
        stats[stat_key] = int(((gt == g) & (pred == p)).sum())

    return error_rgb, stats


def make_skeleton_overlay(
    image: np.ndarray,
    skeleton: np.ndarray,
    alpha: float = _SKEL_ALPHA,
) -> np.ndarray:
    skel_colors = {
        1: [255, 160, 160],
        2: [160, 200, 255],
    }
    overlay = image.copy()
    for cls, color in skel_colors.items():
        px = skeleton == cls
        if not px.any():
            continue
        c = np.array(color, dtype=np.float32) / 255.0
        overlay[px] = (1 - alpha) * image[px] + alpha * c
    return np.clip(overlay, 0.0, 1.0)


def _vessel_legend(ax: plt.Axes) -> None:
    patches = [
        mpatches.Patch(color=tuple(np.array(COLOR_MAP[1]) / 255.0), label="Artery"),
        mpatches.Patch(color=tuple(np.array(COLOR_MAP[2]) / 255.0), label="Vein"),
    ]
    ax.legend(
        handles=patches, loc="lower right", fontsize=7, framealpha=0.7, markerscale=0.8
    )


def _error_legend(ax: plt.Axes, stats: dict[str, int]) -> None:
    entries = [
        ("missed_artery", _ERROR_COLORS[(1, 0)], "FN artery"),
        ("missed_vein", _ERROR_COLORS[(2, 0)], "FN vein"),
        ("hallucinated_artery", _ERROR_COLORS[(0, 1)], "FP artery"),
        ("hallucinated_vein", _ERROR_COLORS[(0, 2)], "FP vein"),
        ("artery_as_vein", _ERROR_COLORS[(1, 2)], "A→V confusion"),
        ("vein_as_artery", _ERROR_COLORS[(2, 1)], "V→A confusion"),
    ]
    patches = [
        mpatches.Patch(
            color=np.array(c) / 255.0,
            label=f"{lbl}: {stats.get(k, 0):,}px",
        )
        for k, c, lbl in entries
        if stats.get(k, 0) > 0
    ]
    if patches:
        ax.legend(
            handles=patches,
            loc="lower right",
            fontsize=6.5,
            framealpha=0.75,
            markerscale=0.8,
        )


def _mask_skeleton(mask: np.ndarray) -> np.ndarray:
    if not _HAS_SKIMAGE:
        return np.zeros_like(mask)
    skel = np.zeros_like(mask, dtype=np.int32)
    for cls in (1, 2):
        binary = mask == cls
        if binary.any():
            s = skeletonize(binary)
            skel[s] = cls
    return skel


def _predict_batch(
    model,
    images: torch.Tensor,
    device: str,
) -> torch.Tensor:
    from torch.amp import autocast

    from training import get_amp_dtype, needs_sw, sw_inference

    amp_dtype = get_amp_dtype()
    use_amp = Config.USE_AMP and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"

    use_sw = needs_sw() and (
        images.shape[2] > Config.IMG_SIZE or images.shape[3] > Config.IMG_SIZE
    )
    if use_sw:
        preds_list: list[torch.Tensor] = []
        for b in range(images.size(0)):
            base, refinement = sw_inference(
                model=model,
                image=images[b : b + 1],
                tile_size=Config.IMG_SIZE,
                overlap=Config.IMG_SIZE // 4,
                num_classes=Config.NUM_CLASSES,
            )
            if refinement:
                preds_list.append(refinement[-1].argmax(dim=1))
            else:
                preds_list.append(base.argmax(dim=1))
        return torch.cat(preds_list, dim=0)

    with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
        outputs = model(images)

    refinement = outputs.get("refinement")
    if refinement and isinstance(refinement, list) and len(refinement) > 0:
        return refinement[-1].argmax(dim=1)
    return outputs["segmentation"].argmax(dim=1)


@torch.no_grad()
def visualize_predictions(
    model,
    loader,
    device: str,
    output_dir: str,
    epoch: int,
    num_samples: int = 3,
) -> None:
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    for batch in loader:
        if saved >= num_samples:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        filenames = batch["filename"]

        preds = _predict_batch(model, images, device).cpu()

        for i in range(images.size(0)):
            if saved >= num_samples:
                break

            img_np = denormalize(images[i])
            gt_np = masks[i].numpy()
            pred_np = preds[i].numpy()
            skel_np = _mask_skeleton(gt_np)
            pred_skel_np = _mask_skeleton(pred_np)

            error_rgb, stats = make_error_map(gt_np, pred_np)

            fig = plt.figure(figsize=(22, 11))
            gs = GridSpec(2, 4, figure=fig, hspace=0.08, wspace=0.04)

            panels = [
                (0, 0, img_np, "Input image", "gray", False),
                (0, 1, class_mask_to_rgb(gt_np), "Ground truth", None, True),
                (0, 2, class_mask_to_rgb(pred_np), "Prediction", None, True),
                (0, 3, make_overlay(img_np, pred_np), "Pred overlay", None, True),
                (
                    1,
                    0,
                    make_skeleton_overlay(img_np, skel_np),
                    "GT skeleton",
                    None,
                    False,
                ),
                (
                    1,
                    1,
                    make_skeleton_overlay(img_np, pred_skel_np),
                    "Pred skeleton",
                    None,
                    False,
                ),
                (1, 2, error_rgb, "Error map", None, False),
                (1, 3, make_overlay(img_np, pred_np), "Error overlay", None, True),
            ]

            ax_objs: list[plt.Axes] = []
            for row, col, data, title, cmap, add_legend in panels:
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(data, cmap=cmap, interpolation="nearest")
                ax.set_title(title, fontsize=10, pad=4)
                ax.axis("off")
                if add_legend:
                    _vessel_legend(ax)
                ax_objs.append(ax)

            ax_err = ax_objs[6]
            _error_legend(ax_err, stats)

            ax_err_ov = ax_objs[7]
            ax_err_ov.cla()
            ax_err_ov.imshow(img_np, interpolation="nearest")
            err_rgba = np.zeros((*error_rgb.shape[:2], 4), dtype=np.float32)
            err_rgba[..., :3] = error_rgb.astype(np.float32) / 255.0
            err_rgba[..., 3] = (error_rgb.sum(axis=-1) > 0).astype(np.float32) * 0.75
            ax_err_ov.imshow(err_rgba, interpolation="nearest")
            ax_err_ov.set_title("Error overlay", fontsize=10, pad=4)
            ax_err_ov.axis("off")
            _error_legend(ax_err_ov, stats)

            total_vessel = int((gt_np > 0).sum())
            correct_fg = stats["correct_artery"] + stats["correct_vein"]
            recall_str = (
                f"{correct_fg / total_vessel * 100:.1f}%" if total_vessel > 0 else "n/a"
            )
            summary = (
                f"FN artery {stats['missed_artery']:,}  |  "
                f"FN vein {stats['missed_vein']:,}  |  "
                f"FP artery {stats['hallucinated_artery']:,}  |  "
                f"FP vein {stats['hallucinated_vein']:,}  |  "
                f"A ↔ V confusion {stats['artery_as_vein'] + stats['vein_as_artery']:,}  |  "
                f"Vessel recall {recall_str}"
            )
            fig.suptitle(
                f"Epoch {epoch:03d}  |  {filenames[i]}\n{summary}",
                fontsize=10,
                y=0.995,
            )

            out_path = os.path.join(
                output_dir,
                f"epoch{epoch:03d}_{saved:02d}_{filenames[i]}",
            )
            fig.savefig(out_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            saved += 1


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_dices: list[float],
    output_dir: str,
    val_f1_artery: list[float] | None = None,
    val_f1_vein: list[float] | None = None,
    train_loss_details: list[dict[str, float]] | None = None,
    val_loss_details: list[dict[str, float]] | None = None,
    vessel_pos_weight: list[float] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    has_perclass = val_f1_artery is not None and val_f1_vein is not None
    has_breakdown = train_loss_details is not None and len(train_loss_details) > 0
    has_posweight = vessel_pos_weight is not None and len(vessel_pos_weight) > 0

    n_plots = 2 + int(has_breakdown) + int(has_posweight)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    ax = axes[ax_idx]
    ax_idx += 1
    ax.plot(epochs, train_losses, "b-", label="Train", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[ax_idx]
    ax_idx += 1
    ax.plot(epochs, val_dices, "g-", label="Mean vessel Dice", linewidth=2)
    if has_perclass:
        ax.plot(
            epochs,
            val_f1_artery,
            color="#e84040",
            linestyle="--",
            label="F1 artery",
            linewidth=1.5,
        )
        ax.plot(
            epochs,
            val_f1_vein,
            color="#4080e8",
            linestyle="--",
            label="F1 vein",
            linewidth=1.5,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Dice / F1")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if has_breakdown:
        ax = axes[ax_idx]
        ax_idx += 1

        all_keys: list[str] = []
        for d in (train_loss_details or []) + (val_loss_details or []):
            for k in d:
                if k != "total" and k not in all_keys:
                    all_keys.append(k)

        colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(all_keys)))

        for key, color in zip(all_keys, colors):
            if train_loss_details:
                vals = [d.get(key, 0.0) for d in train_loss_details]
                ax.plot(
                    epochs, vals, "-", color=color, linewidth=1.8, label=f"train/{key}"
                )
            if val_loss_details:
                vals = [d.get(key, 0.0) for d in val_loss_details]
                ax.plot(
                    epochs, vals, "--", color=color, linewidth=1.4, label=f"val/{key}"
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss component")
        ax.set_title("Loss breakdown")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    if has_posweight:
        ax = axes[ax_idx]
        ax_idx += 1
        ax.plot(
            epochs[: len(vessel_pos_weight)],
            vessel_pos_weight,
            color="#e07020",
            linewidth=2,
            label="vessel pos_weight (EMA)",
        )
        ax.axhline(
            y=np.mean(vessel_pos_weight),
            color="gray",
            linestyle=":",
            linewidth=1,
            label=f"mean={np.mean(vessel_pos_weight):.1f}",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("pos_weight")
        ax.set_title("Dynamic vessel BCE pos_weight")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
