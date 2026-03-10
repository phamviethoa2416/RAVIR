import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config

COLOR_MAP = {
    0: [0, 0, 0],       # background
    1: [255, 0, 0],      # artery
    2: [0, 0, 255],      # vein
}


def _class_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color
    return rgb


def _denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    return img_tensor.cpu().numpy() * 0.5 + 0.5


@torch.no_grad()
def visualize_predictions(
        model,
        loader,
        device: str,
        output_dir: str,
        epoch: int,
        num_samples: int = 3,
):
    from training import get_amp_dtype
    from torch.amp import autocast

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    amp_dtype = get_amp_dtype()
    use_amp = Config.USE_AMP and device == "cuda"

    saved = 0
    for batch in loader:
        if saved >= num_samples:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        filenames = batch["filename"]

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
        preds = torch.argmax(outputs["segmentation"], dim=1).cpu()

        for i in range(images.size(0)):
            if saved >= num_samples:
                break

            img_np = _denormalize(images[i, 0])
            gt_np = masks[i].numpy()
            pred_np = preds[i].numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("Input Image")

            axes[1].imshow(_class_mask_to_rgb(gt_np))
            axes[1].set_title("Ground Truth")

            axes[2].imshow(_class_mask_to_rgb(pred_np))
            axes[2].set_title("Prediction")

            for ax in axes:
                ax.axis("off")

            plt.suptitle(f"Epoch {epoch} | {filenames[i]}", fontsize=14)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"epoch{epoch:03d}_{saved:02d}_{filenames[i]}"),
                dpi=100,
                bbox_inches="tight",
            )
            plt.close(fig)
            saved += 1


@torch.no_grad()
def visualize_binary_predictions(
        model,
        loader,
        device: str,
        output_dir: str,
        epoch: int,
        num_samples: int = 3,
        threshold: float = 0.5,
):
    from training import get_amp_dtype
    from torch.amp import autocast

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    amp_dtype = get_amp_dtype()
    use_amp = Config.USE_AMP and device == "cuda"

    saved = 0
    for batch in loader:
        if saved >= num_samples:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        filenames = batch["filename"]

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)

        logits = outputs.get("vessel_prob", None) if isinstance(outputs, dict) else None
        if logits is None:
            continue

        probs = torch.sigmoid(logits).detach().cpu()
        preds = (probs >= threshold).float()

        for i in range(images.size(0)):
            if saved >= num_samples:
                break

            # Support both 1ch and 3ch inputs
            img = images[i].detach().cpu()
            if img.size(0) == 1:
                img_np = _denormalize(img[0])
                img_show = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                img_show = np.stack([img_show] * 3, axis=-1)
            else:
                img_np = img.permute(1, 2, 0).numpy()
                img_show = (img_np * 255.0).clip(0, 255).astype(np.uint8)

            gt = masks[i]
            if isinstance(gt, torch.Tensor):
                gt_np = gt.squeeze().cpu().numpy()
            else:
                gt_np = np.asarray(gt).squeeze()

            prob_np = probs[i, 0].numpy()
            pred_np = preds[i, 0].numpy()

            fig, axes = plt.subplots(1, 4, figsize=(18, 5))
            axes[0].imshow(img_show)
            axes[0].set_title("Input")

            axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("GT (binary)")

            axes[2].imshow(prob_np, cmap="magma", vmin=0, vmax=1)
            axes[2].set_title("Prob (sigmoid)")

            axes[3].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[3].set_title(f"Pred (thr={threshold:.2f})")

            for ax in axes:
                ax.axis("off")

            plt.suptitle(f"Epoch {epoch} | {filenames[i]}", fontsize=14)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"epoch{epoch:03d}_bin_{saved:02d}_{filenames[i]}"),
                dpi=120,
                bbox_inches="tight",
            )
            plt.close(fig)
            saved += 1


def plot_training_curves(
        train_losses: list,
        val_losses: list,
        val_dices: list,
        output_dir: str,
        val_f1_artery: list | None = None,
        val_f1_vein: list | None = None,
):
    epochs = range(1, len(train_losses) + 1)
    has_perclass = val_f1_artery is not None and val_f1_vein is not None
    n_plots = 3 if has_perclass else 2

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))

    axes[0].plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_dices, "g-", label="Mean Vessel Dice", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Validation Mean Vessel Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if has_perclass:
        axes[2].plot(epochs, val_f1_artery, "r-", label="F1 Artery", linewidth=2)
        axes[2].plot(epochs, val_f1_vein, "b-", label="F1 Vein", linewidth=2)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1 Score")
        axes[2].set_title("Per-class F1 — Artery & Vein")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "training_curves.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
