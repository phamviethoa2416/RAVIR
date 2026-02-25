import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_MAP = {
    0: [0,   0,   0  ],
    1: [255, 0,   0  ],
    2: [0,   0,   255],
}

def _class_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color
    return rgb

def visualize_predictions(
    model,
    loader,
    device,
    output_dir : str,
    epoch      : int,
    num_samples: int = 3,
):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for images, masks, filenames in loader:
            if saved >= num_samples:
                break

            images = images.to(device)
            output = model(images)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            preds = torch.argmax(logits, dim=1)

            for i in range(images.size(0)):
                if saved >= num_samples:
                    break

                img = images[i, 0].cpu().numpy()
                gt = masks[i].cpu().numpy()
                pred = preds[i].cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img, cmap="gray")
                axes[0].set_title("Input Image")
                axes[1].imshow(_class_mask_to_rgb(gt))
                axes[1].set_title("Ground Truth")
                axes[2].imshow(_class_mask_to_rgb(pred))
                axes[2].set_title("Prediction")

                for ax in axes:
                    ax.axis("off")

                plt.suptitle(f"Epoch {epoch} | {filenames[i]}", fontsize=14)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        output_dir,
                        f"epoch{epoch:03d}_{saved:02d}_{filenames[i]}"
                    ),
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.close(fig)
                saved += 1

def plot_training_curves(
    train_losses : list,
    val_losses   : list,
    val_dices    : list,
    output_dir   : str,
    val_f1_artery: list = None,
    val_f1_vein  : list = None,
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