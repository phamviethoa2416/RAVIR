from __future__ import annotations

import argparse
import logging
import os
import time

import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from curriculum.dataset import build_datasets
from curriculum.trainer import train_one_epoch, validate
from losses import BinaryDiceBCELoss
from metrics import BinarySegmentationMetrics
from models import RAVIRNet
from training import get_amp_dtype
from transform import (
    get_round1_train_transform,
    get_round1_val_transform,
    get_round2_train_transform,
    get_round2_val_transform,
)
from transform.ravir import RAVIRDataset
from utils import seed_everything, setup_logging, plot_training_curves

logger = logging.getLogger(__name__)

DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

def _create_scaler() -> GradScaler | None:
    if Config.USE_AMP and Config.AMP_DTYPE == "float16":
        return GradScaler("cuda")
    return None


def _amp_settings() -> tuple[bool, torch.dtype]:
    use = Config.USE_AMP and Config.DEVICE == "cuda"
    dtype = get_amp_dtype()
    return use, dtype


def adapt_state_dict_channels(
        state_dict: dict,
        src_channels: int,
        tgt_channels: int,
) -> dict:
    if src_channels == tgt_channels:
        return state_dict

    new_sd = {}
    adapted = False
    for key, value in state_dict.items():
        if (
                not adapted
                and value.ndim == 4
                and value.shape[1] == src_channels
                and "encoder" in key
        ):
            if tgt_channels < src_channels:
                new_sd[key] = value.mean(dim=1, keepdim=True).repeat(1, tgt_channels, 1, 1)
            else:
                repeats = tgt_channels // src_channels + 1
                new_sd[key] = value.repeat(1, repeats, 1, 1)[:, :tgt_channels]
            adapted = True
            continue
        new_sd[key] = value
    return new_sd


# ── RAVIR binary wrapper ─────────────────────────────────────────────────────


class RAVIRBinaryDataset(torch.utils.data.Dataset):
    """Thin wrapper around RAVIRDataset that returns a binary vessel mask."""

    def __init__(self, ravir_dataset: RAVIRDataset):
        self.ds = ravir_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        sample = self.ds[idx]
        return {
            "image": sample["image"],
            "mask": sample["vessel_prob"],
            "filename": sample["filename"],
        }


# ═════════════════════════════════════════════════════════════════════════════
#  Round 1 — Vessel Discovery
# ═════════════════════════════════════════════════════════════════════════════


def train_round1(args):
    seed_everything(Config.SEED)

    run_dir = os.path.join(args.output_dir, "round1")
    os.makedirs(run_dir, exist_ok=True)
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    log = setup_logging(run_dir, name="CurriculumR1")
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

    device = Config.DEVICE
    if device == "cuda":
        torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
        if Config.ALLOW_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ── Data ──────────────────────────────────────────────────────────
    patch_size = args.patch_size

    train_dataset, val_dataset = build_datasets(
        data_root=DATASET_ROOT,
        train_transform=get_round1_train_transform(patch_size),
        val_transform=get_round1_val_transform(patch_size),
        val_ratio=0.2,
        seed=Config.SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
        persistent_workers=Config.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.NUM_WORKERS > 0,
    )

    log.info("=" * 60)
    log.info("  Round 1 — Vessel Discovery (from scratch)")
    log.info("=" * 60)
    log.info("  Datasets  : DRIVE + STARE + CHASE_DB1")
    log.info("  Train size: %d", len(train_dataset))
    log.info("  Val size  : %d", len(val_dataset))
    log.info("  Patch size: %d×%d", patch_size, patch_size)
    log.info("  Batch size: %d", args.batch_size)
    log.info("  Epochs    : %d", args.epochs)
    log.info("  LR        : %.1e", args.lr)
    log.info("  Device    : %s", device)

    # ── Model ─────────────────────────────────────────────────────────
    model = RAVIRNet(
        in_channels=3,
        num_classes=1,
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,
        dropout_rate=Config.DROPOUT_RATE,
        use_attention=Config.USE_ATTENTION,
        binary_mode=True,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Params    : %s total / %s trainable", f"{total_p:,}", f"{train_p:,}")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────
    criterion = BinaryDiceBCELoss(dice_weight=0.5, bce_weight=0.5).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = _create_scaler()
    use_amp, amp_dtype = _amp_settings()

    metrics_calc = BinarySegmentationMetrics()

    # ── Training loop ─────────────────────────────────────────────────
    best_dice = 0.0
    patience = 0
    train_losses, val_losses, val_dices = [], [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=args.grad_accum,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        val_loss, metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            metrics_calculator=metrics_calc,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        scheduler.step()
        dt = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(metrics["Dice"])

        log.info(
            "Epoch %3d/%d | Train %.4f | Val %.4f | "
            "Dice=%.4f  IoU=%.4f  Sens=%.4f  Spec=%.4f | "
            "LR=%.2e | %.0fs",
            epoch + 1, args.epochs, train_loss, val_loss,
            metrics["Dice"], metrics["IoU"],
            metrics["Sensitivity"], metrics["Specificity"],
            lr, dt,
        )

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Metrics/Dice", metrics["Dice"], epoch + 1)
        writer.add_scalar("Metrics/IoU", metrics["IoU"], epoch + 1)
        writer.add_scalar("LR", lr, epoch + 1)

        if metrics["Dice"] > best_dice:
            best_dice = metrics["Dice"]
            patience = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "metrics": metrics,
                "round": 1,
                "in_channels": 3,
                "config": {
                    "encoder_name": Config.ENCODER_NAME,
                    "dropout_rate": Config.DROPOUT_RATE,
                    "use_attention": Config.USE_ATTENTION,
                    "patch_size": patch_size,
                },
            }, os.path.join(run_dir, "round1_best.pth"))
            log.info("  ★ New best! Dice=%.4f  → round1_best.pth", best_dice)
        else:
            patience += 1
            log.info("  No improvement. Patience: %d/%d", patience, args.patience)

        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "round": 1,
                "in_channels": 3,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_dices": val_dices,
            }, os.path.join(run_dir, f"checkpoint_epoch{epoch + 1:03d}.pth"))

        if len(train_losses) > 1:
            try:
                plot_training_curves(train_losses, val_losses, val_dices, run_dir)
            except Exception:
                pass

        if patience >= args.patience:
            log.info("Early stopping at epoch %d", epoch + 1)
            break

    writer.close()
    log.info("=" * 60)
    log.info("  Round 1 Complete — Best Dice: %.4f", best_dice)
    log.info("  Checkpoint: %s", os.path.join(run_dir, "round1_best.pth"))
    log.info("=" * 60)
    return best_dice

def train_round2(args):
    seed_everything(Config.SEED)

    run_dir = os.path.join(args.output_dir, "round2")
    os.makedirs(run_dir, exist_ok=True)
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    log = setup_logging(run_dir, name="CurriculumR2")
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

    device = Config.DEVICE
    if device == "cuda":
        torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
        if Config.ALLOW_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ── Data (RAVIR, binary task) ─────────────────────────────────────
    patch_size = args.patch_size
    ravir_img_dir = os.path.join(DATASET_ROOT, "RAVIR", "train", "training_images")
    ravir_mask_dir = os.path.join(DATASET_ROOT, "RAVIR", "train", "training_masks")

    all_files = sorted(
        f for f in os.listdir(ravir_img_dir) if f.endswith(".png")
    )

    kf = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    folds = list(kf.split(all_files))
    train_idx, val_idx = folds[Config.VAL_FOLD]
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]

    ravir_train = RAVIRDataset(
        img_dir=ravir_img_dir,
        mask_dir=ravir_mask_dir,
        file_list=train_files,
        transform=get_round2_train_transform(patch_size),
    )
    ravir_val = RAVIRDataset(
        img_dir=ravir_img_dir,
        mask_dir=ravir_mask_dir,
        file_list=val_files,
        transform=get_round2_val_transform(Config.IMG_SIZE),
    )

    train_dataset = RAVIRBinaryDataset(ravir_train)
    val_dataset = RAVIRBinaryDataset(ravir_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
        persistent_workers=Config.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.NUM_WORKERS > 0,
    )

    log.info("=" * 60)
    log.info("  Round 2 — Domain Adaptation (RAVIR IR, Binary)")
    log.info("=" * 60)
    log.info("  Checkpoint : %s", args.checkpoint)
    log.info("  Train size : %d", len(train_dataset))
    log.info("  Val size   : %d", len(val_dataset))
    log.info("  Patch size : %d", patch_size)
    log.info("  Batch size : %d", args.batch_size)
    log.info("  Epochs     : %d", args.epochs)
    log.info("  LR         : %.1e", args.lr)

    # ── Model — load Round 1 weights, adapt 3ch → 1ch ────────────────
    model = RAVIRNet(
        in_channels=1,
        num_classes=1,
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,
        dropout_rate=Config.DROPOUT_RATE,
        use_attention=Config.USE_ATTENTION,
        binary_mode=True,
    ).to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        src_channels = ckpt.get("in_channels", 3)
        sd = adapt_state_dict_channels(ckpt["model_state_dict"], src_channels, tgt_channels=1)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info("Loaded Round 1 checkpoint (epoch %d, dice=%.4f)",
                 ckpt.get("epoch", -1), ckpt.get("best_dice", -1))
        if missing:
            log.info("  Missing keys : %s", missing)
        if unexpected:
            log.info("  Unexpected   : %s", unexpected)
    else:
        log.warning("No valid checkpoint provided — training Round 2 from scratch!")

    total_p = sum(p.numel() for p in model.parameters())
    log.info("  Params     : %s", f"{total_p:,}")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────
    criterion = BinaryDiceBCELoss(dice_weight=0.5, bce_weight=0.5).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = _create_scaler()
    use_amp, amp_dtype = _amp_settings()

    metrics_calc = BinarySegmentationMetrics()

    # ── Training loop ─────────────────────────────────────────────────
    best_dice = 0.0
    patience = 0
    train_losses, val_losses, val_dices = [], [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=args.grad_accum,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        val_loss, metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            metrics_calculator=metrics_calc,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        scheduler.step()
        dt = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(metrics["Dice"])

        log.info(
            "Epoch %3d/%d | Train %.4f | Val %.4f | "
            "Dice=%.4f  IoU=%.4f  Sens=%.4f  Spec=%.4f | "
            "LR=%.2e | %.0fs",
            epoch + 1, args.epochs, train_loss, val_loss,
            metrics["Dice"], metrics["IoU"],
            metrics["Sensitivity"], metrics["Specificity"],
            lr, dt,
        )

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Metrics/Dice", metrics["Dice"], epoch + 1)
        writer.add_scalar("Metrics/IoU", metrics["IoU"], epoch + 1)
        writer.add_scalar("LR", lr, epoch + 1)

        if metrics["Dice"] > best_dice:
            best_dice = metrics["Dice"]
            patience = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "metrics": metrics,
                "round": 2,
                "in_channels": 1,
                "config": {
                    "encoder_name": Config.ENCODER_NAME,
                    "dropout_rate": Config.DROPOUT_RATE,
                    "use_attention": Config.USE_ATTENTION,
                    "patch_size": patch_size,
                },
            }, os.path.join(run_dir, "round2_best.pth"))
            log.info("  ★ New best! Dice=%.4f  → round2_best.pth", best_dice)
        else:
            patience += 1
            log.info("  No improvement. Patience: %d/%d", patience, args.patience)

        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "round": 2,
                "in_channels": 1,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_dices": val_dices,
            }, os.path.join(run_dir, f"checkpoint_epoch{epoch + 1:03d}.pth"))

        if len(train_losses) > 1:
            try:
                plot_training_curves(train_losses, val_losses, val_dices, run_dir)
            except Exception:
                pass

        if patience >= args.patience:
            log.info("Early stopping at epoch %d", epoch + 1)
            break

    writer.close()
    log.info("=" * 60)
    log.info("  Round 2 Complete — Best Dice: %.4f", best_dice)
    log.info("  Checkpoint: %s", os.path.join(run_dir, "round2_best.pth"))
    log.info("=" * 60)
    return best_dice


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="Curriculum Learning for Retinal Vessel Segmentation",
    )
    p.add_argument("--round", type=int, required=True, choices=[1, 2],
                   help="Curriculum round (1=Vessel Discovery, 2=Domain Adaptation)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of epochs (default: 150 for R1, 100 for R2)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 1e-3 for R1, 5e-4 for R2)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size (default: auto from GPU profile)")
    p.add_argument("--patch-size", type=int, default=256,
                   help="Patch size for random cropping (default: 256)")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="Gradient accumulation steps (default: auto from GPU)")
    p.add_argument("--patience", type=int, default=60,
                   help="Early stopping patience")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to Round 1 checkpoint (required for --round 2)")
    p.add_argument("--output-dir", type=str,
                   default=os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "curriculum",
                   ),
                   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()

    if args.batch_size is None:
        args.batch_size = Config.BATCH_SIZE
    if args.grad_accum is None:
        args.grad_accum = Config.GRAD_ACCUMULATION_STEPS

    if args.round == 1:
        if args.epochs is None:
            args.epochs = 150
        if args.lr is None:
            args.lr = 1e-3
        train_round1(args)
    else:
        if args.epochs is None:
            args.epochs = 100
        if args.lr is None:
            args.lr = 5e-4
        if not args.checkpoint:
            print("WARNING: --checkpoint not provided for Round 2. "
                  "Training will start from scratch.")
        train_round2(args)


if __name__ == "_main_":
    main()