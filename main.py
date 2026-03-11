import argparse
import os
import time
from datetime import datetime

import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from losses import TverskyCELoss, MultiHeadLoss
from metrics import SegmentationMetrics
from models import RAVIRNet
from training import train_one_epoch, validate, create_scaler
from transform import get_val_transform, get_train_transform, compute_class_weights
from transform.ravir import RAVIRDataset
from utils import seed_everything, setup_logging, visualize_predictions, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="RAVIRNet Training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--summary", action="store_true",
                        help="Print model summary and exit")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config: KEY=VALUE pairs")
    return parser.parse_args()

def apply_cli_overrides(args):
    mapping = {
        "epochs": "EPOCHS",
        "batch_size": "BATCH_SIZE",
        "lr": "LEARNING_RATE",
        "img_size": "IMG_SIZE",
        "grad_accum": "GRAD_ACCUMULATION_STEPS",
        "output_dir": "OUTPUT_DIR",
    }
    overrides = {}
    for arg_name, config_name in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            overrides[config_name] = val

    for item in args.override:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                continue
        if val == "True":
            val = True
        elif val == "False":
            val = False
        overrides[key] = val

    if overrides:
        Config.override(**overrides)
    return overrides


def summary():
    print(Config.summary())
    print()

    model = RAVIRNet(
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        channels=Config.CHANNELS,
        dropout_rate=Config.DROPOUT_RATE,
        use_attention=Config.USE_ATTENTION,
    ).to(Config.DEVICE)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model Architecture:")
    print(f"  Channels   : {Config.CHANNELS}")
    print(f"  Input      : {Config.IN_CHANNELS}ch  {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"  Dropout    : {Config.DROPOUT_RATE}")
    print(f"  Params     : {total:,} total  /  {trainable:,} trainable")
    print("-" * 60)

    dummy = torch.randn(1, Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    output = model(dummy)
    print(f"  Input shape          : {tuple(dummy.shape)}")
    for k, v in output.items():
        print(f"  {k:22s}: {tuple(v.shape)}")
    print("=" * 60)


def train(args):
    seed_everything(Config.SEED)

    # ── Run directory ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(Config.OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    logger = setup_logging(run_dir)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

    device = Config.DEVICE
    if device == "cuda":
        torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
        if Config.ALLOW_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    logger.info(Config.summary())
    logger.info("-" * 60)
    for k, v in [
        ("LR", Config.LEARNING_RATE),
        ("Weight Decay", Config.WEIGHT_DECAY),
        ("Epochs", Config.EPOCHS),
        ("Early Stopping", f"patience={Config.EARLY_STOPPING_PATIENCE}"),
        ("Channels", Config.CHANNELS),
        ("Scheduler", f"{Config.LR_SCHEDULER} (T0={Config.COSINE_T0}, Tmult={Config.COSINE_T_MULT})"),
        ("Loss", f"TverskyCE (α={Config.TVERSKY_ALPHA}, β={Config.TVERSKY_BETA}) + VesselProb"),
        ("Label Smoothing", Config.LABEL_SMOOTHING),
        ("Dropout", Config.DROPOUT_RATE),
        ("Output Dir", run_dir),
    ]:
        logger.info(f"  {k:<20}: {v}")
    logger.info("-" * 60)

    all_files = sorted([f for f in os.listdir(Config.TRAIN_IMG_DIR) if f.endswith(".png")])
    logger.info(f"Total training images: {len(all_files)}")

    kf = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    folds = list(kf.split(all_files))
    train_idx, val_idx = folds[Config.VAL_FOLD]
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]

    logger.info(f"Fold {Config.VAL_FOLD}: train={len(train_files)}, val={len(val_files)}")

    class_weights = compute_class_weights(Config.TRAIN_MASK_DIR, train_files)
    logger.info(f"Class weights (computed): {class_weights.tolist()}")

    train_dataset = RAVIRDataset(
        img_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        file_list=train_files,
        transform=get_train_transform(),
    )
    val_dataset = RAVIRDataset(
        img_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        file_list=val_files,
        transform=get_val_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
        persistent_workers=Config.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.NUM_WORKERS > 0,
    )

    model = RAVIRNet(
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        channels=Config.CHANNELS,
        dropout_rate=Config.DROPOUT_RATE,
        use_attention=Config.USE_ATTENTION,
    ).to(device)

    if Config.COMPILE_MODEL and hasattr(torch, "compile"):
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    seg_criterion = TverskyCELoss(
        num_classes=Config.NUM_CLASSES,
        tversky_weight=Config.TVERSKY_WEIGHT,
        tversky_alpha=Config.TVERSKY_ALPHA,
        tversky_beta=Config.TVERSKY_BETA,
        ce_weight=Config.TVERSKY_CE_WEIGHT,
        label_smoothing=Config.LABEL_SMOOTHING,
        class_weights=class_weights,
    ).to(device)

    criterion = MultiHeadLoss(
        seg_criterion=seg_criterion,
        vessel_prob_weight=Config.VESSEL_PROB_LOSS_WEIGHT,
        vessel_prob_pos_weight=Config.VESSEL_PROB_POS_WEIGHT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=1e-6,
    )

    scaler = create_scaler()

    metrics_calc = SegmentationMetrics(Config.NUM_CLASSES)

    start_epoch = 0
    best_dice = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_dices = []

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing:
            logger.info(f"  Missing keys : {missing}")
        if unexpected:
            logger.info(f"  Unexpected   : {unexpected}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_dice = checkpoint["best_dice"]
        patience_counter = checkpoint.get("patience_counter", 0)
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        val_dices = checkpoint.get("val_dices", [])
        logger.info(f"Resumed at epoch {start_epoch}, best_dice={best_dice:.4f}")

    logger.info("=" * 60)
    logger.info("  Starting Training")
    logger.info("=" * 60)

    for epoch in range(start_epoch, Config.EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch + 1}/{Config.EPOCHS} (lr={current_lr:.2e})")

        train_loss, train_details = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=Config.GRAD_ACCUMULATION_STEPS,
            scaler=scaler,
        )

        val_loss, metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            metrics_calculator=metrics_calc,
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_dice = metrics["Mean_Vessel_Dice"]
        val_dices.append(current_dice)

        logger.info(
            f"Epoch {epoch + 1:3d}/{Config.EPOCHS} | "
            f"Train {train_loss:.4f} "
            f"(seg={train_details.get('seg_loss', 0):.3f} "
            f"vp={train_details.get('vessel_prob_loss', 0):.3f}) | "
            f"Val {val_loss:.4f} | "
            f"Dice: Art={metrics['artery_Dice']:.4f} "
            f"Vein={metrics['vein_Dice']:.4f} "
            f"Mean={current_dice:.4f} | "
            f"IoU={metrics['Mean_Vessel_IoU']:.4f} | "
            f"LR={current_lr:.2e} | {epoch_time:.0f}s"
        )

        for class_name in Config.CLASS_NAMES:
            d = metrics[f"{class_name}_Dice"]
            iou = metrics[f"{class_name}_IoU"]
            sens = metrics[f"{class_name}_Sensitivity"]
            logger.info(f"    {class_name:12s} -> Dice={d:.4f}  IoU={iou:.4f}  Sens={sens:.4f}")

        logger.info(f"  Epoch time:  {epoch_time:.1f}s")

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Loss/seg", train_details.get("seg_loss", 0), epoch + 1)
        writer.add_scalar("Loss/vessel_prob", train_details.get("vessel_prob_loss", 0), epoch + 1)
        writer.add_scalar("Metrics/Mean_Vessel_Dice", current_dice, epoch + 1)
        writer.add_scalar("Metrics/Mean_Vessel_IoU", metrics["Mean_Vessel_IoU"], epoch + 1)
        writer.add_scalar("LR", current_lr, epoch + 1)
        for cls in Config.CLASS_NAMES:
            writer.add_scalar(f"Dice/{cls}", metrics[f"{cls}_Dice"], epoch + 1)
            writer.add_scalar(f"IoU/{cls}", metrics[f"{cls}_IoU"], epoch + 1)

        if current_dice > best_dice:
            best_dice = current_dice
            patience_counter = 0
            best_path = os.path.join(run_dir, "best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "metrics": metrics,
                "config": {
                    "channels": Config.CHANNELS,
                    "num_classes": Config.NUM_CLASSES,
                    "img_size": Config.IMG_SIZE,
                    "dropout": Config.DROPOUT_RATE,
                },
            }, best_path)
            logger.info(f"  ★ New best model saved! Dice={best_dice:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

        if (epoch + 1) % 50 == 0:
            periodic_path = os.path.join(run_dir, f"checkpoint_epoch{epoch + 1:03d}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "patience_counter": patience_counter,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_dices": val_dices,
            }, periodic_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            try:
                visualize_predictions(model, val_loader, device, vis_dir, epoch + 1)
            except Exception as e:
                logger.warning(f"  Visualization failed: {e}")

        if len(train_losses) > 1:
            try:
                plot_training_curves(train_losses, val_losses, val_dices, run_dir)
            except Exception as e:
                logger.warning(f"  Plot failed: {e}")

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info(f"\nEarly stopping at epoch {epoch + 1} "
                        f"(no improvement for {Config.EARLY_STOPPING_PATIENCE} epochs)")
            break

    writer.close()
    logger.info("\n" + "=" * 60)
    logger.info("  Training Complete")
    logger.info("=" * 60)
    logger.info(f"Best Mean Vessel Dice: {best_dice:.4f}")

    best_ckpt = torch.load(os.path.join(run_dir, "best_model.pth"), map_location=Config.DEVICE)
    if os.path.isfile(best_ckpt):
        best_ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        _, final = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            metrics_calculator=metrics_calc,
        )

        logger.info("Final Metrics (best checkpoint):")
        logger.info(f"  Mean Vessel Dice : {final['Mean_Vessel_Dice']:.4f}")
        logger.info(f"  Mean Vessel IoU  : {final['Mean_Vessel_IoU']:.4f}")
        logger.info(f"  Mean Vessel Sens : {final['Mean_Vessel_Sensitivity']:.4f}")

        for cls in Config.CLASS_NAMES:
            logger.info(
                f"  {cls:<12}  Dice={final[f'{cls}_Dice']:.4f}  "
                f"IoU={final[f'{cls}_IoU']:.4f}  "
                f"Sens={final[f'{cls}_Sensitivity']:.4f}  "
                f"Spec={final[f'{cls}_Specificity']:.4f}"
            )

        try:
            visualize_predictions(
                model, val_loader, device, vis_dir,
                epoch=999, num_samples=len(val_files),
            )
        except Exception as e:
            logger.warning(f"Final visualization failed: {e}")

    logger.info(f"\nCheckpoint : {best_ckpt}")
    logger.info(f"Run dir    : {run_dir}")

    return best_dice


if __name__ == "__main__":
    args = parse_args()
    applied = apply_cli_overrides(args)
    if applied:
        print(f"Config overrides: {applied}")

    if args.summary:
        summary()
    else:
        train(args)
