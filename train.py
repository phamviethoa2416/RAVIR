from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import torch
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from losses import VesselSegmentationLoss
from metrics import SegmentationMetrics
from models import RAVIRNet
from training import create_scaler, train_one_epoch, validate
from transform import (
    compute_class_weights,
    RAVIRDataset,
    get_train_transform,
    get_val_transform,
)
from utils import seed_everything, setup_logging, visualize_predictions, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="RAVIRNet Training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print model summary and exit",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config: KEY=VALUE pairs",
    )
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
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        use_deep_supervision=Config.USE_DEEP_SUPERVISION,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: RAVIRNet")
    print(f"  Encoder       : {Config.ENCODER_NAME} ({Config.ENCODER_WEIGHTS})")
    print(
        f"  Input         : {Config.IN_CHANNELS}ch × {Config.IMG_SIZE}×{Config.IMG_SIZE}"
    )
    print(f"  Skip channels : {model.encoder.skip_channels}")
    print(f"  Bottleneck    : {model.encoder.bottleneck_channels}")
    print(f"  Params        : {total:,} total / {trainable:,} trainable")
    print()

    dummy = torch.randn(
        1,
        Config.IN_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy)

    print(f"  Input shape   : {tuple(dummy.shape)}")
    print(f"  Segmentation output    : {tuple(output['seg'].shape)}")
    print(f"  Deep Supervision heads      : {len(output['ds'])}")
    for i, ds in enumerate(output["ds"]):
        print(f"    ds[{i}]       : {tuple(ds.shape)}")


def train(args):
    seed_everything(Config.SEED)

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

    logger.info("=" * 60)
    logger.info("  RAVIR Training")
    logger.info("=" * 60)
    logger.info(Config.summary())
    logger.info("-" * 60)

    for k, v in [
        ("LR", Config.LEARNING_RATE),
        ("Weight Decay", Config.WEIGHT_DECAY),
        ("Epochs", Config.EPOCHS),
        ("Early Stopping", f"patience={Config.EARLY_STOPPING_PATIENCE}"),
        (
                "Scheduler",
                f"{Config.LR_SCHEDULER} (T0={Config.COSINE_T0}, Tmult={Config.COSINE_T_MULT})",
        ),
        (
                "Loss",
                f"CE({Config.CE_WEIGHT}) + Dice({Config.DICE_WEIGHT}) + SkelRecall({Config.SKELETON_WEIGHT})",
        ),
        ("DS", f"weight={Config.DS_WEIGHT}, decay={Config.DS_DECAY}"),
        ("Label Smoothing", Config.LABEL_SMOOTHING),
        ("Dropout", Config.DROPOUT_RATE),
        ("Output Dir", run_dir),
    ]:
        logger.info(f"  {k:<20}: {v}")
    logger.info("-" * 60)

    all_files = sorted(
        f for f in os.listdir(Config.TRAIN_IMG_DIR) if f.endswith(".png")
    )
    logger.info(f"Total training images: {len(all_files)}")

    kf = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    folds = list(kf.split(all_files))
    train_idx, val_idx = folds[Config.VAL_FOLD]
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]

    logger.info(
        f"Fold {Config.VAL_FOLD}: train={len(train_files)}, val={len(val_files)}"
    )

    class_weights = compute_class_weights(Config.TRAIN_MASK_DIR, train_files)
    class_weights = class_weights.to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    train_dataset = RAVIRDataset(
        img_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        file_list=train_files,
        transform=get_train_transform(),
        skeleton_cache_dir=Config.SKELETON_CACHE_DIR,
        tube_radius=Config.TUBE_RADIUS,
    )

    val_dataset = RAVIRDataset(
        img_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        file_list=val_files,
        transform=get_val_transform(),
        skeleton_cache_dir=Config.SKELETON_CACHE_DIR,
        tube_radius=Config.TUBE_RADIUS,
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
        encoder_name=Config.ENCODER_NAME,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        encoder_weights=Config.ENCODER_WEIGHTS,
        use_deep_supervision=Config.USE_DEEP_SUPERVISION,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    if Config.COMPILE_MODEL and hasattr(torch, "compile"):
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total / {trainable_params:,} trainable")

    criterion = VesselSegmentationLoss(
        num_classes=Config.NUM_CLASSES,
        ce_weight=Config.CE_WEIGHT,
        dice_weight=Config.DICE_WEIGHT,
        skeleton_weight=Config.SKELETON_WEIGHT,
        ds_weight=Config.DS_WEIGHT,
        ds_decay=Config.DS_DECAY,
        label_smoothing=Config.LABEL_SMOOTHING,
        class_weights=class_weights,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.COSINE_T0,
        T_mult=Config.COSINE_T_MULT,
        eta_min=1e-6,
    )

    scaler = create_scaler()

    metrics_calc = SegmentationMetrics(
        num_classes=Config.NUM_CLASSES,
        class_names=Config.CLASS_NAMES,
        compute_cldice=True,
    )

    start_epoch = 0
    best_dice = 0.0
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
    }

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_dice = ckpt["best_dice"]
        patience_counter = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", history)
        logger.info(f"Resumed at epoch {start_epoch}, best_dice={best_dice:.4f}")

    logger.info("=" * 60)
    logger.info("  Start training")
    logger.info("=" * 60)

    for epoch in range(start_epoch, Config.EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch + 1}/{Config.EPOCHS}  lr={current_lr:.2e}")

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

        current_dice = metrics["Mean_Vessel_Dice"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(current_dice)

        skel_str = f"  skel={train_details.get('skel', 0):.3f}" if "skel" in train_details else ""
        ds_str = f"  ds={train_details.get('ds', 0):.3f}" if "ds" in train_details else ""
        cldice_str = ""
        if "Mean_Vessel_clDice" in metrics:
            cldice_str = f"  clDice={metrics['Mean_Vessel_clDice']:.4f}"

        logger.info(
            f"  Train loss={train_loss:.4f}"
            f"  (seg={train_details.get('seg', 0):.3f}{skel_str}{ds_str})"
        )
        logger.info(
            f"Val loss={val_loss:.4f} | "
            f"Dice: Artery={metrics['artery_Dice']:.4f} "
            f"Vein={metrics['vein_Dice']:.4f} "
            f"Mean={current_dice:.4f} | "
            f"IoU={metrics['Mean_Vessel_IoU']:.4f}"
            f"{cldice_str} | {epoch_time:.0f}s"
        )

        for name in Config.CLASS_NAMES:
            d = metrics[f"{name}_Dice"]
            iou = metrics[f"{name}_IoU"]
            sens = metrics[f"{name}_Sensitivity"]
            cl = metrics.get(f"{name}_clDice", None)
            cl_str = f"  clDice={cl:.4f}" if cl is not None else ""
            logger.info(f"    {name:12s}  Dice={d:.4f}  IoU={iou:.4f}  Sens={sens:.4f}{cl_str}")

        logger.info(f"  Epoch time:  {epoch_time:.1f}s")

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        for k, v in train_details.items():
            writer.add_scalar(f"Loss/{k}", v, epoch + 1)
        writer.add_scalar("Metrics/Mean_Vessel_Dice", current_dice, epoch + 1)
        writer.add_scalar("Metrics/Mean_Vessel_IoU", metrics["Mean_Vessel_IoU"], epoch + 1)
        if "Mean_Vessel_clDice" in metrics:
            writer.add_scalar("Metrics/Mean_Vessel_clDice", metrics["Mean_Vessel_clDice"], epoch + 1)
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
                    "encoder_name": Config.ENCODER_NAME,
                    "in_channels": Config.IN_CHANNELS,
                    "num_classes": Config.NUM_CLASSES,
                    "img_size": Config.IMG_SIZE,
                    "dropout": Config.DROPOUT_RATE,
                },
            }, best_path)
            logger.info(f"  ★ New best model saved! Dice={best_dice:.4f}")
        else:
            patience_counter += 1
            logger.info(
                f"  No improvement ({patience_counter}/{Config.EARLY_STOPPING_PATIENCE})"
            )

        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_epoch{epoch + 1:03d}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "patience_counter": patience_counter,
                "history": history,
            }, ckpt_path)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            try:
                visualize_predictions(model, val_loader, device, vis_dir, epoch + 1)
            except Exception as e:
                logger.warning(f"  Visualization failed: {e}")

        if len(history["train_loss"]) > 1:
            try:
                plot_training_curves(history["train_loss"], history["val_loss"], history["val_dice"], run_dir)
            except Exception as e:
                logger.warning(f"  Plot failed: {e}")

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info(
                f"\nEarly stopping at epoch {epoch + 1} "
                f"(no improvement for {Config.EARLY_STOPPING_PATIENCE} epochs)"
            )
            break

    writer.close()
    logger.info("\n" + "=" * 60)
    logger.info("  Training Complete")
    logger.info("=" * 60)
    logger.info(f"Best Mean Vessel Dice: {best_dice:.4f}")

    best_path = os.path.join(run_dir, "best_model.pth")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        _, final = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            metrics_calculator=metrics_calc,
        )

        logger.info("\nFinal metrics (best checkpoint):")
        logger.info(metrics_calc.summary())

    try:
        visualize_predictions(
            model,
            val_loader,
            device,
            vis_dir,
            epoch=999,
            num_samples=len(val_files),
        )
    except Exception as e:
        logger.warning(f"Final visualization failed: {e}")

    logger.info(f"\nCheckpoint : {best_path}")
    logger.info(f"Run directory: {run_dir}")
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
