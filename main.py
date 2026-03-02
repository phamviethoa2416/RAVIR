import os
import time
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from losses import CombinedLoss
from metrics import SegmentationMetrics
from models import UNet
from training import train_one_epoch, validate
from transform import get_val_transform, get_train_transform
from transform.ravir import RAVIRDataset
from utils import set_seed, set_logging, visualize_predictions, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced ResUNet (CE+Dice) - RAVIR Vessel Segmentation")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--img-size", type=int, default=Config.IMG_SIZE)
    parser.add_argument("--grad-accum", type=int, default=Config.GRAD_ACCUMULATION_STEPS,
                        help="Gradient accumulation steps")
    parser.add_argument("--output-dir", type=str, default=Config.OUTPUT_DIR)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--summary", action="store_true",
                        help="Print model summary and exit")

    return parser.parse_args()


def summary():
    model = UNet(
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        channels=Config.CHANNELS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Input      : {Config.IN_CHANNELS}ch  {Config.IMG_SIZE}×{Config.IMG_SIZE}")
    print(f"  Output     : {Config.NUM_CLASSES} classes")
    print(f"  Channels   : {Config.CHANNELS}")
    print(f"  Dropout    : {Config.DROPOUT_RATE}")
    print(f"  Params     : {total:,} total  /  {trainable:,} trainable")
    print("-" * 60)

    dummy = torch.randn(1, Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    output = model(dummy)
    print(f"  Input  shape : {tuple(dummy.shape)}")
    print(f"  Output shape : {tuple(output.shape)}")
    print("=" * 60)


def train(args):
    set_seed(Config.SEED)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(Config.OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    logger = set_logging(run_dir)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

    logger.info("=" * 60)
    for k, v in [
        ("Device", Config.DEVICE),
        ("Image Size", f"{Config.IMG_SIZE}×{Config.IMG_SIZE}"),
        ("Num Classes", Config.NUM_CLASSES),
        ("Channels", Config.CHANNELS),
        ("Batch Size", Config.BATCH_SIZE),
        ("Grad Accum", Config.GRAD_ACCUMULATION_STEPS),
        ("Effective Batch", Config.BATCH_SIZE * Config.GRAD_ACCUMULATION_STEPS),
        ("Learning Rate", Config.LEARNING_RATE),
        ("Weight Decay", Config.WEIGHT_DECAY),
        ("Epochs", Config.EPOCHS),
        ("Early Stopping", f"patience={Config.EARLY_STOPPING_PATIENCE}"),
        ("Dropout", Config.DROPOUT_RATE),
        ("Output Dir", run_dir),
    ]:
        logger.info(f"  {k:<20}: {v}")
    logger.info("-" * 60)

    all_files = sorted(os.listdir(Config.TRAIN_IMG_DIR))
    logger.info(f"Total training images: {len(all_files)}")

    import random
    random.seed(Config.SEED)
    indices = list(range(len(all_files)))
    random.shuffle(indices)

    num_val = 4
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    val_files = [all_files[i] for i in val_indices]
    train_files = [all_files[i] for i in train_indices]

    logger.info(f"Train set: {len(train_files)} images")
    logger.info(f"Val set:   {len(val_files)} images")
    logger.info(f"Train files: {train_files}")
    logger.info(f"Val files:   {val_files}")

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
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model = UNet(
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        channels=Config.CHANNELS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Determine static weights if dynamic is False
    static_weights = None
    if not Config.USE_DYNAMIC_WEIGHTS:
        static_weights = torch.tensor(Config.CE_CLASS_WEIGHTS, dtype=torch.float32)

    criterion = CombinedLoss(
        num_classes=Config.NUM_CLASSES,
        dice_weight=Config.DICE_WEIGHT,
        ce_weight=Config.CE_WEIGHT,
        label_smoothing=0.1,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=Config.LR_DECAY_EVERY_N_EPOCHS,
        gamma=Config.LR_DECAY_FACTOR,
    )

    metrics_calc = SegmentationMetrics(Config.NUM_CLASSES)

    start_epoch = 0
    best_dice = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_dices = []

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
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

    max_epochs = Config.EPOCHS
    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch + 1}/{Config.EPOCHS} (lr={current_lr:.2e})")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            Config.DEVICE, grad_accum_steps=Config.GRAD_ACCUMULATION_STEPS,
            class_weights=static_weights
        )

        if Config.USE_DYNAMIC_WEIGHTS:
            # Dynamic weights are handled inside CombinedLoss if passed, 
            # but wait, trainer.py passes class_weights to criterion.
            # Let's check CombinedLoss again.
            pass

        val_loss, metrics = validate(
            model, val_loader, criterion, Config.DEVICE, metrics_calc,
            class_weights=static_weights
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_dice = metrics["Mean_Vessel_Dice"]
        val_dices.append(current_dice)

        epoch_time = time.time() - epoch_start
        logger.info(f"  Train Loss:  {train_loss:.4f}")
        logger.info(f"  Val Loss:    {val_loss:.4f}")
        logger.info(f"  Vessel Dice: {metrics['Mean_Vessel_Dice']:.4f}")
        logger.info(f"  Vessel IoU:  {metrics['Mean_Vessel_IoU']:.4f}")
        logger.info(f"  Vessel Sens: {metrics['Mean_Vessel_Sensitivity']:.4f}")

        for class_name in Config.CLASS_NAMES:
            d = metrics[f"{class_name}_Dice"]
            iou = metrics[f"{class_name}_IoU"]
            sens = metrics[f"{class_name}_Sensitivity"]
            logger.info(f"    {class_name:12s} -> Dice={d:.4f}  IoU={iou:.4f}  Sens={sens:.4f}")

        logger.info(f"  Epoch time:  {epoch_time:.1f}s")

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Metrics/Mean_Vessel_Dice", metrics["Mean_Vessel_Dice"], epoch + 1)
        writer.add_scalar("Metrics/Mean_Vessel_IoU", metrics["Mean_Vessel_IoU"], epoch + 1)
        writer.add_scalar("Metrics/Mean_Vessel_Sensitivity", metrics["Mean_Vessel_Sensitivity"], epoch + 1)
        writer.add_scalar("LR", current_lr, epoch + 1)
        for cls in Config.CLASS_NAMES:
            writer.add_scalar(f"Dice/{cls}", metrics[f"{cls}_Dice"], epoch + 1)
            writer.add_scalar(f"IoU/{cls}", metrics[f"{cls}_IoU"], epoch + 1)

        # Save the best model
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

        # Periodic checkpoint (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_epoch{epoch + 1:03d}.pth")
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
            }, ckpt_path)

        # Visualizations (epoch 0 + every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            visualize_predictions(model, val_loader, Config.DEVICE, vis_dir, epoch + 1)

        # Training curves
        if len(train_losses) > 1:
            plot_training_curves(train_losses, val_losses, val_dices, run_dir)

        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            logger.info(f"Best validation Dice: {best_dice:.4f}")
            break

    writer.close()
    logger.info("\n" + "=" * 60)
    logger.info("  Training Complete")
    logger.info("=" * 60)
    logger.info(f"Best Validation Mean Vessel Dice: {best_dice:.4f}")
    logger.info(f"Model saved at: {os.path.join(run_dir, 'best_model.pth')}")
    logger.info(f"Training curves saved at: {os.path.join(run_dir, 'training_curves.png')}")

    logger.info("\n--- Final Evaluation with Best Model ---")
    best_ckpt = torch.load(os.path.join(run_dir, "best_model.pth"), map_location=Config.DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])

    _, final = validate(
        model, val_loader, criterion, Config.DEVICE, metrics_calc,
        class_weights=static_weights
    )

    logger.info("Final Metrics:")
    logger.info(f"  Mean Vessel Dice:        {final['Mean_Vessel_Dice']:.4f}")
    logger.info(f"  Mean Vessel IoU:         {final['Mean_Vessel_IoU']:.4f}")
    logger.info(f"  Mean Vessel Sensitivity: {final['Mean_Vessel_Sensitivity']:.4f}")

    for cls in Config.CLASS_NAMES:
        logger.info(
            f"  {cls:<12}  Dice={final[f'{cls}_Dice']:.4f}  "
            f"IoU={final[f'{cls}_IoU']:.4f}  "
            f"Sens={final[f'{cls}_Sensitivity']:.4f}  "
            f"Spec={final[f'{cls}_Specificity']:.4f}"
        )

    visualize_predictions(
        model, val_loader, Config.DEVICE, vis_dir,
        epoch=999, num_samples=len(val_files),
    )
    return best_dice, final


if __name__ == "__main__":
    args = parse_args()
    if args.summary:
        summary()
    else:
        train(args)
