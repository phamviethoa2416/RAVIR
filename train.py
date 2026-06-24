from __future__ import annotations

import gc
import os
import time
from datetime import datetime

import torch
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from losses import SegmentationLoss
from metrics import SegmentationMetrics
from models import RAVIRNet
from training import (
    EMAWeights,
    build_training_checkpoint,
    create_scaler,
    get_scheduler,
    load_model_state_dict,
    model_weights_for_save,
    train_one_epoch,
    validate,
)
from transform import (
    RAVIRDataset,
    compute_class_weights,
    get_train_transform,
    get_val_transform,
)
from utils import (
    plot_training_curves,
    seed_everything,
    setup_logging,
    visualize_predictions,
)


def train(args):
    seed_everything(Config.SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(Config.OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    visualization_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    logger = setup_logging(run_dir)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

    device = Config.DEVICE
    if device == "cuda":
        torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = not Config.CUDNN_BENCHMARK
        if Config.ALLOW_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    logger.info("=" * 60)
    logger.info("  RAVIR Training")
    logger.info("=" * 60)
    logger.info(Config.summary())
    logger.info("-" * 60)

    all_files = sorted(
        f for f in os.listdir(Config.TRAIN_IMG_DIR) if f.endswith(".png")
    )
    logger.info(f"Total training images: {len(all_files)}")

    kf = KFold(
        n_splits=Config.NUM_FOLDS,
        shuffle=True,
        random_state=Config.SEED,
    )
    folds = list(kf.split(all_files))
    train_idx, val_idx = folds[Config.VAL_FOLD]
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]

    logger.info(
        f"Fold {Config.VAL_FOLD}: train={len(train_files)}, val={len(val_files)}"
    )

    if Config.USE_DYNAMIC_WEIGHTS:
        class_weights = compute_class_weights(Config.TRAIN_MASK_DIR, train_files).to(
            device
        )
    else:
        class_weights = torch.tensor(Config.CE_CLASS_WEIGHTS, dtype=torch.float32).to(
            device
        )
    logger.info(
        f"Class weights (dynamic={Config.USE_DYNAMIC_WEIGHTS}): {class_weights.tolist()}"
    )

    frangi_cache_dir = (
        Config.FRANGI_CACHE_DIR
        if (Config.USE_FRANGI or Config.USE_FRANGI_RECON)
        else None
    )
    if Config.USE_FRANGI_RECON:
        Config.USE_FRANGI = False
        Config.IN_CHANNELS = 3

    train_dataset = RAVIRDataset(
        img_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        file_list=train_files,
        transform=get_train_transform(),
        frangi_cache_dir=frangi_cache_dir,
        return_frangi_target=Config.USE_FRANGI_RECON,
        use_rotation_expansion=Config.USE_ROTATION_EXPANSION,
        use_branch_labels=Config.USE_CONTRASTIVE_LOSS,
        branch_crossing_radius=Config.SEGCON_BRANCH_CROSSING_RADIUS,
        branch_min_pixels=Config.SEGCON_BRANCH_MIN_PIXELS,
        branch_node_proximity=Config.SEGCON_BRANCH_NODE_PROXIMITY,
        branch_small_mode=Config.SEGCON_BRANCH_SMALL_MODE,
    )

    val_dataset = RAVIRDataset(
        img_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        file_list=val_files,
        transform=get_val_transform(),
        frangi_cache_dir=frangi_cache_dir,
        return_frangi_target=Config.USE_FRANGI_RECON,
        use_branch_labels=Config.USE_CONTRASTIVE_LOSS,
        branch_crossing_radius=Config.SEGCON_BRANCH_CROSSING_RADIUS,
        branch_min_pixels=Config.SEGCON_BRANCH_MIN_PIXELS,
        branch_node_proximity=Config.SEGCON_BRANCH_NODE_PROXIMITY,
        branch_small_mode=Config.SEGCON_BRANCH_SMALL_MODE,
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

    resume_checkpoint: dict | None = None
    if getattr(args, "resume", None) and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint '{args.resume}'")
        resume_checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )

    model = RAVIRNet(
        encoder_name=Config.ENCODER_NAME,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        encoder_weights=Config.ENCODER_WEIGHTS,
        dropout=Config.DROPOUT_RATE,
        use_scse=Config.USE_SCSE,
        use_attention=Config.USE_ATTENTION,
        use_deep_supervision=Config.USE_DEEP_SUPERVISION,
        use_frangi_recon=Config.USE_FRANGI_RECON,
        aux_mid_channels=Config.AUX_DECODER_CHANNELS,
        cl_projector_stage_idx=(
            Config.SEGCON_PROJECTOR_STAGE_IDX if Config.USE_CONTRASTIVE_LOSS else None
        ),
        cl_embedding_dim=Config.SEGCON_EMBEDDING_DIM,
        cl_hidden_dim=Config.SEGCON_HIDDEN_DIM,
        use_refinement=Config.USE_RECURSIVE_REFINEMENT,
        refinement_iterations=Config.REFINEMENT_ITERATIONS,
        refinement_base_channels=Config.REFINEMENT_BASE_CHANNELS,
    ).to(device)

    if resume_checkpoint is not None:
        load_model_state_dict(model, resume_checkpoint["model_state_dict"])
        logger.info("Restored model weights from checkpoint")

    if Config.COMPILE_MODEL and hasattr(torch, "compile"):
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile()")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total / {trainable_params:,} trainable")

    criterion = SegmentationLoss(
        num_classes=Config.NUM_CLASSES,
        dice_weight=Config.DICE_WEIGHT,
        ce_weight=Config.CE_WEIGHT,
        skeleton_weight=Config.SKELETON_WEIGHT,
        ds_weight=Config.DS_WEIGHT,
        ds_decay=Config.DS_DECAY,
        class_weights=class_weights,
        use_clidce=Config.USE_CLDICE,
        cldice_num_iterations=Config.CLDICE_NUM_ITERATIONS,
        # Contrastive Learning ───────────────────
        segcon_weight=Config.SEGCON_WEIGHT if Config.USE_CONTRASTIVE_LOSS else 0.0,
        segcon_temperature=Config.SEGCON_TEMPERATURE,
        segcon_num_anchors=Config.SEGCON_NUM_ANCHORS,
        segcon_num_positives=Config.SEGCON_NUM_POSITIVES,
        segcon_num_negatives=Config.SEGCON_NUM_NEGATIVES,
        segcon_negative_radius=Config.SEGCON_NEGATIVE_RADIUS,
        segcon_confidence_gated=Config.SEGCON_CONFIDENCE_GATED,
        segcon_confidence_gamma=Config.SEGCON_CONFIDENCE_GAMMA,
        segcon_confidence_detach=Config.SEGCON_CONFIDENCE_DETACH,
        # Recursive Refinement ───────────────────
        use_refinement=Config.USE_RECURSIVE_REFINEMENT,
        refinement_mode=Config.REFINEMENT_MODE,
        refinement_iteration_weight=Config.REFINEMENT_ITERATION_WEIGHT,
        refinement_base_segmentation_weight=Config.REFINEMENT_BASE_SEGMENTATION_WEIGHT,
        # Frangi auxiliary head ───────────────────
        frangi_recon_weight=(
            Config.FRANGI_RECON_WEIGHT if Config.USE_FRANGI_RECON else 0.0
        ),
        frangi_recon_loss=Config.FRANGI_RECON_LOSS,
        frangi_recon_vessel_weight=Config.FRANGI_RECON_VESSEL_WEIGHT,
        frangi_recon_frangi_weight=Config.FRANGI_RECON_FRANGI_WEIGHT,
        frangi_recon_frangi_vessel_only=Config.FRANGI_RECON_FRANGI_VESSEL_ONLY,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_epochs=Config.WARMUP_EPOCHS,
        cosine_t0=Config.COSINE_T0,
        cosine_t_mult=Config.COSINE_T_MULT,
        cosine_eta_min=1e-6,
    )

    scaler = create_scaler()

    ema = (
        EMAWeights(
            model=model,
            decay=Config.EMA_DECAY,
            warmup=Config.EMA_WARMUP_STEPS,
            device=device,
        )
        if Config.USE_EMA
        else None
    )

    if ema is not None:
        logger.info(
            f"EMA enabled: decay={Config.EMA_DECAY}, warmup={Config.EMA_WARMUP_STEPS} steps"
        )

    metrics_calc = SegmentationMetrics(
        num_classes=Config.NUM_CLASSES,
        class_names=Config.CLASS_NAMES,
    )

    start_epoch = 0
    best_dice = 0.0
    patience = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
    }

    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"])
        best_dice = float(resume_checkpoint.get("best_dice", 0.0))
        patience = int(resume_checkpoint.get("patience", 0))
        history = resume_checkpoint.get("history", history)

        if ema is not None and resume_checkpoint.get("ema_state_dict"):
            ema.load_state_dict(resume_checkpoint["ema_state_dict"])
            logger.info("Restored EMA weights from checkpoint")

        if scaler is not None and resume_checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
            logger.info("Restored GradScaler state from checkpoint")

        logger.info(
            f"Resumed at epoch {start_epoch + 1}/{Config.EPOCHS}, "
            f"best dice={best_dice:.4f}"
        )

    logger.info("=" * 60)
    logger.info("  Start training")
    logger.info("=" * 60)

    for epoch in range(start_epoch, Config.EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"\nEpoch {epoch + 1}/{Config.EPOCHS}, learning rate = {current_lr:.5f}"
        )

        train_loss, train_details = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=Config.GRAD_ACCUMULATION_STEPS,
            scaler=scaler,
            ema=ema,
        )

        if ema is not None:
            with ema.swap(model):
                val_loss, metrics = validate(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    metrics_calc=metrics_calc,
                )
        else:
            val_loss, metrics = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                metrics_calc=metrics_calc,
            )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        current_dice = metrics.get(
            "Mean_Vessel_Dice_per_image", metrics["Mean_Vessel_Dice"]
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(current_dice)

        ds_str = (
            f" | ds={train_details.get('ds', 0):.3f}" if "ds" in train_details else ""
        )
        logger.info(
            f"Train loss = {train_loss:.4f} | "
            f"seg={train_details.get('segmentation', 0):.4f}"
            f"{ds_str}"
        )

        logger.info(
            f"Validation loss = {val_loss:.4f} | "
            f"Dice: Artery = {metrics['artery_dice']:.4f} "
            f"Vein = {metrics['vein_dice']:.4f} "
            f"Mean = {current_dice:.4f} | "
            f"IoU = {metrics['Mean_Vessel_IoU']:.4f} | "
            f"{epoch_time:.0f}s"
        )

        for name in Config.CLASS_NAMES:
            d = metrics[f"{name}_dice"]
            iou = metrics[f"{name}_iou"]
            sens = metrics[f"{name}_sensitivity"]
            cl = metrics.get(f"{name}_clDice", None)
            cl_str = f"  clDice={cl:.4f}" if cl is not None else ""
            logger.info(
                f"    {name:12s}  Dice={d:.4f}  IoU={iou:.4f}  Sens={sens:.4f}{cl_str}"
            )

        logger.info(f"  Epoch time:  {epoch_time:.1f}s")

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        for k, v in train_details.items():
            writer.add_scalar(f"Loss/{k}", v, epoch + 1)
        writer.add_scalar("Metrics/Mean_Vessel_Dice", current_dice, epoch + 1)
        writer.add_scalar(
            "Metrics/Mean_Vessel_IoU", metrics["Mean_Vessel_IoU"], epoch + 1
        )
        writer.add_scalar("LR", current_lr, epoch + 1)
        for cls in Config.CLASS_NAMES:
            writer.add_scalar(f"Dice/{cls}", metrics[f"{cls}_dice"], epoch + 1)
            writer.add_scalar(f"IoU/{cls}", metrics[f"{cls}_iou"], epoch + 1)

        if current_dice > best_dice:
            best_dice = current_dice
            patience = 0
            best_path = os.path.join(run_dir, "best_model.pth")
            best_payload = build_training_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_dice=best_dice,
                patience=patience,
                history=history,
                ema=ema,
                scaler=scaler,
            )
            best_payload["model_state_dict"] = model_weights_for_save(
                model, ema, prefer_ema=ema is not None
            )
            best_payload["metrics"] = metrics
            best_payload["config"] = {
                "encoder_name": Config.ENCODER_NAME,
                "encoder_weights": Config.ENCODER_WEIGHTS,
                "in_channels": Config.IN_CHANNELS,
                "num_classes": Config.NUM_CLASSES,
                "img_size": Config.IMG_SIZE,
                "dropout": Config.DROPOUT_RATE,
                "refinement_num_iterations": Config.REFINEMENT_ITERATIONS,
                "refinement_base_channels": Config.REFINEMENT_BASE_CHANNELS,
            }
            torch.save(best_payload, best_path)
            logger.info(f"★ New best model saved! Dice={best_dice:.4f}")
        else:
            patience += 1
            logger.info(f"No improvement ({patience}/{Config.EARLY_STOPPING_PATIENCE})")

        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch{epoch + 1}.pth")
            torch.save(
                build_training_checkpoint(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_dice=best_dice,
                    patience=patience,
                    history=history,
                    ema=ema,
                    scaler=scaler,
                ),
                checkpoint_path,
            )

        if (epoch + 1) % 20 == 0 or epoch == 0:
            try:
                if ema is not None:
                    with ema.swap(model):
                        visualize_predictions(
                            model,
                            val_loader,
                            device,
                            visualization_dir,
                            epoch + 1,
                        )
                else:
                    visualize_predictions(
                        model,
                        val_loader,
                        device,
                        visualization_dir,
                        epoch + 1,
                    )
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")

        if len(history["train_loss"]) > 1:
            try:
                plot_training_curves(
                    history["train_loss"],
                    history["val_loss"],
                    history["val_dice"],
                    run_dir,
                )
            except Exception as e:
                logger.warning(f"Plot failed: {e}")

        if patience >= Config.EARLY_STOPPING_PATIENCE:
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
        load_model_state_dict(model, ckpt["model_state_dict"])
        validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            metrics_calc=metrics_calc,
        )

        logger.info("\nFinal metrics (best checkpoint):")
        logger.info(metrics_calc.summary())
        del ckpt

    try:
        if ema is not None:
            with ema.swap(model):
                visualize_predictions(
                    model,
                    val_loader,
                    device,
                    visualization_dir,
                    epoch=999,
                    num_samples=len(val_files),
                )
        else:
            visualize_predictions(
                model,
                val_loader,
                device,
                visualization_dir,
                epoch=999,
                num_samples=len(val_files),
            )
    except Exception as e:
        logger.warning(f"Final visualization failed: {e}")

    logger.info(f"\nCheckpoint : {best_path}")
    logger.info(f"Run directory: {run_dir}")

    del model, criterion, optimizer, scheduler, writer
    del train_loader, val_loader, train_dataset, val_dataset
    del metrics_calc, class_weights, history
    if ema is not None:
        del ema
    if scaler is not None:
        del scaler
    if resume_checkpoint is not None:
        del resume_checkpoint

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_dice, run_dir
