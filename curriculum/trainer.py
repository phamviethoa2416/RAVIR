from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm


def train_one_epoch_binary(
        model: nn.Module,
        loader,
        criterion: nn.Module,
        optimizer,
        device: str,
        grad_accum_steps: int = 1,
        scaler: GradScaler | None = None,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float32,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  Train", leave=False)
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
            main_loss = criterion(outputs["vessel_prob"], masks)

            # Deep supervision: auxiliary segmentation heads (if available)
            if isinstance(outputs, dict) and "deep_supervision" in outputs:
                ds_maps = outputs["deep_supervision"]
                # Weights for deeper to shallower outputs
                ds_weights = [0.5, 0.3, 0.2]
                ds_loss = 0.0
                for w, ds_logits in zip(ds_weights, ds_maps):
                    ds_loss = ds_loss + w * criterion(ds_logits, masks)
                loss = main_loss + ds_loss
            else:
                loss = main_loss

        scaled_loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate_binary(
        model: nn.Module,
        loader,
        criterion: nn.Module,
        device: str,
        metrics_calculator,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float32,
) -> tuple[float, dict]:
    model.eval()
    metrics_calculator.reset()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
            main_loss = criterion(outputs["vessel_prob"], masks)

            if isinstance(outputs, dict) and "deep_supervision" in outputs:
                ds_maps = outputs["deep_supervision"]
                ds_weights = [0.5, 0.3, 0.2]
                ds_loss = 0.0
                for w, ds_logits in zip(ds_weights, ds_maps):
                    ds_loss = ds_loss + w * criterion(ds_logits, masks)
                loss = main_loss + ds_loss
            else:
                loss = main_loss

        running_loss += loss.item()
        num_batches += 1
        metrics_calculator.update(outputs["vessel_prob"], masks)
        pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss, metrics_calculator.compute()
