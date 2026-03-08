from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import Config
from metrics import SegmentationMetrics


def get_amp_dtype() -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(Config.AMP_DTYPE, torch.float32)


def create_scaler() -> GradScaler | None:
    if Config.USE_AMP and Config.AMP_DTYPE == "float16":
        return GradScaler("cuda")
    return None


def needs_sliding_window() -> bool:
    return Config.IMG_SIZE < Config.ORIGINAL_SIZE

# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(
        model: nn.Module,
        loader,
        criterion: nn.Module,
        optimizer,
        device: str,
        grad_accum_steps: int = 1,
        scaler: GradScaler | None = None,
) -> tuple[float, dict[str, float]]:
    model.train()
    running_loss = 0.0
    running_details: dict[str, float] = {}
    num_batches = 0
    optimizer.zero_grad()

    amp_dtype = get_amp_dtype()
    use_amp = Config.USE_AMP and device == "cuda"

    pbar = tqdm(loader, desc="  Train", leave=False)
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        vessel_prob = batch["vessel_prob"].to(device, non_blocking=True)

        targets = {"mask": masks, "vessel_prob": vessel_prob}

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
            loss, details = criterion(outputs, targets)

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
        for k, v in details.items():
            running_details[k] = running_details.get(k, 0.0) + v
        num_batches += 1
        pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

    avg_loss = running_loss / max(num_batches, 1)
    avg_details = {k: v / max(num_batches, 1) for k, v in running_details.items()}
    return avg_loss, avg_details

def sliding_window_inference(
        model: nn.Module,
        image: torch.Tensor,
        tile_size: int = 512,
        overlap: int = 128,
        num_classes: int = 3,
) -> dict[str, torch.Tensor]:
    amp_dtype = get_amp_dtype()
    use_amp = Config.USE_AMP and image.is_cuda

    _, C, H, W = image.shape
    stride = tile_size - overlap

    pad_h = max(0, tile_size - H) if H < tile_size else (stride - H % stride) % stride
    pad_w = max(0, tile_size - W) if W < tile_size else (stride - W % stride) % stride
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = image.shape

    acc_seg = torch.zeros(1, num_classes, Hp, Wp, device=image.device)
    acc_vp = torch.zeros(1, 1, Hp, Wp, device=image.device)
    wsum = torch.zeros(1, 1, Hp, Wp, device=image.device)

    ramp = torch.hann_window(tile_size, periodic=False, device=image.device)
    win = (ramp.unsqueeze(0) * ramp.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

    ys = list(range(0, Hp - tile_size + 1, stride))
    xs = list(range(0, Wp - tile_size + 1, stride))
    if not ys or ys[-1] + tile_size < Hp:
        ys.append(Hp - tile_size)
    if not xs or xs[-1] + tile_size < Wp:
        xs.append(Wp - tile_size)

    for y in ys:
        for x in xs:
            tile = image[:, :, y:y + tile_size, x:x + tile_size]
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(tile)
            acc_seg[:, :, y:y + tile_size, x:x + tile_size] += out["segmentation"].float() * win
            acc_vp[:, :, y:y + tile_size, x:x + tile_size] += out["vessel_prob"].float() * win
            wsum[:, :, y:y + tile_size, x:x + tile_size] += win

    wc = wsum.clamp(min=1e-6)
    return {
        "segmentation": (acc_seg / wc)[:, :, :H, :W],
        "vessel_prob": (acc_vp / wc)[:, :, :H, :W],
    }


# ── Validation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(
        model: nn.Module,
        loader,
        criterion: nn.Module,
        device: str,
        metrics_calculator: SegmentationMetrics,
) -> tuple[float, dict]:
    model.eval()
    metrics_calculator.reset()
    running_loss = 0.0
    running_details: dict[str, float] = {}
    num_batches = 0

    amp_dtype = get_amp_dtype()
    use_amp = Config.USE_AMP and device == "cuda"
    use_sw = needs_sliding_window()

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        vessel_prob = batch["vessel_prob"].to(device, non_blocking=True)

        targets = {"mask": masks, "vessel_prob": vessel_prob}

        if use_sw:
            B = images.shape[0]
            merged: dict[str, list] = {}
            for b in range(B):
                tile_out = sliding_window_inference(
                    model, images[b:b + 1],
                    tile_size=Config.IMG_SIZE,
                    overlap=Config.IMG_SIZE // 4,
                    num_classes=Config.NUM_CLASSES,
                )
                for k, v in tile_out.items():
                    merged.setdefault(k, []).append(v)
            outputs = {k: torch.cat(v, dim=0) for k, v in merged.items()}
        else:
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            loss, details = criterion(outputs, targets)

        running_loss += loss.item()
        for k, v in details.items():
            running_details[k] = running_details.get(k, 0.0) + v
        num_batches += 1

        preds = torch.argmax(outputs["segmentation"], dim=1)
        metrics_calculator.update(preds, masks)
        pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

    avg_loss = running_loss / max(num_batches, 1)
    avg_details = {k: v / max(num_batches, 1) for k, v in running_details.items()}
    metrics = metrics_calculator.compute()
    metrics.update(avg_details)
    return avg_loss, metrics


# ── Binary Training (Curriculum Learning) ─────────────────────────────────────


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
            loss = criterion(outputs["vessel_prob"], masks)

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
            loss = criterion(outputs["vessel_prob"], masks)

        running_loss += loss.item()
        num_batches += 1
        metrics_calculator.update(outputs["vessel_prob"], masks)
        pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss, metrics_calculator.compute()