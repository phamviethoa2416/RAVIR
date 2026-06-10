from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from metrics import SegmentationMetrics
from .ema import EMAWeights


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


def needs_sw() -> bool:
    return Config.IMG_SIZE < Config.ORIGINAL_SIZE


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def sw_inference(
    model: nn.Module,
    image: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 128,
    num_classes: int = 3,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    amp_dtype = get_amp_dtype()
    use_amp = Config.USE_AMP and image.is_cuda
    autocast_device = "cuda" if image.is_cuda else "cpu"

    _, C, H, W = image.shape
    stride = tile_size - overlap

    pad_h = max(0, tile_size - H) if H < tile_size else (stride - H % stride) % stride
    pad_w = max(0, tile_size - W) if W < tile_size else (stride - W % stride) % stride
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, Hp, Wp = image.shape

    acc_seg = torch.zeros(1, num_classes, Hp, Wp, device=image.device)
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
            tile = image[:, :, y : y + tile_size, x : x + tile_size]
            with autocast(
                device_type=autocast_device, dtype=amp_dtype, enabled=use_amp
            ):
                out = model(tile)
            acc_seg[:, :, y : y + tile_size, x : x + tile_size] += (
                out["segmentation"].float() * win
            )
            wsum[:, :, y : y + tile_size, x : x + tile_size] += win

    wc = wsum.clamp(min=1e-6)
    assembled = (acc_seg / wc)[:, :, :H, :W]

    core = unwrap_model(model)
    with torch.no_grad():
        with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            refinement = core.refinement(assembled)
    return assembled.float(), [r.float() for r in refinement]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str,
    grad_accum_steps: int = 1,
    scaler: GradScaler | None = None,
    ema: EMAWeights | None = None,
    scheduler=None,
    scheduler_step: bool = False,
) -> tuple[float, dict[str, float]]:
    model.train()
    running_loss = 0.0
    running_details: dict[str, float] = {}
    num_batches = 0
    optimizer.zero_grad()

    amp_dtype = get_amp_dtype()
    use_amp = Config.USE_AMP and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"

    pbar = tqdm(loader, desc="  Training", leave=False)
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
            loss, details = criterion(outputs, masks)

        scaled_loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            if scheduler is not None and scheduler_step:
                scheduler.step()

        running_loss += loss.item()
        for k, v in details.items():
            running_details[k] = running_details.get(k, 0) + v
        num_batches += 1
        pbar.set_postfix({"Loss": f"{running_loss / num_batches:.4f}"})

    avg_loss = running_loss / max(num_batches, 1)
    avg_details = {k: v / max(num_batches, 1) for k, v in running_details.items()}
    return avg_loss, avg_details


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    metrics_calc: SegmentationMetrics,
) -> tuple[float, dict]:
    model.eval()
    metrics_calc.reset()
    running_loss = 0.0
    running_details: dict[str, float] = {}
    num_batches = 0

    amp_dtype = get_amp_dtype()
    use_amp = Config.USE_AMP and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    use_sw = needs_sw()

    pbar = tqdm(loader, desc="  Validation", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            if use_sw:
                B = images.shape[0]
                base_list: list[torch.Tensor] = []
                refinement_list: list[list[torch.Tensor]] = []
                for b in range(B):
                    base_logits, refinement = sw_inference(
                        model=model,
                        image=images[b : b + 1],
                        tile_size=Config.IMG_SIZE,
                        overlap=Config.IMG_SIZE // 4,
                        num_classes=Config.NUM_CLASSES,
                    )
                    base_list.append(base_logits)
                    refinement_list.append(refinement)

                outputs = {
                    "segmentation": torch.cat(base_list, dim=0),
                    "ds": [],
                    "refinement": [
                        torch.cat(
                            [refinement_list[b][k] for b in range(B)],
                            dim=0,
                        )
                        for k in range(len(refinement_list[0]))
                    ],
                }
            else:
                outputs = model(images)

            loss, details = criterion(outputs, masks)

        running_loss += loss.item()
        for k, v in details.items():
            running_details[k] = running_details.get(k, 0) + v
        num_batches += 1

        refinement = outputs.get("refinement")
        if (
            refinement is not None
            and isinstance(refinement, list)
            and len(refinement) > 0
        ):
            preds = refinement[-1].argmax(dim=1)
        else:
            preds = outputs["segmentation"].argmax(dim=1)
        metrics_calc.update(preds, masks)

        pbar.set_postfix({"Loss": f"{running_loss / num_batches:.4f}"})

    avg_loss = running_loss / max(num_batches, 1)
    avg_details = {k: v / max(num_batches, 1) for k, v in running_details.items()}
    metrics = metrics_calc.compute()
    metrics.update(avg_details)

    return avg_loss, metrics
