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
from .helpers import unwrap_model


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


def _sw_padding(height: int, width: int, tile_size: int, stride: int) -> tuple[int, int]:
    pad_h = (
        max(0, tile_size - height)
        if height < tile_size
        else (stride - height % stride) % stride
    )
    pad_w = (
        max(0, tile_size - width)
        if width < tile_size
        else (stride - width % stride) % stride
    )
    return pad_h, pad_w


def _sw_starts(padded_size: int, tile_size: int, stride: int) -> list[int]:
    starts = list(range(0, padded_size - tile_size + 1, stride))
    if not starts or starts[-1] + tile_size < padded_size:
        starts.append(padded_size - tile_size)
    return starts


def _sw_blend_window(tile_size: int, device: torch.device) -> torch.Tensor:
    ramp = torch.hann_window(tile_size, periodic=False, device=device).clamp_min(1e-3)
    return (ramp.unsqueeze(0) * ramp.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def _pad_image_for_sw(
    image: torch.Tensor, pad_h: int, pad_w: int
) -> torch.Tensor:
    if not (pad_h or pad_w):
        return image
    mode = (
        "reflect"
        if pad_h < image.shape[-2] and pad_w < image.shape[-1]
        else "replicate"
    )
    return F.pad(image, (0, pad_w, 0, pad_h), mode=mode)


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

    pad_h, pad_w = _sw_padding(H, W, tile_size, stride)
    image = _pad_image_for_sw(image, pad_h, pad_w)

    _, _, Hp, Wp = image.shape

    acc_seg = torch.zeros(1, num_classes, Hp, Wp, device=image.device)
    wsum = torch.zeros(1, 1, Hp, Wp, device=image.device)

    win = _sw_blend_window(tile_size, image.device)
    ys = _sw_starts(Hp, tile_size, stride)
    xs = _sw_starts(Wp, tile_size, stride)

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
    refinement_head = getattr(core, "refinement", None)
    if refinement_head is None:
        return assembled.float(), []
    with torch.no_grad():
        with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            refinement = refinement_head(assembled)
    return assembled.float(), [r.float() for r in refinement]


def sw_loss(
    model: nn.Module,
    criterion: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    branch_labels: torch.Tensor | None = None,
    frangi_target: torch.Tensor | None = None,
    tile_size: int = 512,
    overlap: int = 128,
) -> tuple[float, dict[str, float]]:
    _, _, H, W = images.shape
    stride = tile_size - overlap
    pad_h, pad_w = _sw_padding(H, W, tile_size, stride)

    images_p = _pad_image_for_sw(images, pad_h, pad_w)
    masks_p = F.pad(masks, (0, pad_w, 0, pad_h), mode="constant", value=0)
    branch_p = (
        F.pad(branch_labels, (0, pad_w, 0, pad_h), mode="constant", value=0)
        if branch_labels is not None
        else None
    )
    frangi_p = (
        F.pad(frangi_target, (0, pad_w, 0, pad_h), mode="constant", value=0)
        if frangi_target is not None
        else None
    )

    _, _, Hp, Wp = images_p.shape
    ys = _sw_starts(Hp, tile_size, stride)
    xs = _sw_starts(Wp, tile_size, stride)

    running_loss = 0.0
    running_details: dict[str, float] = {}
    num_tiles = 0

    for y in ys:
        for x in xs:
            tile_outputs = model(images_p[:, :, y : y + tile_size, x : x + tile_size])
            tile_branch = (
                branch_p[:, y : y + tile_size, x : x + tile_size]
                if branch_p is not None
                else None
            )
            tile_frangi = (
                frangi_p[:, :, y : y + tile_size, x : x + tile_size]
                if frangi_p is not None
                else None
            )
            tile_loss, tile_details = criterion(
                tile_outputs,
                masks_p[:, y : y + tile_size, x : x + tile_size],
                branch_labels=tile_branch,
                frangi_target=tile_frangi,
            )
            running_loss += tile_loss.item()
            for k, v in tile_details.items():
                running_details[k] = running_details.get(k, 0.0) + v
            num_tiles += 1

    denom = max(num_tiles, 1)
    return running_loss / denom, {k: v / denom for k, v in running_details.items()}


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
        branch_labels = batch.get("branch_labels")
        if branch_labels is not None:
            branch_labels = branch_labels.to(device, non_blocking=True)
        frangi_target = batch.get("frangi_target")
        if frangi_target is not None:
            frangi_target = frangi_target.to(device, non_blocking=True)

        with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
            loss, details = criterion(
                outputs, masks, branch_labels=branch_labels, frangi_target=frangi_target
            )

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
        branch_labels = batch.get("branch_labels")
        if branch_labels is not None:
            branch_labels = branch_labels.to(device, non_blocking=True)
        frangi_target = batch.get("frangi_target")
        if frangi_target is not None:
            frangi_target = frangi_target.to(device, non_blocking=True)

        with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            if use_sw:
                loss_value, details = sw_loss(
                    model=model,
                    criterion=criterion,
                    images=images,
                    masks=masks,
                    branch_labels=branch_labels,
                    frangi_target=frangi_target,
                    tile_size=Config.IMG_SIZE,
                    overlap=Config.IMG_SIZE // 4,
                )
                loss = images.new_tensor(loss_value)

                B = images.shape[0]
                segmentation: list[torch.Tensor] = []
                refinement_list: list[list[torch.Tensor]] = []
                for b in range(B):
                    segmentation_logits, refinement = sw_inference(
                        model=model,
                        image=images[b : b + 1],
                        tile_size=Config.IMG_SIZE,
                        overlap=Config.IMG_SIZE // 4,
                        num_classes=Config.NUM_CLASSES,
                    )
                    segmentation.append(segmentation_logits)
                    refinement_list.append(refinement)

                outputs = {
                    "segmentation": torch.cat(segmentation, dim=0),
                    "ds": [],
                    "refinement": [
                        torch.cat(
                            [refinement_list[b][k] for b in range(B)],
                            dim=0,
                        )
                        for k in range(len(refinement_list[0]))
                    ],
                    "embedding": None,
                    "frangi_recon_logits": None,
                }
            else:
                outputs = model(images)
                loss, details = criterion(
                    outputs,
                    masks,
                    branch_labels=branch_labels,
                    frangi_target=frangi_target,
                )

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
