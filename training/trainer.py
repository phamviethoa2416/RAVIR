import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from metrics import SegmentationMetrics


def _compute_batch_weights(masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    total = masks.numel()
    for c in range(num_classes):
        counts[c] = (masks == c).sum().float()
    freq = counts / (total + 1e-6)
    median_freq = freq.median()
    weights = median_freq / (freq + 1e-6)
    weights = weights.clamp(min=1.0, max=10.0)
    return weights


def train_one_epoch(
        model, loader, criterion, optimizer, device,
        grad_accum_steps=1, class_weights=None,
):
    model.train()
    running_loss = 0.0
    running_details: dict[str, float] = {}
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  Train", leave=False)
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        outputs = model(images)

        if Config.USE_DYNAMIC_WEIGHTS:
            bw = _compute_batch_weights(masks, Config.NUM_CLASSES).to(device)
        else:
            bw = class_weights.to(device) if class_weights is not None else None

        targets = {
            "mask": masks,
            "vessel_prob": batch["vessel_prob"].to(device, non_blocking=True),
            "orientation": batch["orientation"].to(device, non_blocking=True),
            "width": batch["width"].to(device, non_blocking=True),
            "endpoint": batch["endpoint"].to(device, non_blocking=True),
        }
        loss, details = criterion(outputs, targets, class_weights=bw)
        (loss / grad_accum_steps).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        for k, v in details.items():
            running_details[k] = running_details.get(k, 0.0) + v
        num_batches += 1

    avg = running_loss / max(num_batches, 1)
    avg_d = {k: v / max(num_batches, 1) for k, v in running_details.items()}
    return avg, avg_d


def sliding_window_inference(model, image, tile_size=256, overlap=64, num_classes=3):
    _, C, H, W = image.shape
    stride = tile_size - overlap

    pad_h = max(0, tile_size - H) if H < tile_size else (stride - H % stride) % stride
    pad_w = max(0, tile_size - W) if W < tile_size else (stride - W % stride) % stride
    if pad_h or pad_w:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = image.shape

    # accumulators
    heads = {
        "segmentation": torch.zeros(1, num_classes, Hp, Wp, device=image.device),
        "vessel_prob": torch.zeros(1, 1, Hp, Wp, device=image.device),
        "orientation": torch.zeros(1, 2, Hp, Wp, device=image.device),
        "width": torch.zeros(1, 2, Hp, Wp, device=image.device),
        "endpoint": torch.zeros(1, 1, Hp, Wp, device=image.device),
    }
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
            out = model(tile)
            for k in heads:
                heads[k][:, :, y:y + tile_size, x:x + tile_size] += out[k] * win
            wsum[:, :, y:y + tile_size, x:x + tile_size] += win

    wc = wsum.clamp(min=1e-6)
    return {k: (v / wc)[:, :, :H, :W] for k, v in heads.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, metrics_calculator, class_weights=None):
    model.eval()
    metrics_calculator.reset()
    running_loss = 0.0
    running_details: dict[str, float] = {}
    num_batches = 0

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        B = images.shape[0]
        merged = {}
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

        targets = {
            "mask": masks,
            "vessel_prob": batch["vessel_prob"].to(device, non_blocking=True),
            "orientation": batch["orientation"].to(device, non_blocking=True),
            "width": batch["width"].to(device, non_blocking=True),
            "endpoint": batch["endpoint"].to(device, non_blocking=True),
        }
        loss, details = criterion(outputs, targets, class_weights=class_weights)

        running_loss += loss.item()
        for k, v in details.items():
            running_details[k] = running_details.get(k, 0.0) + v
        num_batches += 1

        preds = torch.argmax(outputs["segmentation"], dim=1)
        metrics_calculator.update(preds, masks)
        pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

    avg = running_loss / max(num_batches, 1)
    avg_d = {k: v / max(num_batches, 1) for k, v in running_details.items()}
    metrics = metrics_calculator.compute()
    metrics.update(avg_d)
    return avg, metrics
