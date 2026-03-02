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
        model,
        loader,
        criterion: nn.Module,
        optimizer,
        device: str,
        grad_accum_steps: int = 1,
        class_weights: torch.Tensor = None,
) -> float:
    model.train()

    running_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  Train", leave=False)
    for step, (images, masks, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        output = model(images)
        logits = output[0] if isinstance(output, (tuple, list)) else output

        if Config.USE_DYNAMIC_WEIGHTS:
            batch_weights = _compute_batch_weights(
                masks, Config.NUM_CLASSES
            ).to(device)
        else:
            batch_weights = class_weights.to(device) if class_weights is not None else None

        loss = criterion(logits, masks, class_weights=batch_weights)
        loss_scaled = loss / grad_accum_steps

        loss_scaled.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


def sliding_window_inference(
        model,
        image: torch.Tensor,
        tile_size: int = 256,
        overlap: int = 64,
        num_classes: int = 3,
) -> torch.Tensor:
    _, C, H, W = image.shape
    stride = tile_size - overlap

    pad_h = (
        max(0, tile_size - H)
        if H < tile_size
        else (stride - H % stride) % stride
    )
    pad_w = (
        max(0, tile_size - W)
        if W < tile_size
        else (stride - W % stride) % stride
    )
    if pad_h or pad_w:
        image = torch.nn.functional.pad(
            image, (0, pad_w, 0, pad_h), mode="reflect"
        )
    _, _, H_pad, W_pad = image.shape

    logit_sum = torch.zeros(1, num_classes, H_pad, W_pad, device=image.device)
    weight_sum = torch.zeros(1, 1, H_pad, W_pad, device=image.device)

    ramp = torch.hann_window(tile_size, periodic=False, device=image.device)
    win = (ramp.unsqueeze(0) * ramp.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

    ys = list(range(0, H_pad - tile_size + 1, stride))
    xs = list(range(0, W_pad - tile_size + 1, stride))
    if not ys or ys[-1] + tile_size < H_pad:
        ys.append(H_pad - tile_size)
    if not xs or xs[-1] + tile_size < W_pad:
        xs.append(W_pad - tile_size)

    for y in ys:
        for x in xs:
            tile = image[:, :, y:y + tile_size, x:x + tile_size]
            output = model(tile)
            tile_logits = output[0] if isinstance(output, (tuple, list)) else output

            logit_sum[:, :, y:y + tile_size, x:x + tile_size] += tile_logits * win
            weight_sum[:, :, y:y + tile_size, x:x + tile_size] += win

    logits_full = logit_sum / weight_sum.clamp(min=1e-6)
    return logits_full[:, :, :H, :W]


@torch.no_grad()
def validate(
        model,
        loader,
        criterion: nn.Module,
        device: str,
        metrics_calculator: SegmentationMetrics,
        class_weights: torch.Tensor = None,
) -> tuple[float, dict]:
    model.eval()
    metrics_calculator.reset()

    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        B = images.shape[0]
        all_logits = []
        for b in range(B):
            tile_logits = sliding_window_inference(
                model,
                images[b:b + 1],
                tile_size=Config.IMG_SIZE,
                overlap=Config.IMG_SIZE // 4,
                num_classes=Config.NUM_CLASSES,
            )
            all_logits.append(tile_logits)
        logits = torch.cat(all_logits, dim=0)

        loss = criterion(logits, masks, class_weights=class_weights)

        running_loss += loss.item()
        num_batches += 1

        preds = torch.argmax(logits, dim=1)
        metrics_calculator.update(preds, masks)
        pbar.set_postfix({
            "loss": f"{running_loss / num_batches:.4f}",
        })

    avg_loss = running_loss / max(num_batches, 1)
    metrics = metrics_calculator.compute()

    return avg_loss, metrics
