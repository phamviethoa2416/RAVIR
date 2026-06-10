from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _strip_compile_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {
        key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()
    }


def load_model_state_dict(
    model: nn.Module,
    state_dict: dict,
    *,
    strict: bool = True,
) -> None:
    unwrap_model(model).load_state_dict(
        _strip_compile_prefix(state_dict),
        strict=strict,
    )


def model_weights_for_save(
    model: nn.Module,
    ema: EMAWeights | None = None,
    *,
    prefer_ema: bool = False,
) -> dict[str, torch.Tensor]:
    if prefer_ema and ema is not None:
        return ema.module.state_dict()
    return unwrap_model(model).state_dict()


def build_training_checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler,
    best_dice: float,
    patience: int,
    history: dict[str, list[float]],
    ema: EMAWeights | None = None,
    scaler: GradScaler | None = None,
) -> dict:
    payload: dict = {
        "epoch": epoch,
        "model_state_dict": model_weights_for_save(model, ema, prefer_ema=False),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_dice": best_dice,
        "patience": patience,
        "history": history,
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload
