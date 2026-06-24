from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Iterator

import torch
import torch.nn as nn

from .helpers import unwrap_model


class EMAWeights:
    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.9999,
        warmup: int = 2000,
        device: str | None = None,
    ):
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"Decay must be between 0 and 1, got {decay}")

        self.max_decay = float(decay)
        self.warmup = max(0, int(warmup))
        self.step = 0

        self.module: nn.Module = deepcopy(unwrap_model(model))
        if device is not None:
            self.module.to(device)

        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()

    @property
    def current_decay(self) -> float:
        if self.warmup == 0:
            return self.max_decay
        return self.max_decay * min(1.0, self.step / float(self.warmup))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.current_decay
        msd = unwrap_model(model).state_dict()
        esd = self.module.state_dict()
        for k, v in msd.items():
            v_ema = esd[k]
            if v.dtype.is_floating_point:
                v_ema.mul_(decay).add_(v.detach(), alpha=1.0 - decay)
            else:
                v_ema.copy_(v)
        self.step += 1

    @contextmanager
    def swap(self, model: nn.Module) -> Iterator[nn.Module]:
        unwrapped = unwrap_model(model)
        backup = {k: v.detach().clone() for k, v in unwrapped.state_dict().items()}
        unwrapped.load_state_dict(self.module.state_dict(), strict=True)
        try:
            yield model
        finally:
            unwrapped.load_state_dict(backup, strict=True)

    def state_dict(self) -> dict:
        return {
            "module": self.module.state_dict(),
            "step": self.step,
            "max_decay": self.max_decay,
            "warmup": self.warmup,
        }

    def load_state_dict(
        self,
        state_dict: dict,
        strict: bool = True,
    ) -> None:
        if "module" in state_dict and "step" in state_dict:
            self.module.load_state_dict(state_dict["module"], strict=strict)
            self.step = int(state_dict.get("step", 0))
            if "max_decay" in state_dict:
                self.max_decay = float(state_dict["max_decay"])
            if "warmup" in state_dict:
                self.warmup = int(state_dict["warmup"])
        else:
            self.module.load_state_dict(state_dict, strict=strict)
