import random
from typing import Callable, Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations import ToTensorV2

from config import Config

MAX_PIXEL = 255.0
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

ADDITIONAL_TARGETS = {"frangi": "mask"}


class VesselAwareCrop(A.DualTransform):

    def __init__(
        self,
        height: int,
        width: int,
        vessel_bias: float = 0.7,
        vessel_labels: tuple[int, ...] = (1, 2),
        p: float = 1.0,
    ):
        super().__init__(p)
        self.height = height
        self.width = width
        self.vessel_bias = vessel_bias
        self.vessel_labels = vessel_labels

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]
        crop_height = min(self.height, height)
        crop_width = min(self.width, width)
        mask = data.get("mask")

        if mask is not None and random.random() < self.vessel_bias:
            vessel_mask = np.isin(mask, self.vessel_labels)
            ys, xs = np.where(vessel_mask)

            if len(ys) > 0:
                index = random.randint(0, len(ys) - 1)
                cy, cx = int(ys[index]), int(xs[index])

                y1 = np.clip(cy - crop_height // 2, 0, height - crop_height)
                x1 = np.clip(cx - crop_width // 2, 0, width - crop_width)

                return {
                    "y1": int(y1),
                    "x1": int(x1),
                    "crop_height": crop_height,
                    "crop_width": crop_width,
                }

        y1 = random.randint(0, height - crop_height)
        x1 = random.randint(0, width - crop_width)

        return {
            "y1": y1,
            "x1": x1,
            "crop_height": crop_height,
            "crop_width": crop_width,
        }

    def apply(
        self,
        img: np.ndarray,
        y1: int = 0,
        x1: int = 0,
        crop_height: int = 0,
        crop_width: int = 0,
        **params: Any,
    ) -> np.ndarray:
        return img[y1 : y1 + crop_height, x1 : x1 + crop_width]

    def apply_to_mask(
        self,
        mask: np.ndarray,
        y1: int = 0,
        x1: int = 0,
        crop_height: int = 0,
        crop_width: int = 0,
        **params: Any,
    ) -> np.ndarray:
        return mask[y1 : y1 + crop_height, x1 : x1 + crop_width]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "height", "width", "vessel_bias", "vessel_labels"


def get_train_transform() -> A.Compose:
    transforms: list[A.BasicTransform] = []

    if Config.IMG_SIZE < Config.ORIGINAL_SIZE:
        transforms.append(
            VesselAwareCrop(
                height=Config.IMG_SIZE,
                width=Config.IMG_SIZE,
                vessel_bias=0.7,
            )
        )

    # ── Spatial geometric ──────────────────────────────────────────
    transforms.extend(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7,
            ),
            A.Affine(
                translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                scale=(0.85, 1.15),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.4,
            ),
        ]
    )

    # ── Photometric ────────────────────────────────────────────────
    transforms.extend(
        [
            A.CLAHE(
                clip_limit=(1.0, 2.0),
                tile_grid_size=(8, 8),
                p=0.5,
            ),
            A.RandomGamma(gamma_limit=(80, 130), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.4
            ),
        ]
    )

    # ── Noise / blur  ──────────────────────────
    transforms.extend(
        [
            A.GaussianBlur(
                blur_limit=(3, 5),
                sigma_limit=(0.1, 0.6),
                p=0.15,
            ),
            A.GaussNoise(
                std_range=(0.01, 0.035),
                p=0.20,
            ),
        ]
    )

    transforms.extend(
        [
            A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=MAX_PIXEL),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms, additional_targets=ADDITIONAL_TARGETS)


def get_val_transform() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=MAX_PIXEL),
            ToTensorV2(),
        ],
        additional_targets=ADDITIONAL_TARGETS,
    )


def get_test_transform() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=MAX_PIXEL),
            ToTensorV2(),
        ],
        additional_targets=ADDITIONAL_TARGETS,
    )


def get_tta_transform() -> list[tuple[str, A.Compose]]:
    base = [
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=MAX_PIXEL),
        ToTensorV2(),
    ]

    def _make(spatial: list[A.BasicTransform], name: str) -> tuple[str, A.Compose]:
        return name, A.Compose(
            spatial + base,
            additional_targets=ADDITIONAL_TARGETS,
        )

    return [
        _make([], "identity"),
        _make([A.HorizontalFlip(p=1.0)], "hflip"),
        _make([A.VerticalFlip(p=1.0)], "vflip"),
        _make([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], "hflip_vflip"),
        _make([A.Rotate(limit=(90, 90), p=1.0)], "rot90"),
        _make([A.Rotate(limit=(180, 180), p=1.0)], "rot180"),
        _make([A.Rotate(limit=(270, 270), p=1.0)], "rot270"),
        _make(
            [A.HorizontalFlip(p=1.0), A.Rotate(limit=(90, 90), p=1.0)],
            "hflip_rot90",
        ),
    ]


def tta_inverse_transform() -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:

    def _rot90_inv(t):
        return torch.rot90(t, k=-1, dims=(-2, -1))

    return {
        "identity": lambda t: t,
        "hflip": lambda t: t.flip(-1),
        "vflip": lambda t: t.flip(-2),
        "hflip_vflip": lambda t: t.flip(-1).flip(-2),
        "rot90": _rot90_inv,
        "rot180": lambda t: torch.rot90(t, k=2, dims=(-2, -1)),
        "rot270": lambda t: torch.rot90(t, k=1, dims=(-2, -1)),
        "hflip_rot90": lambda t: _rot90_inv(t).flip(-1),
    }
