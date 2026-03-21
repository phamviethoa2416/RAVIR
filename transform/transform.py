from __future__ import annotations

import random
from typing import Any

import albumentations as alb
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

from config import Config

_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)
_MAX_PIXEL = 255.0


class VesselAwareCrop(alb.DualTransform):
    def __init__(
            self,
            height: int,
            width: int,
            vessel_bias: float = 0.7,
            vessel_labels: tuple[int, ...] = (1, 2),
            always_apply: bool = False,
            p: float = 1.0,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.height = height
        self.width = width
        self.vessel_bias = vessel_bias
        self.vessel_labels = vessel_labels

    def get_params_dependent_on_data(
            self,
            params: dict[str, Any],
            data: dict[str, Any],
    ) -> dict[str, Any]:
        h, w = params["shape"][:2]
        crop_h, crop_w = min(self.height, h), min(self.width, w)

        mask = data.get("mask")

        if mask is not None and random.random() < self.vessel_bias:
            vessel_mask = np.isin(mask, self.vessel_labels)
            ys, xs = np.where(vessel_mask)

            if len(ys) > 0:
                idx = random.randint(0, len(ys) - 1)
                cy, cx = int(ys[idx]), int(xs[idx])

                y1 = np.clip(cy - crop_h // 2, 0, h - crop_h)
                x1 = np.clip(cx - crop_w // 2, 0, w - crop_w)

                return {
                    "y1": int(y1),
                    "x1": int(x1),
                    "crop_h": crop_h,
                    "crop_w": crop_w,
                }

        y1 = random.randint(0, h - crop_h)
        x1 = random.randint(0, w - crop_w)
        return {"y1": y1, "x1": x1, "crop_h": crop_h, "crop_w": crop_w}

    def apply(
            self,
            img: np.ndarray,
            y1: int = 0,
            x1: int = 0,
            crop_h: int = 0,
            crop_w: int = 0,
            **params: Any,
    ) -> np.ndarray:
        return img[y1: y1 + crop_h, x1: x1 + crop_w]

    def apply_to_mask(
            self,
            mask: np.ndarray,
            y1: int = 0,
            x1: int = 0,
            crop_h: int = 0,
            crop_w: int = 0,
            **params: Any,
    ) -> np.ndarray:
        return mask[y1: y1 + crop_h, x1: x1 + crop_w]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "height", "width", "vessel_bias", "vessel_labels"


def get_train_transform() -> alb.Compose:
    transforms: list[alb.BasicTransform] = []

    if Config.IMG_SIZE < Config.ORIGINAL_SIZE:
        transforms.append(
            VesselAwareCrop(
                height=Config.IMG_SIZE,
                width=Config.IMG_SIZE,
                vessel_bias=0.7,
            )
        )

    transforms.extend(
        [
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(
                limit=180,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7,
            ),
            alb.Affine(
                translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                scale=(0.92, 1.08),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3,
            ),
            alb.ElasticTransform(
                alpha=8,
                sigma=4,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.25,
            ),
            alb.OneOf(
                [
                    alb.GridDistortion(
                        num_steps=5,
                        distort_limit=0.15,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=1.0,
                    ),
                    alb.OpticalDistortion(
                        distort_limit=0.15,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=1.0,
                    ),
                ],
                p=0.2,
            ),
        ]
    )

    transforms.extend(
        [
            alb.CLAHE(
                clip_limit=(1.0, 3.0),
                tile_grid_size=(8, 8),
                p=0.5,
            ),
            alb.Sharpen(
                alpha=(0.1, 0.3),
                lightness=(0.9, 1.1),
                p=0.3,
            ),
            alb.RandomGamma(gamma_limit=(80, 120), p=0.4),
            alb.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.12,
                p=0.4,
            ),
            alb.GaussianBlur(
                blur_limit=(3, 3),
                sigma_limit=(0.1, 0.7),
                p=0.15,
            ),
            alb.GaussNoise(
                std_range=(0.01, 0.05),
                p=0.2,
            ),
        ]
    )

    transforms.append(
        alb.CoarseDropout(
            num_holes_range=(2, 4),
            hole_height_range=(8, 20),
            hole_width_range=(8, 20),
            fill="random",
            p=0.2,
        ),
    )

    transforms.extend(
        [
            alb.Normalize(mean=_NORM_MEAN, std=_NORM_STD, max_pixel_value=_MAX_PIXEL),
            ToTensorV2(),
        ]
    )

    return alb.Compose(
        transforms,
        additional_targets={"skeleton": "mask"},
    )


def get_val_transform() -> alb.Compose:
    return alb.Compose(
        [
            alb.Normalize(mean=_NORM_MEAN, std=_NORM_STD, max_pixel_value=_MAX_PIXEL),
            ToTensorV2(),
        ],
        additional_targets={"skeleton": "mask"},
    )


def get_test_transform() -> alb.Compose:
    return alb.Compose(
        [
            alb.Normalize(mean=_NORM_MEAN, std=_NORM_STD, max_pixel_value=_MAX_PIXEL),
            ToTensorV2(),
        ],
        additional_targets={"skeleton": "mask"},
    )
