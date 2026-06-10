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
CLAHE = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
FINALIZE = [
    A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=MAX_PIXEL),
    ToTensorV2(),
]


class VesselAwareCrop(A.DualTransform):

    def __init__(
        self,
        height: int,
        width: int,
        vessel_bias: float = 0.85,
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
                idx = random.randint(0, len(ys) - 1)
                cy, cx = int(ys[idx]), int(xs[idx])
                y1 = int(np.clip(cy - crop_height // 2, 0, height - crop_height))
                x1 = int(np.clip(cx - crop_width // 2, 0, width - crop_width))
                return {
                    "y1": y1,
                    "x1": x1,
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

    def apply(self, img, y1=0, x1=0, crop_height=0, crop_width=0, **params):
        return img[y1 : y1 + crop_height, x1 : x1 + crop_width]

    def apply_to_mask(self, mask, y1=0, x1=0, crop_height=0, crop_width=0, **params):
        return mask[y1 : y1 + crop_height, x1 : x1 + crop_width]

    def get_transform_init_args_names(self):
        return "height", "width", "vessel_bias", "vessel_labels"


def build_light_transforms() -> list[A.BasicTransform]:
    transforms: list[A.BasicTransform] = []

    if Config.IMG_SIZE < Config.ORIGINAL_SIZE:
        transforms.append(
            VesselAwareCrop(Config.IMG_SIZE, Config.IMG_SIZE, vessel_bias=0.7)
        )

    transforms.extend(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
        ]
    )

    return transforms


def build_full_transforms() -> list[A.BasicTransform]:
    transforms: list[A.BasicTransform] = []

    if Config.IMG_SIZE < Config.ORIGINAL_SIZE:
        transforms.append(
            VesselAwareCrop(
                height=Config.IMG_SIZE, width=Config.IMG_SIZE, vessel_bias=0.8
            )
        )

    # ── Geometric ─────────────────────────────────────────────────
    transforms.extend(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.6,
            ),
            A.Affine(
                translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                scale=(0.9, 1.1),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.35,
            ),
        ]
    )

    # ── Photometric ─────────────────────────────────────────────────
    transforms.extend(
        [
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.12,
                p=0.4,
            ),
        ]
    )

    transforms.append(A.GaussNoise(std_range=(0.01, 0.03), p=0.15))

    return transforms


def get_train_transform(intensity: str = "full") -> A.Compose:
    if intensity not in ("full", "light"):
        raise ValueError(f"intensity must be 'full' or 'light', got {intensity!r}")

    augmentation = (
        build_full_transforms() if intensity == "full" else build_light_transforms()
    )

    transforms = [CLAHE] + augmentation + FINALIZE
    return A.Compose(transforms, additional_targets=ADDITIONAL_TARGETS)


def get_val_transform() -> A.Compose:
    return A.Compose([CLAHE] + FINALIZE, additional_targets=ADDITIONAL_TARGETS)


def get_test_transform() -> A.Compose:
    return A.Compose([CLAHE] + FINALIZE, additional_targets=ADDITIONAL_TARGETS)


def get_tta_transform() -> list[tuple[str, A.Compose]]:
    def _make(spatial: list[A.BasicTransform], name: str) -> tuple[str, A.Compose]:
        return name, A.Compose(
            spatial + [CLAHE] + FINALIZE,
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
            [A.HorizontalFlip(p=1.0), A.Rotate(limit=(90, 90), p=1.0)], "hflip_rot90"
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
