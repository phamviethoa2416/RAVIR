from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from skimage.filters import frangi
from torch.utils.data import Dataset

from config import Config

if Config.USE_FRANGI:
    NORM_MEAN = np.array(
        [0.485, 0.456, 0.406, Config.FRANGI_NORM_MEAN],
        dtype=np.float32,
    )
    NORM_STD = np.array(
        [0.229, 0.224, 0.225, Config.FRANGI_NORM_STD],
        dtype=np.float32,
    )
else:
    NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def mask_to_class(mask: np.ndarray) -> np.ndarray:
    class_mask = np.zeros_like(mask, dtype=np.int32)
    for pixel, idx in Config.MASK_PIXEL_VALUES.items():
        class_mask[mask == pixel] = idx
    return class_mask


def compute_class_weights(
    mask_dir: str,
    file_list: list[str],
    min_weight: float = 0.5,
    max_weight: float = 5.0,
) -> torch.Tensor:
    class_counts = np.zeros(Config.NUM_CLASSES, dtype=np.float64)

    for filename in file_list:
        mask = np.array(
            Image.open(os.path.join(mask_dir, filename)).convert("L"),
        )
        class_mask = mask_to_class(mask)
        for c in range(Config.NUM_CLASSES):
            class_counts[c] += int((class_mask == c).sum())

    freq = class_counts / (class_counts.sum() + 1e-6)
    median_freq = np.median(freq)
    weights = np.sqrt(median_freq / (freq + 1e-6))
    weights = np.clip(weights, a_min=min_weight, a_max=max_weight)

    return torch.tensor(weights, dtype=torch.float32)


def compute_frangi(
    gray: np.ndarray,
    sigmas: tuple[int, ...] = (1, 2, 3, 4),
    black_ridges: bool = True,
) -> np.ndarray:
    response = frangi(
        gray.astype(np.float64) / 255.0,
        sigmas=sigmas,
        black_ridges=black_ridges,
    )
    vmax = response.max()
    if vmax > 0:
        response /= vmax
    return response.astype(np.float32)


class RAVIRDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str | None = None,
        file_list: list[str] | None = None,
        transform=None,
        is_test: bool = False,
        frangi_cache_dir: str | None = None,
        use_rotation_expansion: bool = False,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        self.frangi_cache_dir = frangi_cache_dir

        self.file_list = (
            sorted(file_list)
            if file_list
            else sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
        )

        self.use_rotation_expansion = bool(use_rotation_expansion) and not is_test
        if self.use_rotation_expansion:
            self.samples: list[tuple[str, int]] = [
                (filename, rotation_k)
                for filename in self.file_list
                for rotation_k in range(4)
            ]
        else:
            self.samples = [(filename, 0) for filename in self.file_list]

        if Config.USE_FRANGI and frangi_cache_dir:
            os.makedirs(frangi_cache_dir, exist_ok=True)
            self.precompute_frangi()

    def precompute_frangi(self) -> None:
        for filename in self.file_list:
            stem = os.path.splitext(filename)[0]
            cache_path = os.path.join(self.frangi_cache_dir, f"{stem}_frangi.npy")
            if os.path.exists(cache_path):
                continue
            gray = np.array(
                Image.open(os.path.join(self.img_dir, filename)).convert("L"),
                dtype=np.uint8,
            )
            fmap = compute_frangi(
                gray,
                sigmas=Config.FRANGI_SIGMAS,
                black_ridges=Config.FRANGI_BLACK_RIDGES,
            )
            np.save(cache_path, fmap)

    def load_frangi(self, filename: str, gray: np.ndarray) -> np.ndarray:
        if self.frangi_cache_dir:
            stem = os.path.splitext(filename)[0]
            cache_path = os.path.join(self.frangi_cache_dir, f"{stem}_frangi.npy")
            if os.path.exists(cache_path):
                return np.load(cache_path)
        return compute_frangi(
            gray,
            sigmas=Config.FRANGI_SIGMAS,
            black_ridges=Config.FRANGI_BLACK_RIDGES,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict:
        filename, rotation_k = self.samples[item]

        gray = np.array(
            Image.open(os.path.join(self.img_dir, filename)).convert("L"),
            dtype=np.uint8,
        )

        image = np.stack([gray, gray, gray], axis=-1)

        frangi_map: np.ndarray | None = None
        if Config.USE_FRANGI:
            frangi_map = self.load_frangi(filename, gray)

        if not self.is_test and self.mask_dir is not None:
            raw_mask = np.array(
                Image.open(os.path.join(self.mask_dir, filename)).convert("L"),
                dtype=np.uint8,
            )
            mask = mask_to_class(raw_mask)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.int32)

        if rotation_k != 0:
            image = np.ascontiguousarray(np.rot90(image, k=rotation_k, axes=(0, 1)))
            mask = np.ascontiguousarray(np.rot90(mask, k=rotation_k, axes=(0, 1)))

            if frangi_map is not None:
                frangi_map = np.ascontiguousarray(
                    np.rot90(frangi_map, k=rotation_k, axes=(0, 1))
                )

        if self.transform:
            transform_kwargs: dict = {
                "image": image,
                "mask": mask,
            }

            if frangi_map is not None:
                transform_kwargs["frangi"] = frangi_map

            augmented = self.transform(**transform_kwargs)
            image = augmented["image"]
            mask = augmented["mask"]
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)
            mask = mask.long()

            if Config.USE_FRANGI and frangi_map is not None:
                frangi_t = augmented["frangi"].float()
                frangi_t = (frangi_t - Config.FRANGI_NORM_MEAN) / Config.FRANGI_NORM_STD
                image = torch.cat([image, frangi_t.unsqueeze(0)], dim=0)
        else:
            rgb = image.astype(np.float32) / 255.0
            rgb = (rgb - NORM_MEAN[:3]) / NORM_STD[:3]

            if Config.USE_FRANGI and frangi_map is not None:
                frangi_norm = (frangi_map - Config.FRANGI_NORM_MEAN) / Config.FRANGI_NORM_STD
                image = np.concatenate([rgb, frangi_norm[..., np.newaxis]], axis=-1)
            else:
                image = rgb

            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "filename": filename,
        }
