from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize, disk
from torch.utils.data import Dataset

from config import Config


def mask_to_class(mask_np: np.ndarray) -> np.ndarray:
    class_mask = np.zeros_like(mask_np, dtype=np.int32)
    for pixel_value, class_idx in Config.MASK_PIXEL_VALUES.items():
        class_mask[mask_np == pixel_value] = class_idx
    return class_mask


def grayscale_to_3ch(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    return image


def compute_skeleton(
        class_mask: np.ndarray,
        vessel_classes: tuple[int, ...] = (1, 2),
        tube_radius: int = 1,
) -> np.ndarray:
    skeleton = np.zeros_like(class_mask, dtype=np.int32)
    selem = disk(tube_radius) if tube_radius > 0 else None

    for c in vessel_classes:
        binary = (class_mask == c).astype(np.uint8)
        if binary.sum() == 0:
            continue
        skel = skeletonize(binary > 0)
        if selem is not None:
            skel = binary_dilation(skel, selem)
        skeleton[skel > 0] = c

    return skeleton


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


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class RAVIRDataset(Dataset):
    def __init__(
            self,
            img_dir: str,
            mask_dir: str | None = None,
            file_list: list[str] | None = None,
            transform=None,
            is_test: bool = False,
            skeleton_cache_dir: str | None = None,
            tube_radius: int = 1,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        self.skeleton_cache_dir = skeleton_cache_dir
        self.tube_radius = tube_radius
        self.file_list = sorted(file_list) if file_list else sorted(os.listdir(img_dir))

        if skeleton_cache_dir and mask_dir and not is_test:
            os.makedirs(skeleton_cache_dir, exist_ok=True)
            self._precompute_skeletons()

    def _precompute_skeletons(self) -> None:
        for filename in self.file_list:
            cache_path = os.path.join(self.skeleton_cache_dir, filename.replace(".png", "_skel.npy"))
            if os.path.exists(cache_path):
                continue
            raw_mask = np.array(
                Image.open(os.path.join(self.mask_dir, filename)).convert("L"),
            )
            class_mask = mask_to_class(raw_mask)
            skel = compute_skeleton(class_mask, tube_radius=self.tube_radius)
            np.save(cache_path, skel.astype(np.int32))

    def _load_skeleton(self, filename: str, class_mask: np.ndarray) -> np.ndarray:
        if self.skeleton_cache_dir:
            cache_path = os.path.join(
                self.skeleton_cache_dir, filename.replace(".png", "_skel.npy"),
            )
            if os.path.exists(cache_path):
                return np.load(cache_path)

        return compute_skeleton(class_mask, tube_radius=self.tube_radius)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        filename = self.file_list[idx]

        # ── Load grayscale image ──────────────────────────────────────
        image = np.array(
            Image.open(os.path.join(self.img_dir, filename)).convert("L"),
            dtype=np.uint8,
        )
        image = grayscale_to_3ch(image)

        if not self.is_test and self.mask_dir is not None:
            raw_mask = np.array(
                Image.open(os.path.join(self.mask_dir, filename)).convert("L"),
                dtype=np.uint8,
            )
            mask = mask_to_class(raw_mask)
            skeleton = self._load_skeleton(filename, mask)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.int32)
            skeleton = np.zeros_like(mask)

        if self.transform:
            augmented = self.transform(
                image=image,
                mask=mask,
                skeleton=skeleton,
            )
            image = augmented["image"]
            mask = augmented["mask"].long()
            skeleton = augmented["skeleton"].long()
        else:
            image = image.astype(np.float32) / 255.0
            image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask = torch.from_numpy(mask).long()
            skeleton = torch.from_numpy(skeleton).long()

        result = {
            "image": image,
            "mask": mask,
            "skeleton": skeleton,
            "filename": filename,
        }

        return result
