"""
Dataset classes for external vessel segmentation datasets (DRIVE, STARE, CHASE_DB1).

Expected directory layout:
    data/curriculum/
    ├── DRIVE/
    │   ├── images/     # *.tif, *.png, *.jpg, ...
    │   └── masks/      # binary vessel masks (same stems or sorted order)
    ├── STARE/
    │   ├── images/
    │   └── masks/
    └── CHASE_DB1/
        ├── images/
        └── masks/
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".bmp", ".gif"}


def _scan_images(directory: str) -> dict[str, str]:
    """Return {stem: full_path} for every supported image file in *directory*."""
    files = {}
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            stem = os.path.splitext(f)[0]
            files[stem] = os.path.join(directory, f)
    return files


def _pair_images_masks(img_dir: str, mask_dir: str) -> list[tuple[str, str]]:
    img_map = _scan_images(img_dir)
    mask_map = _scan_images(mask_dir)

    common = sorted(set(img_map) & set(mask_map))
    if common:
        return [(img_map[s], mask_map[s]) for s in common]

    img_sorted = sorted(img_map.values())
    mask_sorted = sorted(mask_map.values())
    if len(img_sorted) != len(mask_sorted):
        raise ValueError(
            f"Cannot pair: {len(img_sorted)} images vs {len(mask_sorted)} masks "
            f"in {img_dir} / {mask_dir} and no matching stems found."
        )
    return list(zip(img_sorted, mask_sorted))


class VesselDataset(Dataset):
    """Loads RGB retinal images with binary vessel masks."""

    def __init__(
            self,
            pairs: list[tuple[str, str]],
            transform=None,
            dataset_name: str = "unknown",
    ):
        self.pairs = pairs
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
            mask = torch.from_numpy(mask).float()

        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.float().unsqueeze(0)
        elif isinstance(mask, np.ndarray) and mask.ndim == 2:
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "filename": os.path.basename(img_path),
        }


def build_curriculum_datasets(
        data_root: str,
        train_transform=None,
        val_transform=None,
        val_ratio: float = 0.2,
        seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Discover DRIVE / STARE / CHASE_DB1 under *data_root*, merge them,
    then split into global train and validation sets.
    """
    dataset_dirs = {
        "DRIVE": os.path.join(data_root, "DRIVE"),
        "STARE": os.path.join(data_root, "STARE"),
        "CHASE_DB1": os.path.join(data_root, "CHASE_DB1"),
    }

    all_pairs: list[tuple[str, str]] = []
    pair_sources: list[str] = []

    for name, base in dataset_dirs.items():
        img_dir = os.path.join(base, "images")
        mask_dir = os.path.join(base, "masks")
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            logger.warning("Skipping %s — directories not found at %s", name, base)
            continue

        pairs = _pair_images_masks(img_dir, mask_dir)
        logger.info("%-10s : %d image-mask pairs", name, len(pairs))
        all_pairs.extend(pairs)
        pair_sources.extend([name] * len(pairs))

    if not all_pairs:
        raise FileNotFoundError(
            f"No curriculum datasets found under {data_root}. "
            "Expected sub-directories: DRIVE/, STARE/, CHASE_DB1/, "
            "each with images/ and masks/ folders."
        )

    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_pairs))
    rng.shuffle(indices)

    n_val = max(1, int(len(all_pairs) * val_ratio))
    val_idx = set(indices[:n_val].tolist())

    train_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in val_idx]
    val_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i in val_idx]

    logger.info("Global split: %d train, %d val (%.0f%% val)",
                len(train_pairs), len(val_pairs), val_ratio * 100)

    train_dataset = VesselDataset(train_pairs, transform=train_transform, dataset_name="GlobalTrain")
    val_dataset = VesselDataset(val_pairs, transform=val_transform, dataset_name="GlobalVal")

    return train_dataset, val_dataset
