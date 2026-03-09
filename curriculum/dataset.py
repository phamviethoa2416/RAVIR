from __future__ import annotations

import logging
import os
from fileinput import filename

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".bmp", ".gif"}

def _scan_images(directory: str) -> dict[str, str]:
    files = {}
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            stem = os.path.splitext(f)[0]
            files[stem] = os.path.join(directory, f)
    return files

def _pair_images_masks(img_dir: str, mask_dir: str) -> list[tuple[str, str]]:
    img_files = _scan_images(img_dir)
    mask_files = _scan_images(mask_dir)

    common = sorted(set(img_files) & set(mask_files))
    if common:
        return [(img_files[stem], mask_files[stem]) for stem in common]

    img_sorted = sorted(img_files.values())
    mask_sorted = sorted(mask_files.values())

    if len(img_sorted) != len(mask_sorted):
        raise ValueError(
            f"Cannot pair: {len(img_sorted)} images vs {len(mask_sorted)} masks, and no common stems found."
            f"in {img_dir} / {mask_dir} and no common stems found."
        )

    return list(zip(img_sorted, mask_sorted))

class RetinalVesselDataset(Dataset):
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
            mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.float().unsqueeze(0)
        elif isinstance(mask, np.ndarray) and mask.ndim == 2:
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "filename": os.path.basename(img_path),
        }

def build_datasets(
        data_root: str,
        train_transform=None,
        val_transform=None,
        val_ratio: float = 0.2,
        seed: int = 42,
) -> tuple[Dataset, Dataset]:
    all_pairs: list[tuple[str, str]] = []

    # ── DRIVE (training + test) ───────────────────────────────────────
    drive_base = os.path.join(data_root, "DRIVE")
    if os.path.isdir(drive_base):
        drive_splits = [
            ("training", "images", "1st_manual"),
            ("test", "images", "1st_manual"),
        ]
        for split, img_sub, mask_sub in drive_splits:
            img_dir = os.path.join(drive_base, split, img_sub)
            mask_dir = os.path.join(drive_base, split, mask_sub)
            if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
                logger.warning("Skipping DRIVE %s — missing %s or %s", split, img_dir, mask_dir)
                continue
            pairs = _pair_images_masks(img_dir, mask_dir)
            logger.info("DRIVE-%-8s : %d image-mask pairs", split, len(pairs))
            all_pairs.extend(pairs)
    else:
        logger.warning("DRIVE directory not found under %s", data_root)

    # ── STARE (images + masks) ────────────────────────────────────────
    stare_base = os.path.join(data_root, "STARE")
    if os.path.isdir(stare_base):
        img_dir = os.path.join(stare_base, "images")
        mask_dir = os.path.join(stare_base, "masks")
        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            pairs = _pair_images_masks(img_dir, mask_dir)
            logger.info("STARE      : %d image-mask pairs", len(pairs))
            all_pairs.extend(pairs)
        else:
            logger.warning("Skipping STARE — missing %s or %s", img_dir, mask_dir)
    else:
        logger.warning("STARE directory not found under %s", data_root)

    # ── CHASE_DB1 (images + 1stHO masks) ──────────────────────────────
    chase_base = os.path.join(data_root, "CHASE_DB1")
    if os.path.isdir(chase_base):
        img_dir = os.path.join(chase_base, "images")
        mask_dir = os.path.join(chase_base, "masks")
        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            pairs = _pair_images_masks(img_dir, mask_dir)
            logger.info("CHASE_DB1  : %d image-mask pairs", len(pairs))
            all_pairs.extend(pairs)
        else:
            logger.warning("Skipping CHASE_DB1 — missing %s or %s", img_dir, mask_dir)
    else:
        logger.warning("CHASE_DB1 directory not found under %s", data_root)

    # ── RITE (train/val/test) ─────────────────────────────────────────
    rite_base = os.path.join(data_root, "RITE")
    if os.path.isdir(rite_base):
        for split in ["train", "validation", "test"]:
            img_dir = os.path.join(rite_base, split, "images")
            mask_dir = os.path.join(rite_base, split, "masks")
            if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
                logger.warning("Skipping RITE %s — missing %s", split, img_dir if not os.path.isdir(img_dir) else mask_dir)
                continue
            pairs = _pair_images_masks(img_dir, mask_dir)
            logger.info("RITE-%-9s : %d image-mask pairs", split, len(pairs))
            all_pairs.extend(pairs)
    else:
        logger.warning("RITE directory not found under %s", data_root)

    # ── LES-AV (images + vessel-segmentations) ───────────────────────
    les_base = os.path.join(data_root, "LES-AV")
    if os.path.isdir(les_base):
        img_dir = os.path.join(les_base, "images")
        mask_dir = os.path.join(les_base, "vessel-segmentations")
        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            pairs = _pair_images_masks(img_dir, mask_dir)
            logger.info("LES-AV     : %d image-mask pairs", len(pairs))
            all_pairs.extend(pairs)
        else:
            logger.warning("Skipping LES-AV — missing %s or %s", img_dir, mask_dir)
    else:
        logger.warning("LES-AV directory not found under %s", data_root)

    # ── HRF (images + manual1 vessel labels) ─────────────────────────
    hrf_base = os.path.join(data_root, "HRF")
    if os.path.isdir(hrf_base):
        img_dir = os.path.join(hrf_base, "images")
        mask_dir = os.path.join(hrf_base, "manual1")
        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            pairs = _pair_images_masks(img_dir, mask_dir)
            logger.info("HRF        : %d image-mask pairs", len(pairs))
            all_pairs.extend(pairs)
        else:
            logger.warning("Skipping HRF — missing %s or %s", img_dir, mask_dir)
    else:
        logger.warning("HRF directory not found under %s", data_root)

    if not all_pairs:
        raise FileNotFoundError(
            f"No curriculum datasets (DRIVE / STARE / CHASE_DB1 / RITE / LES-AV / HRF) found under {data_root}."
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

    train_dataset = RetinalVesselDataset(train_pairs, transform=train_transform, dataset_name="GlobalTrain")
    val_dataset = RetinalVesselDataset(val_pairs, transform=val_transform, dataset_name="GlobalVal")

    return train_dataset, val_dataset