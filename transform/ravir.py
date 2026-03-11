import os

import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize, dilation, disk
from torch.utils.data import Dataset

from config import Config


def mask_to_class(mask_np: np.ndarray) -> np.ndarray:
    class_mask = np.zeros_like(mask_np, dtype=np.int32)
    for pixel_value, class_idx in Config.MASK_PIXEL_VALUES.items():
        class_mask[mask_np == pixel_value] = class_idx
    return class_mask


def compute_class_weights(
        mask_dir: str,
        file_list: list,
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


class RAVIRDataset(Dataset):
    def __init__(
            self,
            img_dir: str,
            mask_dir: str | None = None,
            file_list: list | None = None,
            transform=None,
            is_test: bool = False,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        self.file_list = sorted(file_list) if file_list else sorted(os.listdir(img_dir))

        assert Config.IMG_SIZE <= 768, (
            f"IMG_SIZE={Config.IMG_SIZE} exceeds original image size 768×768."
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        filename = self.file_list[idx]

        # ── Load grayscale image ──────────────────────────────────────
        image = np.array(
            Image.open(os.path.join(self.img_dir, filename)).convert("L"),
            dtype=np.uint8,
        )

        # ── Load or create mask ───────────────────────────────────────
        if not self.is_test and self.mask_dir is not None:
            raw_mask = np.array(
                Image.open(os.path.join(self.mask_dir, filename)).convert("L"),
                dtype=np.uint8,
            )
            mask = mask_to_class(raw_mask)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.int32)

        # ── Apply transforms ──────────────────────────────────────────
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]          # (1, H, W) float tensor
            mask = augmented["mask"].long()      # (H, W) long tensor
        else:
            image = torch.from_numpy(
                ((image.astype(np.float32) / 255.0) - 0.5) / 0.5
            ).unsqueeze(0)
            mask = torch.from_numpy(mask).long()

        # ── Vessel probability: derived directly from mask tensor ─────
        vessel_prob = (mask > 0).float().unsqueeze(0)   # (1, H, W)

        # ── Tubed skeleton (computed after augmentation) ──────────────
        vp_np = vessel_prob.squeeze().numpy()
        binary = vp_np > 0
        if binary.any():
            skel = skeletonize(binary)
            skel = dilation(skel, disk(2)).astype(np.float32)
        else:
            skel = np.zeros_like(vp_np, dtype=np.float32)
        skeleton = torch.from_numpy(skel).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "vessel_prob": vessel_prob,
            "skeleton": skeleton,
            "filename": filename,
        }
