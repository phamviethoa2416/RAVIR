import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel, convolve
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

from config import Config


def mask_to_class(mask_np: np.ndarray) -> np.ndarray:
    class_mask = np.zeros_like(mask_np, dtype=np.int32)
    valid_pixels = np.zeros_like(mask_np, dtype=bool)
    for pixel_value, class_idx in Config.MASK_PIXEL_VALUES.items():
        region = (mask_np == pixel_value)
        class_mask[region] = class_idx
        valid_pixels |= region
    return class_mask


def _generate_aux_labels(mask_np: np.ndarray) -> dict[str, np.ndarray]:
    """Generate all auxiliary ground-truth maps from a class-index mask.

    Returns dict with keys: vessel_prob, orientation, width_artery,
    width_vein, endpoint.
    """
    artery = (mask_np == 1).astype(np.uint8)
    vein = (mask_np == 2).astype(np.uint8)
    vessel = ((mask_np > 0)).astype(np.uint8)

    # ── vessel probability (binary) ───────────────────────────────────
    vessel_prob = vessel.astype(np.float32)

    # ── orientation field (cos θ, sin θ along vessel) ─────────────────
    smoothed = gaussian_filter(vessel.astype(np.float32), sigma=2.0)
    dy = sobel(smoothed, axis=0)
    dx = sobel(smoothed, axis=1)
    along_x = -dy
    along_y = dx
    mag = np.sqrt(along_x ** 2 + along_y ** 2) + 1e-7
    cos_theta = (along_x / mag * vessel).astype(np.float32)
    sin_theta = (along_y / mag * vessel).astype(np.float32)

    # ── width maps per vessel type ────────────────────────────────────
    width_artery = distance_transform_edt(artery).astype(np.float32)
    width_vein = distance_transform_edt(vein).astype(np.float32)

    # ── endpoint probability ──────────────────────────────────────────
    skeleton = skeletonize(vessel).astype(np.int32)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbors = convolve(skeleton, kernel, mode="constant", cval=0)
    endpoint = ((skeleton > 0) & (neighbors == 1)).astype(np.float32)

    return {
        "vessel_prob": vessel_prob,
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "width_artery": width_artery,
        "width_vein": width_vein,
        "endpoint": endpoint,
    }


def compute_class_weights(
        mask_dir: str,
        file_list: list,
        min_weight: float = 0.5,
        max_weight: float = 5.0,
) -> torch.Tensor:
    class_counts = np.zeros(Config.NUM_CLASSES, dtype=np.float64)
    for filename in file_list:
        mask = np.array(
            Image.open(os.path.join(mask_dir, filename)).convert("L")
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

        image = np.array(
            Image.open(os.path.join(self.img_dir, filename)).convert("L"),
            dtype=np.uint8,
        )

        if not self.is_test and self.mask_dir is not None:
            raw_mask = np.array(
                Image.open(os.path.join(self.mask_dir, filename)).convert("L"),
                dtype=np.uint8,
            )
            mask = mask_to_class(raw_mask)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.int32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            image = torch.from_numpy(
                ((image.astype(np.float32) / 255.0) - 0.5) / 0.5
            ).unsqueeze(0)
            mask = torch.from_numpy(mask).long()

        # ── auxiliary labels (on-the-fly) ─────────────────────────────
        mask_np = mask.numpy()
        aux = _generate_aux_labels(mask_np)

        return {
            "image": image,
            "mask": mask,
            "vessel_prob": torch.from_numpy(aux["vessel_prob"]).unsqueeze(0),           # (1, H, W)
            "orientation": torch.stack([                                                  # (2, H, W)
                torch.from_numpy(aux["cos_theta"]),
                torch.from_numpy(aux["sin_theta"]),
            ], dim=0),
            "width": torch.stack([                                                        # (2, H, W)
                torch.from_numpy(aux["width_artery"]),
                torch.from_numpy(aux["width_vein"]),
            ], dim=0),
            "endpoint": torch.from_numpy(aux["endpoint"]).unsqueeze(0),                  # (1, H, W)
            "filename": filename,
        }
