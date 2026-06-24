from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.filters import frangi
from skimage.morphology import disk, skeletonize
from torch.utils.data import Dataset

from config import Config
from transform.graph import compute_branch_labels

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
    sigmas: tuple[int, ...] = (2, 3, 4, 5),
    black_ridges: bool = True,
    norm_percentile: float = 99.7,
) -> np.ndarray:
    img = gray.astype(np.float64) / 255.0

    response = frangi(
        img,
        sigmas=sigmas,
        black_ridges=black_ridges,
        alpha=Config.FRANGI_ALPHA,
        beta=Config.FRANGI_BETA,
    )

    positive = response[response > 0]
    if positive.size:
        scale = max(float(np.percentile(positive, norm_percentile)), 1e-8)
        response = np.clip(response / scale, 0.0, 1.0)
    elif response.max() > 0:
        response /= response.max()
    return response.astype(np.float32)


def compute_skeleton(
    class_mask: np.ndarray,
    vessel_classes: tuple[int, ...] = (1, 2),
    tube_radius: int = 1,
) -> np.ndarray:
    skeleton = np.zeros_like(class_mask, dtype=np.int32)
    selem = disk(tube_radius) if tube_radius > 0 else None

    for vessel_class in vessel_classes:
        binary = (class_mask == vessel_class).astype(np.uint8)
        if binary.sum() == 0:
            continue
        skel = skeletonize(binary > 0)
        if selem is not None:
            skel = binary_dilation(skel, selem)
        skeleton[skel > 0] = vessel_class

    return skeleton


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
        frangi_cache_dir: str | None = None,
        return_frangi_target: bool = False,
        use_rotation_expansion: bool = False,
        use_branch_labels: bool = False,
        branch_crossing_radius: int = 1,
        branch_min_pixels: int = 8,
        branch_node_proximity: int = 5,
        branch_small_mode: str = "merge",
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        self.skeleton_cache_dir = skeleton_cache_dir
        self.tube_radius = tube_radius
        self.frangi_cache_dir = frangi_cache_dir
        self.return_frangi_target = bool(return_frangi_target)

        self.compute_branch_labels = bool(use_branch_labels)
        self.branch_crossing_radius = int(branch_crossing_radius)
        self.branch_min_pixels = int(branch_min_pixels)
        self.branch_node_proximity = int(branch_node_proximity)
        self.branch_small_mode = str(branch_small_mode).lower()

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

        if skeleton_cache_dir and mask_dir and not is_test:
            os.makedirs(skeleton_cache_dir, exist_ok=True)
            self.precompute_skeleton()

        if frangi_cache_dir and (Config.USE_FRANGI or self.return_frangi_target):
            os.makedirs(frangi_cache_dir, exist_ok=True)
            self.precompute_frangi()

    def precompute_skeleton(self) -> None:
        for filename in self.file_list:
            cache_path = os.path.join(
                self.skeleton_cache_dir, filename.replace(".png", "_skel.npy")
            )
            if os.path.exists(cache_path):
                continue
            raw_mask = np.array(
                Image.open(os.path.join(self.mask_dir, filename)).convert("L")
            )
            class_mask = mask_to_class(raw_mask)
            skel = compute_skeleton(class_mask, tube_radius=self.tube_radius)
            np.save(cache_path, skel.astype(np.int32))

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

    def load_skeleton(self, filename: str, class_mask: np.ndarray) -> np.ndarray:
        if self.skeleton_cache_dir:
            cache_path = os.path.join(
                self.skeleton_cache_dir, filename.replace(".png", "_skel.npy")
            )
            if os.path.exists(cache_path):
                return np.load(cache_path)

        return compute_skeleton(class_mask, tube_radius=self.tube_radius)

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
        need_frangi = Config.USE_FRANGI or self.return_frangi_target

        if need_frangi:
            frangi_map = self.load_frangi(filename, gray)

        if not self.is_test and self.mask_dir is not None:
            raw_mask = np.array(
                Image.open(os.path.join(self.mask_dir, filename)).convert("L"),
                dtype=np.uint8,
            )
            mask = mask_to_class(raw_mask)
            skeleton = self.load_skeleton(filename, mask)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.int32)
            skeleton = np.zeros_like(mask)

        if rotation_k != 0:
            image = np.ascontiguousarray(np.rot90(image, k=rotation_k, axes=(0, 1)))
            mask = np.ascontiguousarray(np.rot90(mask, k=rotation_k, axes=(0, 1)))
            skeleton = np.ascontiguousarray(
                np.rot90(skeleton, k=rotation_k, axes=(0, 1))
            )

            if frangi_map is not None:
                frangi_map = np.ascontiguousarray(
                    np.rot90(frangi_map, k=rotation_k, axes=(0, 1))
                )

        frangi_target_tensor: torch.Tensor | None = None
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
            mask_np_aug = (
                mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            )
            skeleton = torch.from_numpy(
                compute_skeleton(
                    mask_np_aug.astype(np.int32), tube_radius=self.tube_radius
                )
            ).long()
            mask = (
                mask.long()
                if isinstance(mask, torch.Tensor)
                else torch.from_numpy(mask).long()
            )

            if frangi_map is not None:
                frangi_t_raw = augmented["frangi"].float()

                if self.return_frangi_target:
                    frangi_target_tensor = frangi_t_raw.unsqueeze(0).clone()

                if Config.USE_FRANGI:
                    frangi_t_input = (
                        frangi_t_raw - Config.FRANGI_NORM_MEAN
                    ) / Config.FRANGI_NORM_STD

                    image = torch.cat(
                        [image, frangi_t_input.unsqueeze(0)],
                        dim=0,
                    )
        else:
            rgb = image.astype(np.float32) / 255.0
            rgb = (rgb - NORM_MEAN[:3]) / NORM_STD[:3]

            if frangi_map is not None:

                if self.return_frangi_target:
                    frangi_target_tensor = (
                        torch.from_numpy(np.ascontiguousarray(frangi_map))
                        .float()
                        .unsqueeze(0)
                    )

                if Config.USE_FRANGI:
                    frangi_norm = (
                        frangi_map - Config.FRANGI_NORM_MEAN
                    ) / Config.FRANGI_NORM_STD

                    image = np.concatenate(
                        [rgb, frangi_norm[..., np.newaxis]],
                        axis=-1,
                    )
                else:
                    image = rgb
            else:
                image = rgb

            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask = torch.from_numpy(mask).long()
            skeleton = torch.from_numpy(skeleton).long()

        result: dict = {
            "image": image,
            "mask": mask,
            "skeleton": skeleton,
            "filename": filename,
        }

        if frangi_target_tensor is not None:
            result["frangi_target"] = frangi_target_tensor

        if self.compute_branch_labels and not self.is_test:
            mask_for_branch = (
                mask.cpu().numpy().astype(np.int32)
                if isinstance(mask, torch.Tensor)
                else mask.astype(np.int32)
            )

            branch_np = compute_branch_labels(
                mask_for_branch,
                crossing_radius=self.branch_crossing_radius,
                min_branch_pixels=self.branch_min_pixels,
                node_proximity=self.branch_node_proximity,
                small_component_mode=self.branch_small_mode,
            )

            result["branch_labels"] = torch.from_numpy(
                branch_np.astype(np.int32)
            ).long()

        return result
