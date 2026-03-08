"""
Transforms for Curriculum Learning rounds.

Round 1 (Vessel Discovery): RGB colour images, patch-based, scale to [0, 1].
Round 2 (Domain Adaptation): Grayscale IR images, patch-based, scale to [0, 1].
"""

import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2


# ── Round 1: RGB colour retinal images ────────────────────────────────────────


def get_round1_train_transform(patch_size: int = 256) -> alb.Compose:
    return alb.Compose([
        alb.RandomCrop(patch_size, patch_size),

        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        alb.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3,
        ),
        alb.ElasticTransform(alpha=15, sigma=6, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        alb.GridDistortion(p=0.2),

        alb.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        alb.RandomGamma(gamma_limit=(80, 120), p=0.3),
        alb.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        alb.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        alb.GaussNoise(std_range=(0.02, 0.08), p=0.2),

        alb.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_round1_val_transform(patch_size: int = 256) -> alb.Compose:
    return alb.Compose([
        alb.Resize(patch_size, patch_size, interpolation=cv2.INTER_LINEAR),
        alb.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ])


# ── Round 2: Grayscale IR (RAVIR) ────────────────────────────────────────────


def get_round2_train_transform(patch_size: int = 256) -> alb.Compose:
    return alb.Compose([
        alb.RandomCrop(patch_size, patch_size),

        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        alb.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3,
        ),
        alb.ElasticTransform(alpha=15, sigma=6, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        alb.GridDistortion(p=0.2),

        alb.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        alb.RandomGamma(gamma_limit=(80, 120), p=0.3),
        alb.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        alb.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        alb.GaussNoise(std_range=(0.02, 0.08), p=0.2),

        alb.Normalize(mean=(0,), std=(1,), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_round2_val_transform(img_size: int = 512) -> alb.Compose:
    return alb.Compose([
        alb.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        alb.Normalize(mean=(0,), std=(1,), max_pixel_value=255.0),
        ToTensorV2(),
    ])
