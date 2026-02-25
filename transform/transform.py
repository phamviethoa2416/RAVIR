import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2

from config import Config

def get_train_transform() -> alb.Compose:
    return alb.Compose([
        # Patch extraction
        alb.RandomCrop(Config.IMG_SIZE, Config.IMG_SIZE),

        # Geometric
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.Rotate(
            limit=180,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7,
        ),
        alb.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3,
        ),
        alb.ElasticTransform(
            alpha=34,
            sigma=4,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3,
        ),

        # Intensity
        alb.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        alb.RandomGamma(gamma_limit=(80, 120), p=0.3),
        alb.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4,
        ),
        alb.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.25),
        alb.GaussNoise(p=0.2),

        # Normalize
        alb.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])

def get_val_transform() -> alb.Compose:
    return alb.Compose([
        alb.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            p=1.0,
        ),
        alb.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])