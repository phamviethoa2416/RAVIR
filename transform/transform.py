import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2

from config import Config


def get_train_transform() -> alb.Compose:
    transforms = []

    if Config.IMG_SIZE < Config.ORIGINAL_SIZE:
        transforms.append(alb.RandomCrop(Config.IMG_SIZE, Config.IMG_SIZE))

    transforms.extend([
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
            alpha=15,
            sigma=6,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3,
        ),
        alb.GridDistortion(p=0.3),
        alb.OpticalDistortion(distort_limit=0.2, p=0.2),

        # Intensity
        alb.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        alb.RandomGamma(gamma_limit=(80, 120), p=0.3),
        alb.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4,
        ),
        alb.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.25),
        alb.GaussNoise(std_range=(0.02, 0.08), p=0.2),

        # Masking
        alb.CoarseDropout(
            num_holes_range=(3, 6),
            hole_height_range=(15, 40),
            hole_width_range=(15, 40),
            fill=0,
            p=0.3,
        ),

        # Normalize
        alb.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return alb.Compose(transforms)

def get_val_transform() -> alb.Compose:
    transforms = []

    if Config.IMG_SIZE < Config.ORIGINAL_SIZE:
        transforms.append(
            alb.Resize(Config.IMG_SIZE, Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR),
        )

    transforms.extend([
        alb.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return alb.Compose(transforms)


def get_test_transform() -> alb.Compose:
    return alb.Compose([
        alb.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])