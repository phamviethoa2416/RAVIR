import os

import torch


def detect_gpu_profile() -> dict:
    if not torch.cuda.is_available():
        return {
            "gpu_name": "CPU",
            "vram_gb": 0,
            "img_size": 512,
            "batch_size": 2,
            "grad_accum": 8,
            "amp_dtype": "float32",
            "use_amp": False,
            "num_workers": 2,
            "pin_memory": False,
            "cudnn_benchmark": False,
            "allow_tf32": False,
        }

    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cap_major, _ = torch.cuda.get_device_capability(0)

    # ── H100 / A100-80GB (≥70 GB VRAM) ───────────────────────────────
    if vram >= 70:
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gb": vram,
            "img_size": 768,
            "batch_size": 8,
            "grad_accum": 2,
            "amp_dtype": "bfloat16",
            "use_amp": True,
            "num_workers": 4,
            "pin_memory": True,
            "cudnn_benchmark": True,
            "allow_tf32": True,
        }

    # ── A100-40GB (35–70 GB VRAM) ────────────────────────────────────
    if vram >= 35:
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gb": vram,
            "img_size": 768,
            "batch_size": 4,
            "grad_accum": 4,
            "amp_dtype": "bfloat16",
            "use_amp": True,
            "num_workers": 4,
            "pin_memory": True,
            "cudnn_benchmark": True,
            "allow_tf32": True,
        }

    # ── L4 / V100 / RTX 3090 (20–35 GB VRAM) ────────────────────────
    if vram >= 20:
        dtype = "bfloat16" if cap_major >= 8 else "float16"
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gb": vram,
            "img_size": 512,
            "batch_size": 2,
            "grad_accum": 8,
            "amp_dtype": dtype,
            "use_amp": True,
            "num_workers": 4,
            "pin_memory": True,
            "cudnn_benchmark": True,
            "allow_tf32": cap_major >= 8,
        }

    # ── T4 / P100 / RTX 2080 (12–20 GB VRAM) ────────────────────────
    if vram >= 12:
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gb": vram,
            "img_size": 512,
            "batch_size": 2,
            "grad_accum": 8,
            "amp_dtype": "float16",
            "use_amp": True,
            "num_workers": 2,
            "pin_memory": True,
            "cudnn_benchmark": True,
            "allow_tf32": False,
        }
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_gb": vram,
        "img_size": 384,
        "batch_size": 1,
        "grad_accum": 16,
        "amp_dtype": "float16",
        "use_amp": True,
        "num_workers": 2,
        "pin_memory": True,
        "cudnn_benchmark": True,
        "allow_tf32": False,
    }


GPU = detect_gpu_profile()


class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "training_images")
    TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "training_masks")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    FRANGI_CACHE_DIR = os.path.join(DATA_DIR, "frangi_cache")

    # ── Dataset ────────────────────────────────────────────────────────────────
    NUM_CLASSES = 3
    CLASS_NAMES = ["background", "artery", "vein"]
    IMG_SIZE = GPU["img_size"]
    ORIGINAL_SIZE = 768

    # ── Frangi Vesselness Filter ───────────────────────────────────────────────
    USE_FRANGI: bool = True
    FRANGI_SIGMAS = (1, 2, 3, 4)
    FRANGI_BLACK_RIDGES = True
    FRANGI_NORM_MEAN = 0.1
    FRANGI_NORM_STD = 0.2

    IN_CHANNELS = 4 if USE_FRANGI else 3

    MASK_PIXEL_VALUES = {
        0: 0,  # background
        128: 1,  # artery
        255: 2,  # vein
    }
    CLASS_TO_PIXEL = {0: 0, 1: 128, 2: 255}

    # ── Model Architecture ─────────────────────────────────────────────────────
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    DROPOUT_RATE = 0.15

    # ── Training ─────────────────────────────────────────────────
    BATCH_SIZE = GPU["batch_size"]
    GRAD_ACCUMULATION_STEPS = GPU["grad_accum"]
    EPOCHS = 200
    WARMUP_EPOCHS = 10
    NUM_FOLDS = 5
    VAL_FOLD = 0

    # ── Mixed Precision ────────────────────────────────────────
    USE_AMP = GPU["use_amp"]
    AMP_DTYPE = GPU["amp_dtype"]

    # ── Optimizer (AdamW) ──────────────────────────────────────────────────────
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    # ── Early Stopping ─────────────────────────────────────────────────────────
    EARLY_STOPPING_PATIENCE = 80

    # ── Segmentation Loss ──────────────────────────────────────────────────────
    DICE_WEIGHT = 1.0
    CE_WEIGHT = 1.0
    SKELETON_WEIGHT = 1.2
    USE_DEEP_SUPERVISION = True
    USE_CLDICE = True
    USE_RECURSIVE_REFINEMENT = True
    DS_WEIGHT = 0.4
    DS_DECAY = 0.8
    CLDICE_NUM_ITERATIONS = 10

    # ── Recursive Refinement ────────────────────────────────
    REFINEMENT_ITERATIONS = 2
    REFINEMENT_BASE_CHANNELS = 32
    REFINEMENT_MODE = "uniform"
    REFINEMENT_ITERATION_WEIGHT = 1.0
    REFINEMENT_BASE_SEGMENTATION_WEIGHT = 1.0

    # ── 3-Phase Curriculum Training ────────────────────
    USE_CURRICULUM_TRAINING = True
    CURRICULUM_FIRST_PHASE_END = 50
    CURRICULUM_SECOND_PHASE_END = 150
    # ── 4-way Rotation Expansion ────────────────────
    USE_ROTATION_EXPANSION = True

    # ── EMA of model weights ───────────────────────────────────────────────────
    USE_EMA = True
    EMA_DECAY = 0.995
    EMA_WARMUP_STEPS = 50

    # ── Class Imbalance ────────────────────────────────────────────────────────
    USE_DYNAMIC_WEIGHTS = True
    CE_CLASS_WEIGHTS = [1.0, 2.5, 2.5]

    # ── Reproducibility ────────────────────────────────────────────────────────
    SEED = 42

    # ── Hardware ───────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_NAME = GPU["gpu_name"]
    VRAM_GB = GPU["vram_gb"]
    NUM_WORKERS = GPU["num_workers"]
    PIN_MEMORY = GPU["pin_memory"]
    CUDNN_BENCHMARK = GPU["cudnn_benchmark"]
    ALLOW_TF32 = GPU["allow_tf32"]
    COMPILE_MODEL = False

    @classmethod
    def summary(cls) -> str:
        effective_batch = cls.BATCH_SIZE * cls.GRAD_ACCUMULATION_STEPS
        needs_sw = cls.IMG_SIZE < cls.ORIGINAL_SIZE
        lines = [
            f"  GPU            : {cls.GPU_NAME}",
            f"  VRAM           : {cls.VRAM_GB:.1f} GB",
            f"  Encoder        : {cls.ENCODER_NAME} "
            f"(weights={cls.ENCODER_WEIGHTS})",
            f"  IMG_SIZE       : {cls.IMG_SIZE}×{cls.IMG_SIZE}"
            + (
                " (full-size)"
                if cls.IMG_SIZE == 768
                else f" (patch, sliding window for {cls.ORIGINAL_SIZE})"
            ),
            f"  Batch Size     : {cls.BATCH_SIZE} × {cls.GRAD_ACCUMULATION_STEPS} accum = {effective_batch} effective",
            f"  AMP            : {cls.AMP_DTYPE}"
            + (" (disabled)" if not cls.USE_AMP else ""),
            f"  TF32           : {'yes' if cls.ALLOW_TF32 else 'no'}",
            f"  Loss           : CE({cls.CE_WEIGHT}) + Dice({cls.DICE_WEIGHT}) + "
            + ("clDice" if cls.USE_CLDICE else "SkelRecall")
            + f"({cls.SKELETON_WEIGHT})",
            f"  Deep Supervision: {'yes' if cls.USE_DEEP_SUPERVISION else 'no'}"
            + (
                f" (w={cls.DS_WEIGHT}, decay={cls.DS_DECAY})"
                if cls.USE_DEEP_SUPERVISION
                else ""
            ),
            f"  EMA            : {'yes' if cls.USE_EMA else 'no'}"
            + (
                f" (decay={cls.EMA_DECAY}, warmup={cls.EMA_WARMUP_STEPS} steps)"
                if cls.USE_EMA
                else ""
            ),
            f"  RR Refinement  : {'yes' if cls.USE_RECURSIVE_REFINEMENT else 'no'}"
            + (
                f" (K={cls.REFINEMENT_ITERATIONS}, "
                f"base_ch={cls.REFINEMENT_BASE_CHANNELS}, "
                f"mode={cls.REFINEMENT_MODE})"
                if cls.USE_RECURSIVE_REFINEMENT
                else ""
            ),
            f"  Curriculum     : {'yes' if cls.USE_CURRICULUM_TRAINING else 'no'}"
            + (
                f" (light → full → light at epoch "
                f"{cls.CURRICULUM_FIRST_PHASE_END} / {cls.CURRICULUM_SECOND_PHASE_END})"
                if cls.USE_CURRICULUM_TRAINING
                else ""
            ),
            f"  Rotation Expand: {'yes (4-way deterministic)' if cls.USE_ROTATION_EXPANSION else 'no'}",
            f"  Warmup epochs  : {cls.WARMUP_EPOCHS}",
            f"  Frangi Filter  : {'yes (4 channels input)' if cls.USE_FRANGI else 'no (3 channels input)'}",
            f"  Sliding Window : {'yes (train on patches)' if needs_sw else 'no (direct forward)'}",
            f"  Workers        : {cls.NUM_WORKERS}",
        ]
        return "\n".join(lines)

    @classmethod
    def override(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")
