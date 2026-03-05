import os

import torch


def _detect_gpu_profile() -> dict:
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

    torch.cuda.get_device_name(0).lower()
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
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
        # L4 (Ada Lovelace, cap_major=8) supports BF16
        # V100 (Volta, cap_major=7) only FP16
        dtype = "bfloat16" if cap_major >= 8 else "float16"
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gb": vram,
            "img_size": 768,
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

    # ── Small GPU (<12 GB) ───────────────────────────────────────────
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

_GPU = _detect_gpu_profile()


class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "training_images")
    TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "training_masks")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    # ── Dataset ────────────────────────────────────────────────────────────────
    NUM_CLASSES = 3
    CLASS_NAMES = ["background", "artery", "vein"]
    IMG_SIZE = _GPU["img_size"]             # auto-scaled to GPU VRAM
    ORIGINAL_SIZE = 768                     # RAVIR native image size
    IN_CHANNELS = 1

    MASK_PIXEL_VALUES = {
        0: 0,                               # background
        128: 1,                             # artery
        255: 2,                             # vein
    }
    CLASS_TO_PIXEL = {0: 0, 1: 128, 2: 256}

    # ── Model Architecture ─────────────────────────────────────────────────────
    ENCODER_NAME = "resnet34"               # SMP encoder backbone
    ENCODER_WEIGHTS = "imagenet"            # pretrained weights (None to train from scratch)
    DROPOUT_RATE = 0.1                      # dropout probability
    USE_ATTENTION = True                    # enable CBAM attention in decoder

    # ── Training (auto-scaled) ─────────────────────────────────────────────────
    BATCH_SIZE = _GPU["batch_size"]
    GRAD_ACCUMULATION_STEPS = _GPU["grad_accum"]
    EPOCHS = 600
    NUM_FOLDS = 5
    VAL_FOLD = 0

    # ── Mixed Precision (auto-detected) ────────────────────────────────────────
    USE_AMP = _GPU["use_amp"]
    AMP_DTYPE = _GPU["amp_dtype"]

    # ── Optimizer (AdamW) ──────────────────────────────────────────────────────
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    # ── LR Scheduler ───────────────────────────────────────────────────────────
    LR_SCHEDULER = "cosine_warm_restarts"
    COSINE_T0 = 50
    COSINE_T_MULT = 2

    # ── Early Stopping ─────────────────────────────────────────────────────────
    EARLY_STOPPING_PATIENCE = 100

    # ── Segmentation Loss ──────────────────────────────────────────────────────
    DICE_WEIGHT = 1.0
    CE_WEIGHT = 1.0
    TVERSKY_WEIGHT = 0.7
    TVERSKY_CE_WEIGHT = 0.3
    TVERSKY_ALPHA = 0.3
    TVERSKY_BETA = 0.7
    LABEL_SMOOTHING = 0.1

    # ── Auxiliary Loss ─────────────────────────────────────────────────────────
    VESSEL_PROB_LOSS_WEIGHT = 0.5
    VESSEL_PROB_POS_WEIGHT = 3.0

    # ── Class Imbalance ────────────────────────────────────────────────────────
    USE_DYNAMIC_WEIGHTS = False
    CE_CLASS_WEIGHTS = [1.0, 2.5, 2.5]

    # ── Reproducibility ────────────────────────────────────────────────────────
    SEED = 42

    # ── Hardware (auto-detected) ───────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_NAME = _GPU["gpu_name"]
    VRAM_GB = _GPU["vram_gb"]
    NUM_WORKERS = _GPU["num_workers"]
    PIN_MEMORY = _GPU["pin_memory"]
    CUDNN_BENCHMARK = _GPU["cudnn_benchmark"]
    ALLOW_TF32 = _GPU["allow_tf32"]
    COMPILE_MODEL = False

    @classmethod
    def summary(cls) -> str:
        eff_batch = cls.BATCH_SIZE * cls.GRAD_ACCUMULATION_STEPS
        needs_sw = cls.IMG_SIZE < cls.ORIGINAL_SIZE
        lines = [
            f"  GPU            : {cls.GPU_NAME}",
            f"  VRAM           : {cls.VRAM_GB:.1f} GB",
            f"  IMG_SIZE       : {cls.IMG_SIZE}×{cls.IMG_SIZE}"
            + (" (full-size)" if cls.IMG_SIZE == 768 else f" (patch, sliding window for {cls.ORIGINAL_SIZE})"),
            f"  Batch Size     : {cls.BATCH_SIZE} × {cls.GRAD_ACCUMULATION_STEPS} accum = {eff_batch} effective",
            f"  AMP            : {cls.AMP_DTYPE}" + (" (disabled)" if not cls.USE_AMP else ""),
            f"  TF32           : {'yes' if cls.ALLOW_TF32 else 'no'}",
            f"  Sliding Window : {'yes (train on patches)' if needs_sw else 'no (direct forward)'}",
            f"  Workers        : {cls.NUM_WORKERS}",
            "=" * 60,
        ]
        return "\n".join(lines)

    @classmethod
    def override(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")