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
    USE_FRANGI = False
    FRANGI_SIGMAS = (2, 3, 4, 5)
    FRANGI_BLACK_RIDGES = True
    FRANGI_NORM_PERCENTILE = 99.7
    FRANGI_ALPHA = 0.5
    FRANGI_BETA = 1.0
    FRANGI_NORM_MEAN = 0.0375
    FRANGI_NORM_STD = 0.1009

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
    DROPOUT_RATE = 0.1

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

    # ── LR Scheduler ───────────────────────────────────────────────────────────
    LR_SCHEDULER = "cosine_warm_restarts"
    COSINE_T0 = 50
    COSINE_T_MULT = 2

    # ── Early Stopping ─────────────────────────────────────────────────────────
    EARLY_STOPPING_PATIENCE = 20

    # ── Segmentation Loss ──────────────────────────────────────────────────────
    DICE_WEIGHT = 1.0
    CE_WEIGHT = 1.0
    SKELETON_WEIGHT = 1.0
    USE_DEEP_SUPERVISION = True
    USE_SCSE = False
    USE_ATTENTION = False
    DS_WEIGHT = 0.4
    DS_DECAY = 0.8
    USE_CLDICE = False
    CLDICE_NUM_ITERATIONS = 10

    # ── Contrastive Learning ────────────────────────────────
    USE_CONTRASTIVE_LOSS = True
    SEGCON_WEIGHT = 0.1
    SEGCON_PROJECTOR_STAGE_IDX = 2
    SEGCON_EMBEDDING_DIM = 64
    SEGCON_HIDDEN_DIM = 128
    SEGCON_TEMPERATURE = 0.07
    SEGCON_NUM_ANCHORS = 256
    SEGCON_NUM_POSITIVES = 4
    SEGCON_NUM_NEGATIVES = 16
    SEGCON_NEGATIVE_RADIUS = 20
    SEGCON_CONFIDENCE_GATED = True
    SEGCON_CONFIDENCE_GAMMA = 1.0
    SEGCON_CONFIDENCE_DETACH = True
    SEGCON_BRANCH_CROSSING_RADIUS = 1
    SEGCON_BRANCH_MIN_PIXELS = 8
    SEGCON_BRANCH_NODE_PROXIMITY = 5
    SEGCON_BRANCH_SMALL_MODE = "merge"

    # ── Frangi auxiliary head ────────────────────────────────
    USE_FRANGI_RECON = False
    FRANGI_RECON_WEIGHT = 0.1
    FRANGI_RECON_LOSS = "mse"
    AUX_DECODER_CHANNELS = 64

    # ── Recursive Refinement ────────────────────────────────
    USE_RECURSIVE_REFINEMENT = False
    REFINEMENT_ITERATIONS = 2
    REFINEMENT_BASE_CHANNELS = 32
    REFINEMENT_MODE = "uniform"
    REFINEMENT_ITERATION_WEIGHT = 1.0
    REFINEMENT_BASE_SEGMENTATION_WEIGHT = 1.0

    # ── 4-way Rotation Expansion ────────────────────
    USE_ROTATION_EXPANSION = True

    # ── EMA of model weights ───────────────────────────────────────────────────
    USE_EMA = True
    EMA_DECAY = 0.995
    EMA_WARMUP_STEPS = 50

    # ── Skeleton ──────────────────────────────────────────────────────
    TUBE_RADIUS = 1

    # ── Class Imbalance ────────────────────────────────────────────────────────
    USE_DYNAMIC_WEIGHTS = True
    CE_CLASS_WEIGHTS = [1.0, 2.5, 2.5]

    # ── Reproducibility ────────────────────────────────────────────────────────
    SEED = 24

    # ── Hardware ───────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_NAME = GPU["gpu_name"]
    VRAM_GB = GPU["vram_gb"]
    NUM_WORKERS = GPU["num_workers"]
    PIN_MEMORY = GPU["pin_memory"]
    CUDNN_BENCHMARK = GPU["cudnn_benchmark"]
    ALLOW_TF32 = GPU["allow_tf32"]
    COMPILE_MODEL = True

    @classmethod
    def summary(cls) -> str:
        effective_batch = cls.BATCH_SIZE * cls.GRAD_ACCUMULATION_STEPS
        needs_sw = cls.IMG_SIZE < cls.ORIGINAL_SIZE
        lines: list[str] = [
            "── Paths ──",
            f"  DATA_DIR         : {cls.DATA_DIR}",
            f"  OUTPUT_DIR       : {cls.OUTPUT_DIR}",
            f"  FRANGI_CACHE_DIR : {cls.FRANGI_CACHE_DIR}",
            "── Dataset ──",
            f"  Classes          : {cls.NUM_CLASSES} {cls.CLASS_NAMES}",
            f"  IMG_SIZE         : {cls.IMG_SIZE}×{cls.IMG_SIZE}"
            + (
                " (full-size)"
                if cls.IMG_SIZE >= cls.ORIGINAL_SIZE
                else f" (patch; validate/inference SW for {cls.ORIGINAL_SIZE})"
            ),
            f"  ORIGINAL_SIZE    : {cls.ORIGINAL_SIZE}",
            f"  IN_CHANNELS      : {cls.IN_CHANNELS}",
            f"  Rotation expand  : {'yes (4-way)' if cls.USE_ROTATION_EXPANSION else 'no'}",
            f"  Skeleton tube radius  : {cls.TUBE_RADIUS}",
            "── Frangi input filter ──",
        ]

        if cls.USE_FRANGI:
            lines.append(f"  Enabled          : yes (concat as 4th channel)")
            lines.append(f"  Sigmas           : {cls.FRANGI_SIGMAS}")
            lines.append(f"  Black ridges     : {cls.FRANGI_BLACK_RIDGES}")
            lines.append(f"  Alpha / Beta     : {cls.FRANGI_ALPHA} / {cls.FRANGI_BETA}")
            lines.append(f"  Norm percentile  : {cls.FRANGI_NORM_PERCENTILE}")
            lines.append(
                f"  Norm mean / std  : {cls.FRANGI_NORM_MEAN} / {cls.FRANGI_NORM_STD}"
            )
        else:
            lines.append("  Enabled          : no (3-channel RGB input)")

        lines.append("── Model ──")
        lines.append(
            f"  Encoder          : {cls.ENCODER_NAME} (weights={cls.ENCODER_WEIGHTS})"
        )
        lines.append(f"  Dropout          : {cls.DROPOUT_RATE}")
        lines.append(f"  SCSE decoder     : {'yes' if cls.USE_SCSE else 'no'}")
        lines.append(f"  Attention gate   : {'yes' if cls.USE_ATTENTION else 'no'}")
        lines.append(
            f"  Deep supervision : {'yes' if cls.USE_DEEP_SUPERVISION else 'no'}"
            + (
                f" (w={cls.DS_WEIGHT}, decay={cls.DS_DECAY})"
                if cls.USE_DEEP_SUPERVISION
                else ""
            )
        )
        lines.append(
            f"  Recursive Refinement    : {'yes' if cls.USE_RECURSIVE_REFINEMENT else 'no'}"
        )
        if cls.USE_RECURSIVE_REFINEMENT:
            lines.append(f"    iterations     : {cls.REFINEMENT_ITERATIONS}")
            lines.append(f"    base channels  : {cls.REFINEMENT_BASE_CHANNELS}")
            lines.append(f"    iter mode      : {cls.REFINEMENT_MODE}")
            lines.append(f"    iter weight    : {cls.REFINEMENT_ITERATION_WEIGHT}")
            lines.append(
                f"    base seg weight: {cls.REFINEMENT_BASE_SEGMENTATION_WEIGHT}"
            )

        lines.append("── Frangi auxiliary head ──")
        if cls.USE_FRANGI_RECON:
            lines.append("  Enabled          : yes (train-only distillation)")
            lines.append(f"  Loss type        : {cls.FRANGI_RECON_LOSS}")
            lines.append(f"  Weight           : {cls.FRANGI_RECON_WEIGHT}")
            lines.append(f"  Aux mid channels : {cls.AUX_DECODER_CHANNELS}")
        else:
            lines.append("  Enabled          : no")

        lines.append("── SegCon (contrastive learning) ──")
        if cls.USE_CONTRASTIVE_LOSS:
            lines.append("  Enabled          : yes")
            lines.append(f"  Weight           : {cls.SEGCON_WEIGHT}")
            lines.append(f"  Projector stage  : {cls.SEGCON_PROJECTOR_STAGE_IDX}")
            lines.append(
                f"  Embed / hidden   : {cls.SEGCON_EMBEDDING_DIM} / {cls.SEGCON_HIDDEN_DIM}"
            )
            lines.append(f"  Temperature τ    : {cls.SEGCON_TEMPERATURE}")
            lines.append(
                f"  Anchors / positive / negative: {cls.SEGCON_NUM_ANCHORS}"
                f" / {cls.SEGCON_NUM_POSITIVES} / {cls.SEGCON_NUM_NEGATIVES}"
            )
            lines.append(f"  Negative radius  : {cls.SEGCON_NEGATIVE_RADIUS}")
            lines.append(
                f"  Confidence gated : {cls.SEGCON_CONFIDENCE_GATED}"
                + (
                    f" (γ={cls.SEGCON_CONFIDENCE_GAMMA}, detach={cls.SEGCON_CONFIDENCE_DETACH})"
                    if cls.SEGCON_CONFIDENCE_GATED
                    else ""
                )
            )
            lines.append(
                f"  Branch labels    : crossing={cls.SEGCON_BRANCH_CROSSING_RADIUS}, "
                f"min_px={cls.SEGCON_BRANCH_MIN_PIXELS}, "
                f"proximity={cls.SEGCON_BRANCH_NODE_PROXIMITY}, "
                f"small={cls.SEGCON_BRANCH_SMALL_MODE}"
            )
        else:
            lines.append("  Enabled          : no")

        lines.append("── Training ──")
        lines.append(f"  GPU              : {cls.GPU_NAME} ({cls.VRAM_GB:.1f} GB VRAM)")
        lines.append(f"  Device           : {cls.DEVICE}")
        lines.append(
            f"  Batch            : {cls.BATCH_SIZE} × {cls.GRAD_ACCUMULATION_STEPS} accum"
            f" = {effective_batch} effective"
        )
        lines.append(f"  Epochs           : {cls.EPOCHS}")
        lines.append(f"  CV folds         : {cls.NUM_FOLDS} (val fold={cls.VAL_FOLD})")
        lines.append(f"  Early stopping   : patience={cls.EARLY_STOPPING_PATIENCE}")
        lines.append(f"  Seed             : {cls.SEED}")

        lines.append("── Optimizer / scheduler ──")
        lines.append(f"  LR               : {cls.LEARNING_RATE:.2e}")
        lines.append(f"  Weight decay     : {cls.WEIGHT_DECAY:.2e}")
        lines.append(f"  Scheduler        : {cls.LR_SCHEDULER}")
        lines.append(f"  Warmup epochs    : {cls.WARMUP_EPOCHS}")
        lines.append(f"  Cosine T0 / Tmult : {cls.COSINE_T0} / {cls.COSINE_T_MULT}")

        lines.append("── Loss weights ──")
        lines.append(f"  CE               : {cls.CE_WEIGHT}")
        lines.append(f"  Dice             : {cls.DICE_WEIGHT}")
        lines.append(
            f"  clDice           : {'yes' if cls.USE_CLDICE else 'no'}"
            + (
                f" (w={cls.SKELETON_WEIGHT}, iters={cls.CLDICE_NUM_ITERATIONS})"
                if cls.USE_CLDICE
                else f" (w={cls.SKELETON_WEIGHT}, disabled)"
            )
        )
        lines.append(
            f"  Class weights    : "
            + (
                "dynamic (from train masks)"
                if cls.USE_DYNAMIC_WEIGHTS
                else str(cls.CE_CLASS_WEIGHTS)
            )
        )

        lines.append("── Hardware / runtime ──")
        lines.append(
            f"  AMP              : {cls.AMP_DTYPE}"
            + (" (disabled)" if not cls.USE_AMP else "")
        )
        lines.append(f"  TF32             : {'yes' if cls.ALLOW_TF32 else 'no'}")
        lines.append(f"  torch.compile    : {'yes' if cls.COMPILE_MODEL else 'no'}")
        lines.append(f"  Sliding window   : {'yes' if needs_sw else 'no'}")
        lines.append(f"  Workers          : {cls.NUM_WORKERS}")
        lines.append(f"  Pin memory       : {'yes' if cls.PIN_MEMORY else 'no'}")
        lines.append(f"  cuDNN benchmark  : {'yes' if cls.CUDNN_BENCHMARK else 'no'}")
        lines.append(
            f"  EMA              : {'yes' if cls.USE_EMA else 'no'}"
            + (
                f" (decay={cls.EMA_DECAY}, warmup={cls.EMA_WARMUP_STEPS} steps)"
                if cls.USE_EMA
                else ""
            )
        )

        return "\n".join(lines)

    @classmethod
    def override(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")
        if "USE_FRANGI" in kwargs:
            cls.IN_CHANNELS = 4 if cls.USE_FRANGI else 3
        if "USE_FRANGI_RECON" in kwargs and cls.USE_FRANGI_RECON:
            cls.USE_FRANGI = False
            cls.IN_CHANNELS = 3
