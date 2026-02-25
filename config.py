import os
import sys
import torch

class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "training_images")
    TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "training_masks")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    # ── Dataset ────────────────────────────────────────────────────────────────
    NUM_CLASSES = 3  # background, artery, vein
    CLASS_NAMES = ["background", "artery", "vein"]
    IMG_SIZE = 256  # sliding window patch size (px)
    IN_CHANNELS = 1  # grayscale input

    MASK_PIXEL_VALUES = {
        0: 0,  # background
        128: 1,  # artery
        255: 2,  # vein
    }

    # ── Model Architecture ─────────────────────────────────────────────────────
    CHANNELS = [64, 128, 256, 512, 1024]  # encoder feature map sizes
    DROPOUT_RATE = 0.1  # dropout probability

    # ── Training ───────────────────────────────────────────────────────────────
    BATCH_SIZE = 6  # mini batch size
    GRAD_ACCUMULATION_STEPS = 1  # effective batch
    EPOCHS = 600  # total training epochs

    # ── Optimizer (Adam) ───────────────────────────────────────────────────────
    LEARNING_RATE = 1e-3  # initial LR
    WEIGHT_DECAY = 1e-3  # L2 regularization for Adam

    # ── LR Scheduler (Step Decay) ──────────────────────────────────────────────
    LR_SCHEDULER = "step_decay"  # reduce LR by factor every N epochs
    LR_DECAY_FACTOR = 0.5  # LR ×0.5 per step
    LR_DECAY_EVERY_N_EPOCHS = 50  # decay interval in epochs

    # ── Early Stopping ─────────────────────────────────────────────────────────
    EARLY_STOPPING_PATIENCE = 150  # stop if no improvement for N epochs (25% of total)

    # ── Loss Function Weights  ────────────────────────────────────────
    DICE_WEIGHT = 1.0  # weight for Dice loss term
    CE_WEIGHT = 1.0  # weight for Cross-Entropy loss term
    REGULARIZATION_WEIGHT = 0.001  # weight for auxiliary/regularization term

    # ── Loss Function Weights ────────────────────────────────────────
    TAU = 3  # threshold / temperature parameter
    LAMBDA_D = 0.1  # weight for discriminator/distance loss term

    # ── Class Imbalance Handling ───────────────────────────────────────────────
    USE_DYNAMIC_WEIGHTS = True  # recompute class weights per batch
    CE_CLASS_WEIGHTS = [1.0, 2.5, 2.5]  # static weights: upweight artery & vein

    # ── Reproducibility ────────────────────────────────────────────────────────
    SEED = 42  # global random seed

    # ── Hardware ───────────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2 if sys.platform != "win32" else 0