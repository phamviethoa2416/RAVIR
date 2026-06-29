from __future__ import annotations

import argparse
import logging
import os
import warnings

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.amp import autocast
from tqdm import tqdm

from config import Config
from models import RAVIRNet
from training import get_amp_dtype, sw_inference, unwrap_model
from transform import get_test_transform, get_tta_transform, tta_inverse_transform
from utils import setup_logging
from utils.visualization import class_mask_to_rgb

matplotlib.use("Agg")


def save_outputs(
    probs: np.ndarray,
    gray: np.ndarray,
    filename: str,
    pred_dir: str,
    softmax_dir: str | None,
    vis_dir: str | None,
) -> None:
    stem = os.path.splitext(filename)[0]
    class_map = probs.argmax(axis=0).astype(np.int32)

    pixel_mask = np.zeros_like(class_map, dtype=np.uint8)
    for class_idx, pixel in Config.CLASS_TO_PIXEL.items():
        pixel_mask[class_map == class_idx] = pixel
    Image.fromarray(pixel_mask).save(os.path.join(pred_dir, filename))

    binary_vessel = (class_map > 0).astype(np.uint8) * 255
    Image.fromarray(binary_vessel).save(os.path.join(pred_dir, f"{stem}_vessel.png"))

    if softmax_dir is not None:
        np.savez_compressed(
            os.path.join(softmax_dir, f"{stem}.npz"),
            probs=probs.astype(np.float32),
        )

    if vis_dir is not None:
        pred_rgb = class_mask_to_rgb(class_map)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(pred_rgb)
        axes[1].set_title("Prediction")
        axes[1].axis("off")
        plt.suptitle(filename, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)


def predict_probs(
    model: nn.Module | list[nn.Module],
    gray: np.ndarray,
    *,
    device: str,
    use_tta: bool = True,
    use_sw: bool | None = None,
    tile_size: int | None = None,
    overlap: int | None = None,
) -> np.ndarray:
    models = model if isinstance(model, list) else [model]
    ref_model = models[0]

    if tile_size is None:
        tile_size = int(getattr(ref_model, "inference_tile_size", Config.IMG_SIZE))
    if overlap is None:
        overlap = int(getattr(ref_model, "inference_overlap", max(0, tile_size // 4)))
    if use_sw is None:
        use_sw = tile_size < Config.ORIGINAL_SIZE

    amp_dtype = get_amp_dtype()
    use_amp = Config.USE_AMP and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"

    if use_tta:
        transforms = get_tta_transform()
        inverses = tta_inverse_transform()
    else:
        transforms = [("identity", get_test_transform())]
        inverses = {"identity": lambda t: t}

    acc: torch.Tensor | None = None
    num_passes = 0
    for m in models:
        for name, transform in transforms:
            rgb = np.stack([gray, gray, gray], axis=-1)
            image = transform(image=rgb)["image"]
            image = image.unsqueeze(0).to(device, non_blocking=True)

            with torch.no_grad():
                if use_sw:
                    segmentation, refinement = sw_inference(
                        model=m,
                        image=image,
                        tile_size=tile_size,
                        overlap=overlap,
                        num_classes=Config.NUM_CLASSES,
                    )
                    logits = (
                        refinement[-1].float() if refinement else segmentation.float()
                    )
                else:
                    with autocast(
                        device_type=autocast_device,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        out = m(image)
                    refinement = out.get("refinement")
                    if (
                        refinement is not None
                        and isinstance(refinement, list)
                        and refinement
                    ):
                        logits = refinement[-1].float()
                    else:
                        logits = out["segmentation"].float()

            probs = torch.softmax(logits, dim=1)
            probs = inverses[name](probs)
            acc = probs if acc is None else acc + probs
            num_passes += 1

    mean_probs = (acc / float(num_passes)).squeeze(0)
    return mean_probs.cpu().numpy().astype(np.float32)


def predict_test(
    model: nn.Module | list[nn.Module],
    run_dir: str,
    device: str,
    logger: logging.Logger | None = None,
    *,
    use_tta: bool = True,
    save_softmax: bool = True,
    save_visualisation: bool = True,
    tile_size: int | None = None,
    overlap: int | None = None,
) -> None:
    if logger is None:
        logger = logging.getLogger("RAVIR Predict")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    ref_model = model[0] if isinstance(model, list) else model
    if tile_size is None:
        tile_size = int(getattr(ref_model, "inference_tile_size", Config.IMG_SIZE))
    if overlap is None:
        overlap = int(getattr(ref_model, "inference_overlap", max(0, tile_size // 4)))
    use_sw = tile_size < Config.ORIGINAL_SIZE

    test_img_dir = Config.TEST_IMG_DIR
    if not os.path.isdir(test_img_dir):
        logger.info(f"Test directory not found: {test_img_dir}")
        return

    test_files = sorted(
        f for f in os.listdir(test_img_dir) if f.lower().endswith(".png")
    )
    if not test_files:
        logger.info("No test images found")
        return

    num_models = len(model) if isinstance(model, list) else 1
    logger.info(f"\n{'=' * 60}")
    logger.info("Test set prediction")
    logger.info(f"{'=' * 60}")
    logger.info(f"Test images: {len(test_files)}")
    logger.info(f"Ensemble size: {num_models} models")
    logger.info("Test-Time Augmentation: " + ("Enabled" if use_tta else "Disabled"))
    logger.info(f"Sliding window: {use_sw} (tile={tile_size}, overlap={overlap})")
    logger.info(f"Save softmax probs: {save_softmax}")
    logger.info(f"Save visualisation: {save_visualisation}")

    pred_dir = os.path.join(run_dir, "test_predictions")
    softmax_dir = os.path.join(run_dir, "test_softmax") if save_softmax else None
    vis_dir = (
        os.path.join(run_dir, "test_visualizations") if save_visualisation else None
    )
    os.makedirs(pred_dir, exist_ok=True)
    if softmax_dir:
        os.makedirs(softmax_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()

    for filename in tqdm(test_files, desc="  Test prediction"):
        gray = np.array(
            Image.open(os.path.join(test_img_dir, filename)).convert("L"),
            dtype=np.uint8,
        )

        probs = predict_probs(
            model=model,
            gray=gray,
            device=device,
            use_tta=use_tta,
            tile_size=tile_size,
            overlap=overlap,
        )

        save_outputs(
            probs=probs,
            gray=gray,
            filename=filename,
            pred_dir=pred_dir,
            softmax_dir=softmax_dir,
            vis_dir=vis_dir,
        )

    logger.info(f" Prediction masks: {pred_dir}")
    if softmax_dir:
        logger.info(f" Softmax probs: {softmax_dir}")
    if vis_dir:
        logger.info(f" Visualization dir: {vis_dir}")


def load_model(checkpoint_path: str, device: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    sd = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in sd):
        sd = {key.replace("_orig_mod.", "", 1): value for key, value in sd.items()}

    if "refinement_iterations" not in config:
        config["refinement_iterations"] = config.get(
            "refinement_num_iterations",
            Config.REFINEMENT_ITERATIONS,
        )
    if "use_refinement" not in config:
        config["use_refinement"] = any(key.startswith("refinement.") for key in sd)
    if "use_frangi_recon" not in config:
        config["use_frangi_recon"] = any(key.startswith("frangi_head.") for key in sd)
    if "cl_projector_stage_idx" not in config:
        if any(key.startswith("projector.") for key in sd):
            config["cl_projector_stage_idx"] = Config.SEGCON_PROJECTOR_STAGE_IDX
        elif config.get("use_contrastive_loss"):
            config["cl_projector_stage_idx"] = Config.SEGCON_PROJECTOR_STAGE_IDX
        else:
            config["cl_projector_stage_idx"] = None

    tile_size = int(config.get("img_size", Config.IMG_SIZE))
    in_channels = 3

    model = RAVIRNet(
        encoder_name=config.get("encoder_name", Config.ENCODER_NAME),
        in_channels=in_channels,
        num_classes=config.get("num_classes", Config.NUM_CLASSES),
        encoder_weights=None,
        dropout=config.get("dropout", Config.DROPOUT_RATE),
        use_deep_supervision=config.get(
            "use_deep_supervision", Config.USE_DEEP_SUPERVISION
        ),
        use_scse=config.get("use_scse", Config.USE_SCSE),
        use_attention=config.get("use_attention", Config.USE_ATTENTION),
        use_frangi_recon=config.get("use_frangi_recon", Config.USE_FRANGI_RECON),
        aux_mid_channels=config.get("aux_mid_channels", Config.AUX_DECODER_CHANNELS),
        use_refinement=config.get("use_refinement", Config.USE_RECURSIVE_REFINEMENT),
        refinement_iterations=config.get(
            "refinement_iterations", Config.REFINEMENT_ITERATIONS
        ),
        refinement_base_channels=config.get(
            "refinement_base_channels", Config.REFINEMENT_BASE_CHANNELS
        ),
        cl_projector_stage_idx=config.get("cl_projector_stage_idx"),
        cl_embedding_dim=config.get("cl_embedding_dim", Config.SEGCON_EMBEDDING_DIM),
        cl_hidden_dim=config.get("cl_hidden_dim", Config.SEGCON_HIDDEN_DIM),
    ).to(device)

    model.in_channels = in_channels
    missing, unexpected = unwrap_model(model).load_state_dict(sd, strict=False)
    if unexpected:
        warnings.warn(
            f"Ignoring {len(unexpected)} unexpected key(s) when loading "
            f"{checkpoint_path}: {unexpected[:5]}"
            + (" ..." if len(unexpected) > 5 else ""),
            stacklevel=2,
        )
    if missing:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is missing {len(missing)} "
            f"required weight(s), e.g. {missing[:5]}"
        )

    model.inference_tile_size = tile_size
    model.inference_overlap = max(0, tile_size // 4)
    return model


def run_kfold(
    checkpoint_paths: list[str],
    output_dir: str,
    device: str,
    logger: logging.Logger | None = None,
    *,
    use_tta: bool = True,
    save_softmax: bool = True,
    save_visualisation: bool = True,
) -> None:
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for k-fold prediction")

    models: list[nn.Module] = []
    tile_size: int | None = None
    overlap: int | None = None
    for path in checkpoint_paths:
        model = load_model(path, device)
        models.append(model)
        if tile_size is None:
            tile_size = model.inference_tile_size
            overlap = model.inference_overlap
        elif (
            model.inference_tile_size != tile_size or model.inference_overlap != overlap
        ):
            warnings.warn(
                "Ensemble checkpoints have different inference settings; "
                "using settings from the first checkpoint.",
                stacklevel=2,
            )

    target: nn.Module | list[nn.Module] = models[0] if len(models) == 1 else models

    predict_test(
        model=target,
        run_dir=output_dir,
        device=device,
        logger=logger,
        use_tta=use_tta,
        save_softmax=save_softmax,
        save_visualisation=save_visualisation,
        tile_size=tile_size,
        overlap=overlap,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict segmentation masks for test images using a trained model."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="+",
        help="Checkpoint paths to trained model weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="directory for outputs (default: same directory as the first checkpoint)",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="disable 8x Test-Time augmentation",
    )
    parser.add_argument(
        "--no-softmax", action="store_true", help="skip saving softmax files"
    )
    parser.add_argument(
        "--no-visualisation",
        action="store_true",
        help="skip saving visualisation files",
    )

    args = parser.parse_args()
    device = Config.DEVICE
    checkpoints = (
        args.checkpoint if isinstance(args.checkpoint, list) else [args.checkpoint]
    )

    models: list[nn.Module] = []
    tile_size: int | None = None
    overlap: int | None = None
    for path in checkpoints:
        model = load_model(path, device)
        models.append(model)
        if tile_size is None:
            tile_size = model.inference_tile_size
            overlap = model.inference_overlap
        elif (
            model.inference_tile_size != tile_size or model.inference_overlap != overlap
        ):
            warnings.warn(
                "Ensemble checkpoints have different inference settings; "
                "using settings from the first checkpoint.",
                stacklevel=2,
            )

    run_dir = args.output_dir or os.path.dirname(os.path.abspath(checkpoints[0]))
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logging(run_dir, name="RAVIR Predict")

    target = models[0] if len(models) == 1 else models
    predict_test(
        target,
        run_dir,
        device=device,
        logger=logger,
        use_tta=not args.no_tta,
        save_softmax=not args.no_softmax,
        save_visualisation=not args.no_visualisation,
        tile_size=tile_size,
        overlap=overlap,
    )


if __name__ == "__main__":
    main()
