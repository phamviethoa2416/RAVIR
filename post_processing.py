from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import Config
from postprocessing.continuity import (
    compute_image_bg_bias,
    apply_class_bias,
    bridge_vessel_gaps,
    inherit_softmax_at_bridges,
)
from postprocessing.graph_construction import build_vessel_graph
from postprocessing.refinement import Refinement, refinement
from utils import setup_logging
from utils.visualization import class_mask_to_rgb

matplotlib.use("Agg")

DEFAULT_MIN_VESSEL_PIXELS = 200
_REPO_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)


def graph_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "close_radius": getattr(args, "graph_close_radius", 1),
        "min_mask_object_size": getattr(args, "graph_min_mask_object_size", 25),
        "min_branch_length": getattr(args, "graph_min_branch_length", 10),
        "min_component_size": getattr(args, "graph_min_component_size", 15),
        "cleanup_iterations": getattr(args, "graph_cleanup_iterations", 3),
        "dilation_radius": getattr(args, "graph_dilation_radius", 2),
        "num_pixels_for_direction": getattr(args, "graph_direction_pixels", 3),
    }


def class_map_to_pixels(class_map: np.ndarray) -> np.ndarray:
    pixel_mask = np.zeros_like(class_map, dtype=np.uint8)
    for class_idx, pixel in Config.CLASS_TO_PIXEL.items():
        pixel_mask[class_map == class_idx] = pixel
    return pixel_mask


def _apply_class_bias(cnn_probs: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.adaptive_background_bias:
        bg_bias, _ = compute_image_bg_bias(
            cnn_probs,
            feature=args.bg_bias_feature,
            base=args.bg_bias_base,
            slope=args.bg_bias_slope,
            pivot=args.bg_bias_pivot,
            clip=(args.bg_bias_clip_min, args.bg_bias_clip_max),
        )
    else:
        bg_bias = args.bg_bias

    return apply_class_bias(
        cnn_probs,
        bg_bias=bg_bias,
        artery_bias=args.artery_bias,
        vein_bias=args.vein_bias,
    )


def strip_bridge_pixels(
    final_pred: np.ndarray,
    cnn_pred: np.ndarray,
    bridge_mask: np.ndarray | None,
    stages: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    """Topology-only bridge strip: zero bridge pixels that were CNN background."""
    if bridge_mask is None or not bridge_mask.any():
        return final_pred, stages

    bg_only = bridge_mask & (cnn_pred == 0)
    if not bg_only.any():
        return final_pred, stages

    out = final_pred.copy()
    out[bg_only] = 0

    new_stages = stages
    if stages is not None:
        new_stages = {}
        for name, stage_pred in stages.items():
            updated = stage_pred.copy()
            updated[bg_only] = 0
            new_stages[name] = updated

    return out, new_stages


def propagation_diff_overlay(
    before_pred: np.ndarray,
    after_pred: np.ndarray,
    gray: np.ndarray,
) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    base *= 0.40
    out = base.copy()

    artery_to_vein = (before_pred == 1) & (after_pred == 2)
    vein_to_artery = (before_pred == 2) & (after_pred == 1)
    out[artery_to_vein] = (0.15, 0.95, 1.00)
    out[vein_to_artery] = (1.00, 0.35, 0.95)

    changed = artery_to_vein | vein_to_artery
    if changed.any():
        dim = ~changed
        out[dim] *= 0.55

    return np.clip(out, 0.0, 1.0)


def _propagation_change_mask(
    before_pred: np.ndarray,
    after_pred: np.ndarray,
) -> np.ndarray:
    return (before_pred > 0) & (after_pred > 0) & (before_pred != after_pred)


def save_visualisation(
    gray: np.ndarray,
    cnn_pred: np.ndarray,
    final_pred: np.ndarray,
    filename: str,
    out_path: Path,
    bridge_mask: np.ndarray | None = None,
    before_propagation: np.ndarray | None = None,
    after_propagation: np.ndarray | None = None,
    propagation_segment_mask: np.ndarray | None = None,
) -> None:
    pixel_change_mask = np.zeros_like(gray, dtype=bool)
    if before_propagation is not None and after_propagation is not None:
        pixel_change_mask = _propagation_change_mask(
            before_propagation, after_propagation
        )

    segment_mask = (
        propagation_segment_mask
        if propagation_segment_mask is not None
        else np.zeros_like(gray, dtype=bool)
    )
    focus_mask = segment_mask | pixel_change_mask
    show_propagation = (
        before_propagation is not None
        and after_propagation is not None
        and focus_mask.any()
    )

    if show_propagation:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5.4))
        n_px = int(pixel_change_mask.sum())
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
        n_px = 0

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("IR input")
    axes[0].axis("off")

    cnn_rgb = class_mask_to_rgb(cnn_pred).copy()
    n_bridge_pixels = 0
    if bridge_mask is not None and bridge_mask.any():
        n_bridge_pixels = int(bridge_mask.sum())
        cnn_rgb[bridge_mask] = (255, 255, 0)
    axes[1].imshow(cnn_rgb)
    if n_bridge_pixels > 0:
        axes[1].set_title(f"Prediction (+ {n_bridge_pixels} bridged px)")
    else:
        axes[1].set_title("Prediction")
    axes[1].axis("off")

    if show_propagation:
        diff = propagation_diff_overlay(before_propagation, after_propagation, gray)
        if segment_mask.any():
            unchanged_seg = segment_mask & (~pixel_change_mask)
            diff[unchanged_seg & (after_propagation == 1)] = (1.0, 0.25, 0.25)
            diff[unchanged_seg & (after_propagation == 2)] = (0.25, 0.45, 1.0)
        axes[2].imshow(diff)
        axes[2].set_title(
            f"Propagation diff ({n_px} relabelled px)\n"
            "cyan A→V  magenta V→A  red/blue = segment label"
        )
        axes[2].axis("off")

        axes[3].imshow(class_mask_to_rgb(final_pred))
        axes[3].set_title("Post-processed (final)")
        axes[3].axis("off")
    else:
        axes[2].imshow(class_mask_to_rgb(final_pred))
        axes[2].set_title("Post-processed")
        axes[2].axis("off")

    fig.suptitle(filename, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    refined: Refinement,
    bundle: dict,
    pred_dir: Path,
    vis_dir: Path | None,
    stages_root: Path | None = None,
    bridge_mask: np.ndarray | None = None,
) -> None:
    filename = bundle["filename"]
    stem = os.path.splitext(filename)[0]
    final = refined.prediction
    cnn_pred = refined.cnn_argmax
    if cnn_pred is None:
        raise ValueError("Refinement is missing cnn_argmax")

    pred_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(class_map_to_pixels(final)).save(pred_dir / filename)
    Image.fromarray(((final > 0).astype(np.uint8) * 255)).save(
        pred_dir / f"{stem}_vessel.png"
    )

    if vis_dir is not None:
        save_visualisation(
            gray=bundle["gray"],
            cnn_pred=cnn_pred,
            final_pred=final,
            filename=filename,
            out_path=vis_dir / filename,
            bridge_mask=bridge_mask,
            before_propagation=refined.before_propagation,
            after_propagation=refined.after_propagation,
            propagation_segment_mask=refined.propagation_segment_mask,
        )

    if stages_root is not None and refined.stages:
        for step_name, step_pred in refined.stages.items():
            step_dir = stages_root / step_name
            step_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(class_map_to_pixels(step_pred)).save(step_dir / filename)


def run_one(
    bundle: dict,
    args: argparse.Namespace,
    *,
    log: logging.Logger | None = None,
) -> tuple[Refinement, np.ndarray | None] | None:
    log = log or logger
    cnn_probs = bundle["cnn_probs"]
    label = str(bundle.get("filename", bundle.get("name", "image")))

    if cnn_probs is None:
        log.warning("Post-processing skipped for %s: missing cnn_probs", label)
        return None

    cnn_probs = _apply_class_bias(cnn_probs, args)
    baseline_argmax = cnn_probs.argmax(axis=0).astype(np.int32)
    vessel_mask = baseline_argmax > 0

    min_vessel_pixels = int(
        getattr(args, "min_vessel_pixels", DEFAULT_MIN_VESSEL_PIXELS)
    )
    vessel_pixels = int(vessel_mask.sum())
    if vessel_pixels < min_vessel_pixels:
        log.warning(
            "Post-processing skipped for %s: vessel pixels %d < min_vessel_pixels %d",
            label,
            vessel_pixels,
            min_vessel_pixels,
        )
        return None

    bridge_pixel_mask: np.ndarray | None = None

    if args.bridge_gaps:
        continuity = bridge_vessel_gaps(
            vessel_mask,
            cnn_probs,
            max_gap=args.bridge_max_gap,
            max_angle_dev_deg=args.bridge_max_angle,
            tangent_window=args.bridge_tangent_window,
            min_cnn_evidence=args.bridge_min_evidence,
            min_score=args.bridge_min_score,
            require_endpoint_target=args.bridge_endpoints_only,
        )
        vessel_mask = continuity.bridged_mask
        bridge_pixel_mask = continuity.bridge_mask

        if continuity.bridges and args.bridge_inherit_probs:
            cnn_probs = inherit_softmax_at_bridges(cnn_probs, continuity)

    graph = build_vessel_graph(vessel_mask, **graph_kwargs_from_args(args))
    if graph.num_branches == 0:
        log.warning(
            "Post-processing skipped for %s: vessel graph has zero branches",
            label,
        )
        return None

    refined = refinement(
        cnn_probs=cnn_probs,
        vessel_graph=graph,
        vessel_mask=vessel_mask,
        use_branch_refinement=getattr(args, "use_branch_refine", True),
        branch_min_pixels=args.branch_min_pixels,
        branch_min_mean_prob=args.branch_min_mean_prob,
        branch_min_margin=args.branch_min_margin,
        branch_min_argmax_majority=args.branch_min_argmax_majority,
        branch_pixel_max_confidence=args.branch_pixel_max_conf,
        branch_use_confidence_weights=args.use_confidence_weights,
        propagate_bifurcations=args.propagate_bifurcations,
        propagation_min_neighbors=args.propagation_min_neighbors,
        propagation_require_consensus=not args.propagation_majority_only,
        propagation_max_iterations=args.propagation_max_iterations,
        remove_islands=args.remove_islands,
        island_min_size=args.island_min_size,
        island_opposite_ratio=args.island_opposite_ratio,
        island_dilation_radius=args.island_dilation_radius,
    )
    refined = Refinement(
        prediction=refined.prediction,
        branch_decisions=refined.branch_decisions,
        cnn_argmax=baseline_argmax,
        stages=refined.stages,
        before_propagation=refined.before_propagation,
        after_propagation=refined.after_propagation,
        propagation_segment_mask=refined.propagation_segment_mask,
    )

    if bridge_pixel_mask is not None and not getattr(args, "bridge_keep_pixels", False):
        prediction, stages = strip_bridge_pixels(
            refined.prediction,
            baseline_argmax,
            bridge_pixel_mask,
            refined.stages,
        )
        refined = Refinement(
            prediction=prediction,
            branch_decisions=refined.branch_decisions,
            cnn_argmax=refined.cnn_argmax,
            stages=stages,
            before_propagation=refined.before_propagation,
            after_propagation=refined.after_propagation,
            propagation_segment_mask=refined.propagation_segment_mask,
        )
        if refined.stages is not None:
            refined.stages["postprocessed"] = refined.prediction.copy()

    return refined, bridge_pixel_mask


def load_bundle(
    filename: str,
    predictions_dir: Path,
    test_img_dir: Path,
) -> dict[str, Any]:
    stem = os.path.splitext(filename)[0]
    gray = np.array(
        Image.open(test_img_dir / filename).convert("L"),
        dtype=np.uint8,
    )
    npz_path = predictions_dir / "test_softmax" / f"{stem}.npz"
    with np.load(npz_path) as data:
        cnn_probs = data["probs"].astype(np.float32)
    return {
        "filename": filename,
        "gray": gray,
        "cnn_probs": cnn_probs,
    }


def resolve_files(args: argparse.Namespace, predictions_dir: Path) -> list[str]:
    pred_dir = predictions_dir / "test_predictions"
    if args.file:
        return list(args.file)
    all_files = sorted(
        f.name for f in pred_dir.glob("*.png") if "_vessel" not in f.name
    )
    if args.all:
        return all_files
    return all_files[: max(0, int(args.num))]


def run_postprocessing(
    predictions_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    test_img_dir: str | Path | None = None,
    args: argparse.Namespace | None = None,
    process_all: bool = True,
    log: logging.Logger | None = None,
    **overrides: Any,
) -> Path:
    if args is None:
        args = build_namespace(process_all=process_all, **overrides)
    else:
        for key, value in overrides.items():
            setattr(args, key, value)

    args.predictions_dir = str(predictions_dir)
    if out_dir is not None:
        args.out_dir = str(out_dir)
    if test_img_dir is not None:
        args.test_img_dir = str(test_img_dir)

    predictions_path = Path(args.predictions_dir)
    test_img_path = Path(args.test_img_dir or Config.TEST_IMG_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(
        args.out_dir or (_REPO_ROOT / "outputs" / f"postprocess_{timestamp}")
    )

    pred_dir = out_path / "test_predictions"
    vis_dir = None if args.no_visualise else out_path / "test_visualizations"
    substeps_root = out_path / "substeps" if args.save_substeps else None

    log = log or setup_logging(str(out_path), name="RAVIR Post-processing")
    files = resolve_files(args, predictions_path)
    if not files:
        log.warning(
            "No prediction files found under %s",
            predictions_path / "test_predictions",
        )
        return out_path

    log.info("Post-processing %d image(s)", len(files))
    log.info("Predictions dir : %s", predictions_path)
    log.info("Output dir      : %s", out_path)

    saved = 0
    skipped = 0
    for filename in tqdm(files, desc="Post-processing"):
        try:
            bundle = load_bundle(filename, predictions_path, test_img_path)
        except (FileNotFoundError, OSError) as exc:
            log.warning("Skipping %s: %s", filename, exc)
            skipped += 1
            continue

        result = run_one(bundle, args, log=log)
        if result is None:
            skipped += 1
            continue

        refined, bridge_mask = result
        save_outputs(
            refined=refined,
            bundle=bundle,
            pred_dir=pred_dir,
            vis_dir=vis_dir,
            stages_root=substeps_root,
            bridge_mask=bridge_mask,
        )
        saved += 1

    log.info("Done: saved=%d skipped=%d", saved, skipped)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=str(_REPO_ROOT / "outputs" / "predictions"),
        help="Directory holding test_predictions/, test_softmax/ from CNN inference.",
    )
    parser.add_argument(
        "--test-img-dir",
        type=str,
        default=None,
        help="Original IR images. Defaults to Config.TEST_IMG_DIR.",
    )
    parser.add_argument(
        "--num", type=int, default=5, help="Number of images (default: %(default)s)."
    )
    parser.add_argument(
        "--file", nargs="+", default=None, help="Specific image filenames."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every image in test predictions.",
    )

    parser.add_argument(
        "--no-branch-refine",
        dest="use_branch_refine",
        action="store_false",
        help="Disable per-branch CNN-prior voting.",
    )
    parser.set_defaults(use_branch_refine=True)
    parser.add_argument("--branch-min-pixels", type=int, default=15)
    parser.add_argument("--branch-min-mean-prob", type=float, default=0.45)
    parser.add_argument("--branch-min-margin", type=float, default=0.05)
    parser.add_argument("--branch-min-argmax-majority", type=float, default=0.55)
    parser.add_argument("--branch-pixel-max-conf", type=float, default=0.85)
    parser.add_argument(
        "--no-confidence-weights",
        dest="use_confidence_weights",
        action="store_false",
    )
    parser.set_defaults(use_confidence_weights=True)

    parser.add_argument("--propagate-bifurcations", action="store_true")
    parser.add_argument("--propagation-min-neighbors", type=int, default=2)
    parser.add_argument("--propagation-majority-only", action="store_true")
    parser.add_argument("--propagation-max-iterations", type=int, default=5)

    parser.add_argument(
        "--no-remove-islands",
        dest="remove_islands",
        action="store_false",
    )
    parser.set_defaults(remove_islands=True)
    parser.add_argument("--island-min-size", type=int, default=15)
    parser.add_argument("--island-opposite-ratio", type=float, default=0.6)
    parser.add_argument("--island-dilation-radius", type=int, default=2)

    parser.add_argument("--bg-bias", type=float, default=1.0)
    parser.add_argument("--artery-bias", type=float, default=1.0)
    parser.add_argument("--vein-bias", type=float, default=1.0)
    parser.add_argument("--adaptive-background-bias", action="store_true")
    parser.add_argument(
        "--bg-bias-feature",
        type=str,
        default="vessel_fraction",
        choices=[
            "vessel_fraction",
            "mean_max_conf",
            "mean_vessel_prob",
            "mean_argmax_conf",
        ],
    )
    parser.add_argument("--bg-bias-base", type=float, default=1.25)
    parser.add_argument("--bg-bias-slope", type=float, default=0.0)
    parser.add_argument("--bg-bias-pivot", type=float, default=None)
    parser.add_argument("--bg-bias-clip-min", type=float, default=1.00)
    parser.add_argument("--bg-bias-clip-max", type=float, default=1.50)

    parser.add_argument("--no-bridge-gaps", dest="bridge_gaps", action="store_false")
    parser.set_defaults(bridge_gaps=True)
    parser.add_argument("--bridge-max-gap", type=int, default=12)
    parser.add_argument("--bridge-max-angle", type=float, default=35.0)
    parser.add_argument("--bridge-tangent-window", type=int, default=8)
    parser.add_argument("--bridge-min-evidence", type=float, default=0.15)
    parser.add_argument("--bridge-min-score", type=float, default=0.55)
    parser.add_argument("--bridge-endpoints-only", action="store_true")
    parser.add_argument(
        "--no-bridge-inherit-probs",
        dest="bridge_inherit_probs",
        action="store_false",
    )
    parser.set_defaults(bridge_inherit_probs=True)
    parser.add_argument("--bridge-keep-pixels", action="store_true")

    parser.add_argument(
        "--min-vessel-pixels", type=int, default=DEFAULT_MIN_VESSEL_PIXELS
    )
    parser.add_argument("--graph-close-radius", type=int, default=1)
    parser.add_argument("--graph-min-mask-object-size", type=int, default=25)
    parser.add_argument("--graph-min-branch-length", type=int, default=10)
    parser.add_argument("--graph-min-component-size", type=int, default=15)
    parser.add_argument("--graph-cleanup-iterations", type=int, default=3)
    parser.add_argument("--graph-dilation-radius", type=int, default=2)
    parser.add_argument("--graph-direction-pixels", type=int, default=3)

    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--no-visualise", action="store_true")
    parser.add_argument(
        "--save-substeps",
        action="store_true",
        help="Save intermediate post-processing maps under substeps/.",
    )

    return parser


def build_namespace(
    *,
    process_all: bool = True,
    **overrides: Any,
) -> argparse.Namespace:
    args = build_parser().parse_args([])
    if process_all:
        args.all = True
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def main() -> None:
    args = build_parser().parse_args()
    run_postprocessing(
        args.predictions_dir,
        out_dir=args.out_dir,
        test_img_dir=args.test_img_dir,
        args=args,
    )


if __name__ == "__main__":
    main()
