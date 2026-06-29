from __future__ import annotations

import argparse
import os
from datetime import datetime

from config import Config
from train import train
from utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAVIR training entry point.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser(
        "one-fold",
        help="Train validation fold 0 only.",
    )

    kfold = subparsers.add_parser(
        "kfold",
        help=f"Train all {Config.NUM_FOLDS} cross-validation folds sequentially.",
    )
    kfold.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Parent directory for all fold runs "
            f"(default: outputs/kfold_<timestamp>). "
            f"Each fold is saved under fold_0 … fold_{Config.NUM_FOLDS - 1}."
        ),
    )

    return parser


def run_one_fold() -> None:
    train(val_fold=0)


def run_kfold(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = args.output_dir or os.path.join(
        Config.OUTPUT_DIR, f"kfold_{timestamp}"
    )
    os.makedirs(parent_dir, exist_ok=True)

    fold_logger = setup_logging(parent_dir)
    fold_logger.info("=" * 60)
    fold_logger.info("  RAVIR K-Fold Training (%d folds)", Config.NUM_FOLDS)
    fold_logger.info("=" * 60)
    fold_logger.info("Parent directory: %s", parent_dir)

    results: list[tuple[int, float, str]] = []
    for fold in range(Config.NUM_FOLDS):
        fold_logger.info("-" * 60)
        fold_logger.info("Starting fold %d / %d", fold, Config.NUM_FOLDS)
        fold_dir = os.path.join(parent_dir, f"fold_{fold}")
        best_dice, run_dir = train(val_fold=fold, run_dir=fold_dir)
        results.append((fold, best_dice, run_dir))
        fold_logger.info(
            "Fold %d complete — best Dice=%.4f, run_dir=%s",
            fold,
            best_dice,
            run_dir,
        )

    mean_dice = sum(score for _, score, _ in results) / len(results)
    fold_logger.info("=" * 60)
    fold_logger.info("  K-Fold Summary")
    fold_logger.info("=" * 60)
    for fold, score, run_dir in results:
        fold_logger.info("  Fold %d : Dice=%.4f  (%s)", fold, score, run_dir)
    fold_logger.info("  Mean Dice : %.4f", mean_dice)
    fold_logger.info("  Checkpoints:")
    for fold, _, run_dir in results:
        fold_logger.info("    fold_%d : %s", fold, os.path.join(run_dir, "best_model.pth"))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "one-fold":
        run_one_fold()
    elif args.mode == "kfold":
        run_kfold(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
