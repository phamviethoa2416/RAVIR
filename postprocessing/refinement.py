from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import label as sk_label


@dataclass
class BranchDecisions:
    label: np.ndarray
    score: np.ndarray
    mean_a: np.ndarray
    mean_v: np.ndarray
    num_pixels: np.ndarray

    @property
    def num_branches(self) -> int:
        return int(self.label.shape[0]) - 1

    def num_decided(self) -> int:
        return int((self.label > 0).sum())

    def num_decided_per_class(self) -> tuple[int, int]:
        return int((self.label == 1).sum()), int((self.label == 2).sum())


def compute_branch_decisions(
    cnn_probs: np.ndarray,
    vessel_mask: np.ndarray,
    segment_labels: np.ndarray,
    num_branches: int,
    *,
    min_branch_pixels: int = 10,
    min_mean_prob: float = 0.40,
    min_margin: float = 0.03,
    min_argmax_majority: float = 0.50,
    use_confidence_weights: bool = True,
) -> BranchDecisions:
    label = np.zeros(num_branches + 1, dtype=np.int32)
    score = np.zeros(num_branches + 1, dtype=np.float32)
    mean_a_arr = np.zeros(num_branches + 1, dtype=np.float32)
    mean_v_arr = np.zeros(num_branches + 1, dtype=np.float32)
    num_pixels = np.zeros(num_branches + 1, dtype=np.int32)

    if num_branches <= 0:
        return BranchDecisions(label, score, mean_a_arr, mean_v_arr, num_pixels)

    seg = segment_labels.astype(np.int64, copy=False)
    cnn_argmax = cnn_probs.argmax(axis=0).astype(np.int32)
    cnn_max = cnn_probs.max(axis=0).astype(np.float32)

    for bid in range(1, num_branches + 1):
        m = (seg == bid) & vessel_mask
        n = int(m.sum())
        num_pixels[bid] = n
        if n < min_branch_pixels:
            continue
        if use_confidence_weights:
            w = cnn_max[m]
            w_sum = float(w.sum())
            if w_sum <= 0.0:
                continue
            mean_a = float((cnn_probs[1][m] * w).sum()) / w_sum
            mean_v = float((cnn_probs[2][m] * w).sum()) / w_sum
        else:
            mean_a = float(cnn_probs[1][m].mean())
            mean_v = float(cnn_probs[2][m].mean())
        am = cnn_argmax[m]
        major_a = float((am == 1).mean())
        major_v = float((am == 2).mean())

        mean_a_arr[bid] = mean_a
        mean_v_arr[bid] = mean_v

        if (
            mean_a > mean_v + min_margin
            and mean_a >= min_mean_prob
            and major_a >= min_argmax_majority
        ):
            label[bid] = 1
            score[bid] = float(np.clip((mean_a - mean_v) * major_a, 0.0, 1.0))
        elif (
            mean_v > mean_a + min_margin
            and mean_v >= min_mean_prob
            and major_v >= min_argmax_majority
        ):
            label[bid] = 2
            score[bid] = float(np.clip((mean_v - mean_a) * major_v, 0.0, 1.0))

    return BranchDecisions(label, score, mean_a_arr, mean_v_arr, num_pixels)


def propagate_at_bifurcations(
    decisions: BranchDecisions,
    vessel_graph,
    *,
    max_degree: int = 3,
    min_decided_neighbors: int = 2,
    require_consensus: bool = True,
    propagation_score: float = 0.30,
    max_iterations: int = 5,
) -> BranchDecisions:
    label = decisions.label.copy()
    score = decisions.score.copy()

    nodes = getattr(vessel_graph, "nodes", None) or []
    if not nodes:
        return BranchDecisions(
            label,
            score,
            decisions.mean_a,
            decisions.mean_v,
            decisions.num_pixels,
        )

    for _ in range(max(1, int(max_iterations))):
        changed = False
        for node in nodes:
            degree = int(node.get("degree", 0))
            if degree < 1 or degree > max_degree:
                continue
            branch_ids = node.get("branches", [])
            if not branch_ids:
                continue
            num_a = sum(1 for b in branch_ids if label[b] == 1)
            num_v = sum(1 for b in branch_ids if label[b] == 2)
            num_decided = num_a + num_v
            if num_decided < min_decided_neighbors:
                continue
            if require_consensus:
                if num_a > 0 and num_v > 0:
                    continue
                target = 1 if num_a > 0 else 2
            else:
                if num_a == num_v:
                    continue
                target = 1 if num_a > num_v else 2
            for b in branch_ids:
                if label[b] == 0:
                    label[b] = target
                    score[b] = float(propagation_score)
                    changed = True
        if not changed:
            break

    return BranchDecisions(
        label=label,
        score=score,
        mean_a=decisions.mean_a,
        mean_v=decisions.mean_v,
        num_pixels=decisions.num_pixels,
    )


def apply_branch_decisions(
    prediction: np.ndarray,
    cnn_probs: np.ndarray,
    vessel_mask: np.ndarray,
    segment_labels: np.ndarray,
    decisions: BranchDecisions,
    *,
    pixel_max_confidence: float = 0.85,
) -> np.ndarray:
    out = prediction.copy().astype(np.int32, copy=False)

    num_branches = decisions.num_branches
    if num_branches <= 0:
        return out

    cnn_argmax = cnn_probs.argmax(axis=0).astype(np.int32)
    cnn_max = cnn_probs.max(axis=0).astype(np.float32)

    seg = segment_labels.astype(np.int64, copy=False)
    in_range = (seg >= 0) & (seg <= num_branches)
    decided_map = np.zeros_like(out)
    decided_map[in_range] = decisions.label[seg[in_range]]

    flip_mask = (
        vessel_mask
        & (decided_map > 0)
        & (decided_map != cnn_argmax)
        & (cnn_max < pixel_max_confidence)
    )

    out[flip_mask] = decided_map[flip_mask]
    return out


def remove_micro_islands(
    prediction: np.ndarray,
    vessel_mask: np.ndarray,
    *,
    min_size: int = 15,
    opposite_neighbor_threshold: float = 0.6,
    dilation_radius: int = 2,
) -> np.ndarray:
    out = prediction.copy().astype(np.int32, copy=False)
    if min_size <= 0 or not vessel_mask.any():
        return out

    structure = np.ones((dilation_radius * 2 + 1, dilation_radius * 2 + 1), dtype=bool)

    for cls in (1, 2):
        opposite = 2 if (cls == 1) else 1
        class_mask = out == cls
        if not class_mask.any():
            continue
        cc, n_cc = sk_label(class_mask, connectivity=2, return_num=True)
        if n_cc == 0:
            continue
        for cid in range(1, n_cc + 1):
            comp = cc == cid
            n = int(comp.sum())
            if n == 0 or n >= min_size:
                continue
            dilated = binary_dilation(comp, structure=structure) & vessel_mask
            border = dilated & (~comp)
            num_border = int(border.sum())
            if num_border == 0:
                continue
            num_opposite = int(np.sum(out[border] == opposite))
            if num_opposite / float(num_border) >= opposite_neighbor_threshold:
                out[comp] = opposite

    return out


@dataclass
class Refinement:
    prediction: np.ndarray
    branch_decisions: BranchDecisions | None = None
    cnn_argmax: np.ndarray | None = None
    stages: dict[str, np.ndarray] | None = None


def refinement(
    cnn_probs: np.ndarray,
    vessel_graph: Any,
    *,
    vessel_mask: np.ndarray | None = None,
    branch_min_pixels: int = 15,
    branch_min_mean_prob: float = 0.45,
    branch_min_margin: float = 0.05,
    branch_min_argmax_majority: float = 0.55,
    branch_pixel_max_confidence: float = 0.85,
    branch_use_confidence_weights: bool = True,
    propagate_bifurcations: bool = False,
    propagation_min_neighbors: int = 2,
    propagation_require_consensus: bool = True,
    propagation_max_iterations: int = 5,
    propagation_score: float = 0.30,
    remove_islands: bool = True,
    island_min_size: int = 15,
    island_opposite_ratio: float = 0.6,
    island_dilation_radius: int = 2,
) -> Refinement:
    if cnn_probs.ndim != 3 or cnn_probs.shape[0] < 3:
        raise ValueError(
            f"CNN probabilities must have dimension 3 or greater; got {cnn_probs.shape}"
        )

    vessel_mask = (
        vessel_mask.astype(bool, copy=False)
        if vessel_mask is not None
        else cnn_probs.argmax(axis=0) > 0
    )

    cnn_argmax = cnn_probs.argmax(axis=0).astype(np.int32)
    cnn_argmax[~vessel_mask] = 0
    stages: dict[str, np.ndarray] = {"prediction": cnn_argmax.copy()}

    pred = cnn_argmax.copy()
    decisions: BranchDecisions | None = None

    num_branches = int(getattr(vessel_graph, "num_branches", 0))
    decisions = compute_branch_decisions(
        cnn_probs=cnn_probs,
        vessel_mask=vessel_mask,
        segment_labels=vessel_graph.segment_labels,
        num_branches=num_branches,
        min_branch_pixels=branch_min_pixels,
        min_mean_prob=branch_min_mean_prob,
        min_margin=branch_min_margin,
        min_argmax_majority=branch_min_argmax_majority,
        use_confidence_weights=branch_use_confidence_weights,
    )
    if propagate_bifurcations:
        decisions = propagate_at_bifurcations(
            decisions,
            vessel_graph,
            min_decided_neighbors=propagation_min_neighbors,
            require_consensus=propagation_require_consensus,
            propagation_score=propagation_score,
            max_iterations=propagation_max_iterations,
        )
    pred = apply_branch_decisions(
        prediction=pred,
        cnn_probs=cnn_probs,
        vessel_mask=vessel_mask,
        segment_labels=vessel_graph.segment_labels,
        decisions=decisions,
        pixel_max_confidence=branch_pixel_max_confidence,
    )
    stages["branch_refined"] = pred.copy()

    if remove_islands:
        pred = remove_micro_islands(
            prediction=pred,
            vessel_mask=vessel_mask,
            min_size=island_min_size,
            opposite_neighbor_threshold=island_opposite_ratio,
            dilation_radius=island_dilation_radius,
        )
    stages["island_removed"] = pred.copy()
    stages["postprocessed"] = pred.copy()

    return Refinement(
        prediction=pred,
        branch_decisions=decisions,
        cnn_argmax=cnn_argmax,
        stages=stages,
    )
