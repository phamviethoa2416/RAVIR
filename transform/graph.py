from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    convolve,
    distance_transform_edt,
    label as cc_label,
)
from skimage.morphology import disk, skeletonize


def filter_components(
    cc: np.ndarray, num_components: int, min_pixels: int
) -> tuple[np.ndarray, int]:
    if num_components == 0:
        return cc, 0

    counts = np.bincount(cc.ravel(), minlength=num_components + 1)
    keep_mask = counts >= min_pixels
    keep_mask[0] = False
    num_dropped = int((~keep_mask & (counts > 0)).sum())

    lookup = np.zeros(num_components + 1, dtype=cc.dtype)
    lookup[keep_mask] = np.arange(1, keep_mask.sum() + 1, dtype=cc.dtype)
    return lookup[cc], num_dropped


def merge_components(
    cc: np.ndarray,
    num_components: int,
    min_pixels: int,
) -> tuple[np.ndarray, int]:
    if num_components == 0:
        return cc, 0

    counts = np.bincount(cc.ravel(), minlength=num_components + 1)
    keep = counts >= min_pixels
    keep[0] = False
    small_ids = np.where((counts > 0) & ~keep)[0]
    small_ids = small_ids[small_ids > 0]

    if small_ids.size == 0:
        return cc, 0
    if not keep.any():
        dropped, num_dropped = filter_components(cc, num_components, min_pixels)
        return dropped, num_dropped

    is_kept = np.isin(cc, np.where(keep)[0])
    _, indices = distance_transform_edt(
        ~is_kept, return_distances=True, return_indices=True
    )
    nearest = cc[tuple(indices)]

    out = cc.copy()
    small_mask = np.isin(cc, small_ids)
    num_merged = int(small_mask.sum())
    out[small_mask] = nearest[small_mask]
    return out, num_merged


def detect_av_crossings(
    a_mask: np.ndarray,
    v_mask: np.ndarray,
    crossing_radius: int,
    node_proximity: int,
) -> np.ndarray:
    union = a_mask | v_mask
    if not union.any():
        return np.zeros_like(union, dtype=bool)

    skel = skeletonize(union)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
    neighbors = convolve(skel.astype(np.int32), kernel, mode="constant", cval=0)
    branch_points = skel & (neighbors >= 3)
    if not branch_points.any():
        return np.zeros_like(union, dtype=bool)

    near = disk(node_proximity)
    a_near = binary_dilation(a_mask, near)
    v_near = binary_dilation(v_mask, near)
    av_nodes = branch_points & a_near & v_near
    if not av_nodes.any():
        return np.zeros_like(union, dtype=bool)

    return binary_dilation(av_nodes, disk(crossing_radius))


def relabel(cc: np.ndarray) -> tuple[np.ndarray, int]:
    unique = np.unique(cc)
    unique = unique[unique != 0]
    if unique.size == 0:
        return cc, 0

    lookup = np.zeros(int(unique.max()) + 1, dtype=cc.dtype)
    lookup[unique] = np.arange(1, unique.size + 1, dtype=cc.dtype)
    return lookup[cc], int(unique.size)


def compute_branch_labels(
    class_mask: np.ndarray,
    crossing_radius: int = 1,
    min_branch_pixels: int = 8,
    node_proximity: int = 5,
    small_component_mode: str = "drop",
) -> np.ndarray:
    H, W = class_mask.shape
    a_mask = class_mask == 1
    v_mask = class_mask == 2
    if not a_mask.any() and not v_mask.any():
        return np.zeros((H, W), dtype=np.uint16)

    crossing_region = (
        detect_av_crossings(a_mask, v_mask, crossing_radius, node_proximity)
        if crossing_radius > 0
        else np.zeros_like(a_mask, dtype=bool)
    )

    a_safe = a_mask & ~crossing_region
    v_safe = v_mask & ~crossing_region

    structure = np.ones((3, 3), dtype=np.int32)
    a_cc, num_a = cc_label(a_safe, structure=structure)
    v_cc, num_v = cc_label(v_safe, structure=structure)

    if min_branch_pixels > 1:
        handler = (
            merge_components if small_component_mode == "merge" else filter_components
        )
        a_cc, _ = handler(a_cc, num_a, min_branch_pixels)
        v_cc, _ = handler(v_cc, num_v, min_branch_pixels)

    a_cc, num_a = relabel(a_cc)
    v_cc, num_v = relabel(v_cc)

    branch_labels = np.zeros((H, W), dtype=np.uint16)
    branch_labels[a_cc > 0] = a_cc[a_cc > 0].astype(np.uint16)
    branch_labels[v_cc > 0] = (v_cc[v_cc > 0] + num_a).astype(np.uint16)
    return branch_labels
