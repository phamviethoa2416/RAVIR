from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

NEIGHBOR_KERNEL = np.ones((3, 3), dtype=np.uint8)
NEIGHBOR_KERNEL[1, 1] = 0


def skeleton_neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    skel = skeleton.astype(np.uint8)
    return convolve(skel, NEIGHBOR_KERNEL, mode="constant", cval=0)


def endpoint_mask(skeleton: np.ndarray) -> np.ndarray:
    skel = skeleton.astype(bool)
    return skel & (skeleton_neighbor_count(skeleton) == 1)


def branch_point_mask(skeleton: np.ndarray) -> np.ndarray:
    skel = skeleton.astype(bool)
    return skel & (skeleton_neighbor_count(skeleton) >= 3)


def find_endpoints(skeleton: np.ndarray) -> np.ndarray:
    return np.argwhere(endpoint_mask(skeleton))
