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

_NEIGHBOR_KERNEL = np.ones((3, 3), dtype=np.uint8)
_NEIGHBOR_KERNEL[1, 1] = 0


def skeleton_neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    skel = skeleton.astype(np.uint8)
    return convolve(skel, _NEIGHBOR_KERNEL, mode="constant", cval=0)


def endpoint_mask(skeleton: np.ndarray) -> np.ndarray:
    skel = skeleton.astype(bool)
    return skel & (skeleton_neighbor_count(skeleton) == 1)


def branch_point_mask(skeleton: np.ndarray) -> np.ndarray:
    skel = skeleton.astype(bool)
    return skel & (skeleton_neighbor_count(skeleton) >= 3)


def find_endpoints(skeleton: np.ndarray) -> np.ndarray:
    return np.argwhere(endpoint_mask(skeleton))


def skeleton_neighbors_at(
    mask: np.ndarray,
    y: int,
    x: int,
    *,
    exclude: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    h, w = mask.shape
    neighbors: list[tuple[int, int]] = []
    for dy, dx in NEIGHBOR_OFFSETS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
            if exclude is not None and (ny, nx) == exclude:
                continue
            neighbors.append((ny, nx))
    return neighbors


def order_neighbors(
    current: tuple[int, int],
    previous: tuple[int, int] | None,
    candidates: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not candidates:
        return []

    cy, cx = current

    def angle_from_current(point: tuple[int, int]) -> float:
        return float(np.arctan2(point[0] - cy, point[1] - cx))

    if previous is None:
        return sorted(candidates, key=angle_from_current)

    py, px = previous
    incoming = float(np.arctan2(cy - py, cx - px))

    def turn_magnitude(point: tuple[int, int]) -> float:
        outgoing = float(np.arctan2(point[0] - cy, point[1] - cx))
        diff = (outgoing - incoming + np.pi) % (2.0 * np.pi) - np.pi
        return abs(diff)

    return sorted(candidates, key=turn_magnitude)


def adjacent_node_label(
    node_labels: np.ndarray,
    y: int,
    x: int,
    *,
    exclude: int | None = None,
) -> int | None:
    h, w = node_labels.shape
    for dy, dx in NEIGHBOR_OFFSETS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            nl = int(node_labels[ny, nx])
            if nl > 0 and nl != exclude:
                return nl
    return None


def trace_skeleton_segment(
    component: np.ndarray,
    node_labels: np.ndarray,
    start: tuple[int, int],
    from_node_lbl: int,
) -> tuple[list[tuple[int, int]], int | None]:
    h, w = component.shape
    path: list[tuple[int, int]] = [start]
    visited = {start}
    previous: tuple[int, int] | None = None
    cy, cx = start

    end_node = adjacent_node_label(node_labels, cy, cx, exclude=from_node_lbl)
    if end_node is not None:
        return path, end_node

    for _ in range(h * w):
        candidates = [
            n
            for n in skeleton_neighbors_at(component, cy, cx, exclude=previous)
            if n not in visited and int(node_labels[n[0], n[1]]) == 0
        ]
        if not candidates:
            end_node = adjacent_node_label(node_labels, cy, cx, exclude=from_node_lbl)
            return path, end_node

        nxt = order_neighbors((cy, cx), previous, candidates)[0]
        path.append(nxt)
        visited.add(nxt)
        previous = (cy, cx)
        cy, cx = nxt

        end_node = adjacent_node_label(node_labels, cy, cx, exclude=from_node_lbl)
        if end_node is not None:
            return path, end_node

    return path, None


def walk_branch(
    skeleton: np.ndarray,
    start: tuple[int, int],
    max_steps: int,
) -> list[tuple[int, int]]:
    h, w = skeleton.shape
    visited: set[tuple[int, int]] = {start}
    path: list[tuple[int, int]] = [start]
    previous: tuple[int, int] | None = None
    cy, cx = start

    for _ in range(max_steps):
        candidates = [
            n
            for n in skeleton_neighbors_at(skeleton, cy, cx, exclude=previous)
            if n not in visited
        ]
        if not candidates:
            break

        nxt = order_neighbors((cy, cx), previous, candidates)[0]
        visited.add(nxt)
        path.append(nxt)
        previous = (cy, cx)
        cy, cx = nxt

    return path


def prune_skeleton(
    skeleton: np.ndarray,
    min_branch_length: int = 5,
    max_iterations: int = 10,
) -> np.ndarray:
    if min_branch_length <= 0:
        return skeleton.copy()

    skel = skeleton.astype(bool).copy()
    h, w = skel.shape

    for _ in range(max_iterations):
        endpoints = endpoint_mask(skel)
        if not endpoints.any():
            break

        branch_points = branch_point_mask(skel)
        to_remove = np.zeros_like(skel)

        for y, x in np.argwhere(endpoints):
            path = [(int(y), int(x))]
            previous: tuple[int, int] | None = None
            cy, cx = int(y), int(x)
            remove_path = False

            for _ in range(min_branch_length):
                hit_branch_point = False
                candidates: list[tuple[int, int]] = []

                for ny, nx in skeleton_neighbors_at(skel, cy, cx, exclude=previous):
                    if branch_points[ny, nx]:
                        hit_branch_point = True
                        break
                    candidates.append((ny, nx))

                if hit_branch_point:
                    remove_path = True
                    break
                if not candidates:
                    if len(path) < min_branch_length:
                        remove_path = True
                    break

                nxt = order_neighbors((cy, cx), previous, candidates)[0]
                previous = (cy, cx)
                cy, cx = nxt
                path.append((cy, cx))

            if remove_path:
                for py, px in path:
                    to_remove[py, px] = True

        if not to_remove.any():
            break

        skel &= ~to_remove

    return skel
