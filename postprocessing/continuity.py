from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as bresenham_line
from skimage.morphology import dilation, disk, skeletonize

from postprocessing.utils import find_endpoints, walk_branch

W_DIST = 0.30
W_ANGLE = 0.40
W_EVIDENCE = 0.20
W_CLASS = 0.10
EVIDENCE_NORM_DENOM = 0.50
CLASS_MATCH_SCORE = 1.0
CLASS_MISMATCH_SCORE = 0.60
MIN_UNIQUE_FRAC = 0.50

_IMAGE_FEATURES = frozenset(
    {"vessel_fraction", "mean_max_conf", "mean_vessel_prob", "mean_argmax_conf"}
)


@dataclass
class Bridge:
    endpoint: tuple[int, int]
    target: tuple[int, int]
    distance: float
    angle_deviation_deg: float
    cnn_evidence: float
    class_match: bool
    score: float
    pixels: np.ndarray
    source_path: list[tuple[int, int]] = field(default_factory=list)
    target_path: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ContinuityResult:
    bridged_mask: np.ndarray
    bridge_mask: np.ndarray
    bridges: list[Bridge] = field(default_factory=list)
    n_endpoints_before: int = 0
    n_endpoints_after: int = 0
    n_components_before: int = 0
    n_components_after: int = 0


def outward_tangent(path: list[tuple[int, int]]) -> np.ndarray | None:
    if len(path) < 3:
        return None

    points = np.asarray(path, dtype=np.float32)
    centred = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    direction = vh[0]
    if np.dot(direction, points[-1] - points[0]) > 0:
        direction = -direction

    norm = np.linalg.norm(direction)
    return None if norm < 1e-6 else (direction / norm).astype(np.float32)


def foreground_prob_along(
    cnn_probs: np.ndarray | None,
    rr: np.ndarray,
    cc: np.ndarray,
) -> float:
    if cnn_probs is None:
        return 1.0
    h, w = cnn_probs.shape[1:]
    rr = np.clip(rr, 0, h - 1)
    cc = np.clip(cc, 0, w - 1)
    return float((cnn_probs[1, rr, cc] + cnn_probs[2, rr, cc]).mean())


def endpoints_class_match(
    cnn_probs: np.ndarray | None,
    p0: tuple[int, int],
    p1: tuple[int, int],
) -> bool:
    if cnn_probs is None:
        return True
    a = int(cnn_probs[:, p0[0], p0[1]].argmax())
    b = int(cnn_probs[:, p1[0], p1[1]].argmax())
    return a == b and a > 0


def median_vessel_width(edt: np.ndarray, path: list[tuple[int, int]]) -> int:
    if not path:
        return 1
    ys, xs = zip(*path)
    radii = edt[np.asarray(ys), np.asarray(xs)]
    radii = radii[radii > 0]
    return max(int(round(float(np.median(radii)))), 1) if radii.size else 1


def composite_score(
    dist: float,
    max_gap: float,
    cos_dev: float,
    evidence: float,
    class_match: bool,
) -> float:
    dist_score = 1.0 - dist / max_gap
    angle_score = float(cos_dev)
    evidence_score = min(evidence / EVIDENCE_NORM_DENOM, 1.0)
    class_score = CLASS_MATCH_SCORE if class_match else CLASS_MISMATCH_SCORE
    return (
        W_DIST * dist_score
        + W_ANGLE * angle_score
        + W_EVIDENCE * evidence_score
        + W_CLASS * class_score
    )


def find_best_candidate(
    ep: tuple[int, int],
    t_hat: np.ndarray,
    sk_yx: np.ndarray,
    endpoint_set: set[tuple[int, int]],
    cnn_probs: np.ndarray | None,
    max_gap: int,
    cos_thresh: float,
    min_cnn_evidence: float,
    min_score: float,
    require_endpoint_target: bool,
) -> tuple[tuple[int, int], dict] | None:
    ey, ex = ep
    origin = np.array([ey, ex], dtype=np.float32)
    deltas = sk_yx.astype(np.float32) - origin
    d2 = (deltas * deltas).sum(axis=1)
    in_range = (d2 <= float(max_gap) ** 2) & (d2 > 1.0)
    if not in_range.any():
        return None

    candidates = sk_yx[in_range]
    dists = np.sqrt(d2[in_range])
    dirs = (candidates - origin) / dists[:, None]
    cos_dev = dirs @ t_hat
    in_cone = cos_dev >= cos_thresh
    if not in_cone.any():
        return None

    candidates = candidates[in_cone]
    dists = dists[in_cone]
    cos_dev = cos_dev[in_cone]

    if require_endpoint_target:
        ep_mask = np.array(
            [(int(y), int(x)) in endpoint_set for y, x in candidates],
            dtype=bool,
        )
        if not ep_mask.any():
            return None
        candidates = candidates[ep_mask]
        dists = dists[ep_mask]
        cos_dev = cos_dev[ep_mask]

    best_score = -1.0
    best_target: tuple[int, int] | None = None
    best_extras: dict | None = None

    for (cy, cx), d, cos_i in zip(candidates, dists, cos_dev):
        target = (int(cy), int(cx))
        if target == ep:
            continue

        rr, cc = bresenham_line(ey, ex, int(cy), int(cx))
        if rr.size <= 2:
            continue

        evidence = foreground_prob_along(cnn_probs, rr[1:-1], cc[1:-1])
        if evidence < min_cnn_evidence:
            continue

        class_match = endpoints_class_match(cnn_probs, ep, target)
        score = composite_score(float(d), float(max_gap), float(cos_i), evidence, class_match)
        if score > best_score:
            best_score = score
            best_target = target
            best_extras = {
                "distance": float(d),
                "angle_dev": float(
                    np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0)))
                ),
                "evidence": evidence,
                "class_match": class_match,
                "score": score,
                "rr": rr,
                "cc": cc,
            }

    if best_target is None or best_score < min_score or best_extras is None:
        return None
    return best_target, best_extras


def paint_bridge(
    rr: np.ndarray,
    cc: np.ndarray,
    radius: int,
    shape: tuple[int, int],
) -> np.ndarray:
    line_mask = np.zeros(shape, dtype=bool)
    line_mask[rr, cc] = True
    if radius >= 1:
        return dilation(line_mask, footprint=disk(radius)).astype(bool)
    return line_mask


def bridge_vessel_gaps(
    vessel_mask: np.ndarray,
    cnn_probs: np.ndarray | None = None,
    *,
    max_gap: int = 12,
    max_angle_dev_deg: float = 35.0,
    tangent_window: int = 8,
    min_cnn_evidence: float = 0.15,
    min_score: float = 0.55,
    require_endpoint_target: bool = False,
    bridge_dilation: int | None = None,
) -> ContinuityResult:
    vessel_mask = vessel_mask.astype(bool)
    shape = vessel_mask.shape

    skeleton = skeletonize(vessel_mask)
    endpoints_arr = find_endpoints(skeleton)
    endpoint_set = {(int(y), int(x)) for y, x in endpoints_arr}
    edt = ndi.distance_transform_edt(vessel_mask)
    sk_yx = np.argwhere(skeleton)
    cos_thresh = float(np.cos(np.deg2rad(max_angle_dev_deg)))

    paths: dict[tuple[int, int], list[tuple[int, int]]] = {}
    tangents: dict[tuple[int, int], np.ndarray] = {}
    for ep in endpoint_set:
        path = walk_branch(skeleton, ep, tangent_window)
        paths[ep] = path
        tangent = outward_tangent(path)
        if tangent is not None:
            tangents[ep] = tangent

    n_components_before = int(ndi.label(vessel_mask)[1])
    n_endpoints_before = int(endpoints_arr.shape[0])

    bridged = vessel_mask.copy()
    bridge_pixel_mask = np.zeros_like(vessel_mask)
    used_endpoints: set[tuple[int, int]] = set()
    bridges: list[Bridge] = []

    for ep in sorted(tangents.keys(), key=lambda p: -len(paths[p])):
        if ep in used_endpoints:
            continue

        match = find_best_candidate(
            ep=ep,
            t_hat=tangents[ep],
            sk_yx=sk_yx,
            endpoint_set=endpoint_set,
            cnn_probs=cnn_probs,
            max_gap=max_gap,
            cos_thresh=cos_thresh,
            min_cnn_evidence=min_cnn_evidence,
            min_score=min_score,
            require_endpoint_target=require_endpoint_target,
        )
        if match is None:
            continue

        target, extras = match
        radius = (
            int(bridge_dilation)
            if bridge_dilation is not None
            else median_vessel_width(edt, paths[ep])
        )
        painted = paint_bridge(extras["rr"], extras["cc"], radius, shape)
        added = painted & ~bridged
        if not painted.any():
            continue
        if float(added.sum()) / float(painted.sum()) < MIN_UNIQUE_FRAC:
            continue

        bridged |= added
        bridge_pixel_mask |= added

        if target in paths:
            target_path = paths[target]
        elif target in endpoint_set:
            target_path = walk_branch(skeleton, target, tangent_window)
        else:
            target_path = []

        bridges.append(
            Bridge(
                endpoint=ep,
                target=target,
                distance=extras["distance"],
                angle_deviation_deg=extras["angle_dev"],
                cnn_evidence=extras["evidence"],
                class_match=extras["class_match"],
                score=extras["score"],
                pixels=np.argwhere(added),
                source_path=list(paths[ep]),
                target_path=list(target_path),
            )
        )
        used_endpoints.add(ep)
        if target in endpoint_set:
            used_endpoints.add(target)

    skeleton_after = skeletonize(bridged)
    n_endpoints_after = int(find_endpoints(skeleton_after).shape[0])
    n_components_after = int(ndi.label(bridged)[1])

    return ContinuityResult(
        bridged_mask=bridged,
        bridge_mask=bridge_pixel_mask,
        bridges=bridges,
        n_endpoints_before=n_endpoints_before,
        n_endpoints_after=n_endpoints_after,
        n_components_before=n_components_before,
        n_components_after=n_components_after,
    )


def inherit_softmax_at_bridges(
    cnn_probs: np.ndarray,
    result: ContinuityResult,
    *,
    inplace: bool = False,
) -> np.ndarray:
    if cnn_probs.ndim != 3:
        raise ValueError(f"cnn_probs must be (C, H, W); got {cnn_probs.shape}")

    out = cnn_probs if inplace else cnn_probs.copy()

    for br in result.bridges:
        sample_pts = br.source_path + br.target_path
        if not sample_pts or br.pixels.size == 0:
            continue

        ys = np.fromiter((y for y, _ in sample_pts), dtype=np.int64)
        xs = np.fromiter((x for _, x in sample_pts), dtype=np.int64)
        mean_prob = out[:, ys, xs].mean(axis=1)
        total = float(mean_prob.sum())
        if total > 1e-6:
            mean_prob = mean_prob / total

        out[:, br.pixels[:, 0], br.pixels[:, 1]] = mean_prob[:, None]

    return out


def apply_class_bias(
    cnn_probs: np.ndarray,
    *,
    bg_bias: float = 1.0,
    artery_bias: float = 1.0,
    vein_bias: float = 1.0,
) -> np.ndarray:
    if bg_bias == artery_bias == vein_bias == 1.0:
        return cnn_probs
    if cnn_probs.ndim != 3:
        raise ValueError(f"cnn_probs must be (C, H, W); got {cnn_probs.shape}")

    biases = np.array(
        [bg_bias, artery_bias, vein_bias], dtype=cnn_probs.dtype
    )[: cnn_probs.shape[0]]
    scaled = cnn_probs * biases[:, None, None]
    denom = scaled.sum(axis=0, keepdims=True)
    denom = np.where(denom > 1e-12, denom, 1.0)
    return scaled / denom


def image_feature(cnn_probs: np.ndarray, feature: str) -> float:
    if cnn_probs.ndim != 3 or cnn_probs.shape[0] < 3:
        raise ValueError(f"Expected (C>=3, H, W); got {cnn_probs.shape}")
    if feature not in _IMAGE_FEATURES:
        raise ValueError(
            f"Unknown feature {feature!r}; choose from {sorted(_IMAGE_FEATURES)}"
        )

    argmax = cnn_probs.argmax(axis=0)
    if feature == "vessel_fraction":
        return float((argmax > 0).mean())
    if feature == "mean_max_conf":
        return float(cnn_probs.max(axis=0).mean())
    if feature == "mean_vessel_prob":
        return float((cnn_probs[1] + cnn_probs[2]).mean())

    vessel_mask = argmax > 0
    return float(cnn_probs.max(axis=0)[vessel_mask].mean()) if vessel_mask.any() else 0.0


def compute_image_bg_bias(
    cnn_probs: np.ndarray,
    *,
    feature: str = "vessel_fraction",
    base: float = 1.25,
    slope: float = 0.0,
    pivot: float | None = None,
    clip: tuple[float, float] = (1.00, 1.50),
) -> tuple[float, float]:
    feat = image_feature(cnn_probs, feature)
    if slope == 0.0 or pivot is None:
        return float(np.clip(base, *clip)), feat
    bias = base + slope * (feat - float(pivot))
    return float(np.clip(bias, clip[0], clip[1])), feat


__all__ = [
    "Bridge",
    "ContinuityResult",
    "apply_class_bias",
    "bridge_vessel_gaps",
    "compute_image_bg_bias",
    "inherit_softmax_at_bridges",
]
