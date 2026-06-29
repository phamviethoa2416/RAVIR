from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_closing
from skimage.measure import label as sk_label
from skimage.morphology import disk, remove_small_objects, skeletonize

from postprocessing.utils import (
    NEIGHBOR_OFFSETS,
    branch_point_mask,
    endpoint_mask,
    prune_skeleton,
    trace_skeleton_segment,
)


@dataclass
class VesselGraph:
    skeleton: np.ndarray
    branch_points: np.ndarray
    endpoints: np.ndarray
    branch_labels: np.ndarray
    segment_labels: np.ndarray
    num_branches: int

    nodes: list[dict] = field(default_factory=list)
    branches: list[dict] = field(default_factory=list)
    adjacency: dict[int, list[int]] = field(default_factory=dict)

    def node_by_id(self, node_id: int) -> dict:
        return self.nodes[node_id]

    def branch_by_id(self, branch_id: int) -> dict:
        return self.branches[branch_id - 1]

    def crossing_candidates(self) -> list[int]:
        return [node["id"] for node in self.nodes if node["degree"] >= 4]

    def bifurcation_candidates(self) -> list[int]:
        return [node["id"] for node in self.nodes if node["degree"] == 3]


def _attachment_pixel(
    ordered: list[tuple[int, int]],
    node_pixels: list[tuple[int, int]],
) -> tuple[int, int]:
    if not ordered:
        return node_pixels[0]

    node_set = set(node_pixels)
    for y, x in ordered:
        if (y, x) in node_set:
            return y, x
        for dy, dx in NEIGHBOR_OFFSETS:
            ny, nx = y + dy, x + dx
            if (ny, nx) in node_set:
                return y, x

    pts_arr = np.asarray(ordered, dtype=np.float64)
    nodes_arr = np.asarray(node_pixels, dtype=np.float64)
    dists = ((pts_arr[:, None, :] - nodes_arr[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    return ordered[int(np.argmin(dists))]


def compute_branch_directions(
    branch_pixels: list[tuple[int, int]] | np.ndarray,
    node_a_pixels: list[tuple[int, int]] | None = None,
    node_b_pixels: list[tuple[int, int]] | None = None,
    num_pixels_for_direction: int = 5,
) -> tuple[float, float]:
    pixels = np.asarray(branch_pixels, dtype=np.int32)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape[0] == 0:
        return 0.0, 0.0

    ordered = [(int(y), int(x)) for y, x in pixels]
    k = int(max(2, min(num_pixels_for_direction, len(ordered))))
    head = ordered[:k]
    tail = ordered[-k:]

    if node_a_pixels:
        origin_a = np.asarray(
            _attachment_pixel(ordered[: max(k, 3)], node_a_pixels),
            dtype=np.float64,
        )
    else:
        origin_a = np.asarray(head[0], dtype=np.float64)

    if node_b_pixels:
        origin_b = np.asarray(
            _attachment_pixel(ordered[-max(k, 3) :], node_b_pixels),
            dtype=np.float64,
        )
    else:
        origin_b = np.asarray(tail[-1], dtype=np.float64)

    head_end = np.asarray(head[-1], dtype=np.float64)
    tail_start = np.asarray(tail[0], dtype=np.float64)
    dy_a, dx_a = head_end - origin_a
    angle_a = float(np.arctan2(dy_a, dx_a))
    dy_b, dx_b = tail_start - origin_b
    angle_b = float(np.arctan2(dy_b, dx_b))

    return angle_a, angle_b


def assign_vessel_pixels(
    vessel_mask: np.ndarray,
    branch_labels: np.ndarray,
    dilation_radius: int = 2,
) -> np.ndarray:
    vessel_mask = vessel_mask.astype(bool)
    skeleton_labels = branch_labels.astype(np.int32, copy=False)
    if not vessel_mask.any() or not (skeleton_labels > 0).any():
        return np.zeros_like(skeleton_labels, dtype=np.int32)

    dist, nearest_idx = distance_transform_edt(
        ~(skeleton_labels > 0),
        return_distances=True,
        return_indices=True,
    )
    assigned = skeleton_labels[nearest_idx[0], nearest_idx[1]].astype(np.int32)

    if dilation_radius > 0:
        assigned = np.where(dist <= float(dilation_radius), assigned, 0)

    segmentation = np.where(vessel_mask, assigned, 0).astype(np.int32)
    return segmentation


def build_vessel_graph(
    vessel_mask: np.ndarray,
    *,
    close_radius: int = 1,
    min_mask_object_size: int = 25,
    min_branch_length: int = 10,
    min_component_size: int = 15,
    cleanup_iterations: int = 3,
    dilation_radius: int = 2,
    num_pixels_for_direction: int = 3,
) -> VesselGraph:
    vessel_mask = np.asarray(vessel_mask).astype(bool)
    H, W = vessel_mask.shape
    dist_map = distance_transform_edt(vessel_mask)
    if not vessel_mask.any():
        empty_i32 = np.zeros((H, W), dtype=np.int32)
        empty_bool = np.zeros((H, W), dtype=bool)
        return VesselGraph(
            skeleton=empty_bool,
            branch_points=empty_bool,
            endpoints=empty_bool,
            branch_labels=empty_i32,
            segment_labels=empty_i32,
            num_branches=0,
        )

    # ── 1. Mask-level cleanup + skeletonization ──────────────────────────────
    clean_mask = vessel_mask
    if min_mask_object_size > 1:
        clean_mask = remove_small_objects(
            clean_mask, min_size=min_mask_object_size, connectivity=2
        )

    mask_for_skel = clean_mask
    if close_radius > 0:
        mask_for_skel = binary_closing(clean_mask, structure=disk(close_radius))
    skeleton = skeletonize(mask_for_skel).astype(bool)

    # ── 2. Alternating spur-prune + small-component removal ─────────────────
    for _ in range(max(1, cleanup_iterations)):
        before = int(skeleton.sum())
        if min_branch_length > 0:
            skeleton = prune_skeleton(skeleton, min_branch_length=min_branch_length)
        # Remove small skeleton components
        if min_component_size > 1 and skeleton.any():
            skeleton = remove_small_objects(
                skeleton.astype(bool), min_size=min_component_size, connectivity=2
            )
        if int(skeleton.sum()) == before:
            break

    # ── 3. Branch points / endpoints ────────────────────────────────────────
    branch_points = branch_point_mask(skeleton)
    endpoints = endpoint_mask(skeleton)

    # Removes BP clusters that are not real junctions (< 3 outgoing directions).
    bp = branch_points.astype(bool).copy()
    for _ in range(5):
        comps = sk_label(bp, connectivity=2).astype(np.int32)
        n_comp = int(comps.max())
        if n_comp == 0:
            break
        skel_no_bp = skeleton & ~bp
        external_comp = sk_label(skel_no_bp, connectivity=2).astype(np.int32)
        changed = False
        for cid in range(1, n_comp + 1):
            cluster = comps == cid
            ys, xs = np.where(cluster)
            cluster_size = len(ys)
            external_pixels: set[tuple[int, int]] = set()
            external_branches: set[int] = set()
            for y, x in zip(ys, xs):
                for dy, dx in NEIGHBOR_OFFSETS:
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < H and 0 <= nx < W):
                        continue
                    if not skeleton[ny, nx]:
                        continue
                    if cluster[ny, nx]:
                        continue
                    external_pixels.add((ny, nx))
                    ext_id = int(external_comp[ny, nx])
                    if ext_id > 0:
                        external_branches.add(ext_id)
            if cluster_size == 1:
                demote = len(external_pixels) < 3
            else:
                demote = len(external_branches) < 3
            if demote:
                bp[cluster] = False
                changed = True
        if not changed:
            break
    branch_pts = bp

    # ── 4. Node labeling ────────────────────────────────────────────────────
    node_labels = np.zeros((H, W), dtype=np.int32)
    bp_components = sk_label(branch_pts, connectivity=2).astype(np.int32)
    n_bp_components = int(bp_components.max())
    if n_bp_components > 0:
        node_labels[bp_components > 0] = bp_components[bp_components > 0]

    ep_only = endpoints & (node_labels == 0)
    n_nodes = n_bp_components
    for y, x in np.argwhere(ep_only):
        n_nodes += 1
        node_labels[y, x] = n_nodes

    nodes: list[dict] = []
    for nid in range(1, n_nodes + 1):
        component = node_labels == nid
        ys, xs = np.where(component)
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        node_type = "branch_point" if nid <= n_bp_components else "endpoint"
        nodes.append(
            {
                "id": nid - 1,
                "type": node_type,
                "position": (cy, cx),
                "degree": 0,
                "branches": [],
                "pixels": [(int(y), int(x)) for y, x in zip(ys, xs)],
            }
        )

    # ── 5. Absorb tight-loop bumps into BP clusters ──────────────────────────
    for _ in range(3):
        absorbed_any = False
        interior = skeleton & (node_labels == 0)
        ys, xs = np.where(interior)
        for y, x in zip(ys, xs):
            neighbour_node_ids: set[int] = set()
            neighbour_skel_count = 0
            for dy, dx in NEIGHBOR_OFFSETS:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if not skeleton[ny, nx]:
                    continue
                neighbour_skel_count += 1
                nl = int(node_labels[ny, nx])
                if nl > 0:
                    neighbour_node_ids.add(nl)
                else:
                    neighbour_node_ids.add(-1)
                    break
            if (
                neighbour_skel_count >= 2
                and len(neighbour_node_ids) == 1
                and -1 not in neighbour_node_ids
            ):
                nid = next(iter(neighbour_node_ids))
                if nodes[nid - 1]["type"] != "branch_point":
                    continue
                node_labels[y, x] = nid
                branch_pts[y, x] = True
                nodes[nid - 1]["pixels"].append((int(y), int(x)))
                absorbed_any = True
        if not absorbed_any:
            break

    # ── 6. Re-demote clusters after absorption ───────────────────────────────
    bp = branch_pts.astype(bool).copy()
    for _ in range(5):
        comps = sk_label(bp, connectivity=2).astype(np.int32)
        n_comp = int(comps.max())
        if n_comp == 0:
            break
        skel_no_bp = skeleton & ~bp
        external_comp = sk_label(skel_no_bp, connectivity=2).astype(np.int32)
        changed = False
        for cid in range(1, n_comp + 1):
            cluster = comps == cid
            ys, xs = np.where(cluster)
            cluster_size = len(ys)
            external_pixels = set()
            external_branches = set()
            for y, x in zip(ys, xs):
                for dy, dx in NEIGHBOR_OFFSETS:
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < H and 0 <= nx < W):
                        continue
                    if not skeleton[ny, nx]:
                        continue
                    if cluster[ny, nx]:
                        continue
                    external_pixels.add((ny, nx))
                    ext_id = int(external_comp[ny, nx])
                    if ext_id > 0:
                        external_branches.add(ext_id)
            if cluster_size == 1:
                demote = len(external_pixels) < 3
            else:
                demote = len(external_branches) < 3
            if demote:
                bp[cluster] = False
                changed = True
        if not changed:
            break
    branch_pts_new = bp

    if not np.array_equal(branch_pts_new, branch_pts):
        branch_pts = branch_pts_new
        # Rebuild node labels after demotion (inline _label_nodes again).
        node_labels = np.zeros((H, W), dtype=np.int32)
        bp_components = sk_label(branch_pts, connectivity=2).astype(np.int32)
        n_bp_components = int(bp_components.max())
        if n_bp_components > 0:
            node_labels[bp_components > 0] = bp_components[bp_components > 0]
        ep_only = endpoints & (node_labels == 0)
        n_nodes = n_bp_components
        for y, x in np.argwhere(ep_only):
            n_nodes += 1
            node_labels[y, x] = n_nodes
        nodes = []
        for nid in range(1, n_nodes + 1):
            component = node_labels == nid
            ys, xs = np.where(component)
            cy = float(np.mean(ys))
            cx = float(np.mean(xs))
            node_type = "branch_point" if nid <= n_bp_components else "endpoint"
            nodes.append(
                {
                    "id": nid - 1,
                    "type": node_type,
                    "position": (cy, cx),
                    "degree": 0,
                    "branches": [],
                    "pixels": [(int(y), int(x)) for y, x in zip(ys, xs)],
                }
            )

    # ── 7. Split skeleton into branch segments between nodes ───────────────
    interior = skeleton & (node_labels == 0)
    comp_labels, n_components = sk_label(interior, connectivity=2, return_num=True)

    branches: list[dict] = []
    branch_labels = np.zeros((H, W), dtype=np.int32)
    seen_segments: set[tuple[int, int, frozenset[tuple[int, int]]]] = set()

    def _register_branch(
        ordered: list[tuple[int, int]],
        node_a_id: int | None,
        node_b_id: int | None,
    ) -> None:
        if not ordered:
            return

        branch_id = len(branches) + 1
        for py, px in ordered:
            branch_labels[py, px] = branch_id

        node_a_pixels = nodes[node_a_id]["pixels"] if node_a_id is not None else None
        node_b_pixels = nodes[node_b_id]["pixels"] if node_b_id is not None else None
        angle_a, angle_b = compute_branch_directions(
            ordered,
            node_a_pixels=node_a_pixels,
            node_b_pixels=node_b_pixels,
            num_pixels_for_direction=num_pixels_for_direction,
        )
        ys_w, xs_w = zip(*ordered)
        widths_arr = dist_map[list(ys_w), list(xs_w)] * 2.0
        branch_width = float(np.median(widths_arr))

        branches.append(
            {
                "id": branch_id,
                "node_a": node_a_id,
                "node_b": node_b_id,
                "pixels": ordered,
                "length": len(ordered),
                "direction_a": angle_a,
                "direction_b": angle_b,
                "width": branch_width,
            }
        )

    for cid in range(1, n_components + 1):
        component = comp_labels == cid

        touching: dict[int, set[tuple[int, int]]] = {}
        ys, xs = np.where(component)
        for y, x in zip(ys, xs):
            for dy, dx in NEIGHBOR_OFFSETS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nl = int(node_labels[ny, nx])
                    if nl > 0:
                        touching.setdefault(nl, set()).add((int(y), int(x)))

        touching_ids = sorted(touching.keys())

        if not touching_ids:
            y0, x0 = int(ys[0]), int(xs[0])
            n_nodes += 1
            synth_id = n_nodes
            node_labels[y0, x0] = synth_id
            nodes.append(
                {
                    "id": synth_id - 1,
                    "type": "loop_anchor",
                    "position": (float(y0), float(x0)),
                    "degree": 0,
                    "branches": [],
                    "pixels": [(y0, x0)],
                }
            )
            loop_pixels = [
                (int(y), int(x))
                for y, x in zip(*np.where(component))
                if not (int(y) == y0 and int(x) == x0)
            ]
            if loop_pixels:
                _register_branch(loop_pixels, synth_id - 1, synth_id - 1)
            continue

        for start_node_lbl in touching_ids:
            for start_px in touching[start_node_lbl]:
                ordered, end_node_lbl = trace_skeleton_segment(
                    component,
                    node_labels,
                    start_px,
                    start_node_lbl,
                )
                if not ordered:
                    continue

                node_a_id = start_node_lbl - 1
                node_b_id = (end_node_lbl - 1) if end_node_lbl is not None else None
                end_lbl = end_node_lbl if end_node_lbl is not None else start_node_lbl
                seg_key = (
                    min(start_node_lbl, end_lbl),
                    max(start_node_lbl, end_lbl),
                    frozenset(ordered),
                )
                if seg_key in seen_segments:
                    continue
                seen_segments.add(seg_key)
                _register_branch(ordered, node_a_id, node_b_id)

    # ── 8. Zero-length branches between directly adjacent nodes ─────────────
    seen_node_pairs: set[tuple[int, int]] = set()
    for br in branches:
        if br["node_a"] is not None and br["node_b"] is not None:
            seen_node_pairs.add(tuple(sorted((br["node_a"], br["node_b"]))))

    for nid_a in range(1, n_nodes + 1):
        for y, x in nodes[nid_a - 1]["pixels"]:
            for dy, dx in NEIGHBOR_OFFSETS:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                nid_b = int(node_labels[ny, nx])
                if nid_b <= nid_a:
                    continue
                pair = (nid_a - 1, nid_b - 1)
                if pair in seen_node_pairs:
                    continue
                seen_node_pairs.add(pair)
                branch_id = len(branches) + 1
                branches.append(
                    {
                        "id": branch_id,
                        "node_a": nid_a - 1,
                        "node_b": nid_b - 1,
                        "pixels": [],
                        "length": 0,
                        "direction_a": 0.0,
                        "direction_b": 0.0,
                        "width": 0.0,
                    }
                )

    # ── 9. Adjacency + degree bookkeeping ────────────────────────────────────
    adjacency: dict[int, list[int]] = {n["id"]: [] for n in nodes}
    for br in branches:
        if br["node_a"] is not None:
            adjacency[br["node_a"]].append(br["id"])
        if br["node_b"] is not None:
            adjacency[br["node_b"]].append(br["id"])

    for n in nodes:
        n["branches"] = adjacency[n["id"]]
        n["degree"] = len(n["branches"])

    # ── 10. Propagate branch IDs to the full vessel mask ─────────────────────
    segment_labels = assign_vessel_pixels(
        vessel_mask, branch_labels, dilation_radius=dilation_radius
    )

    return VesselGraph(
        skeleton=skeleton,
        branch_points=branch_pts,
        endpoints=endpoints,
        branch_labels=branch_labels,
        segment_labels=segment_labels,
        num_branches=len(branches),
        nodes=nodes,
        branches=branches,
        adjacency=adjacency,
    )
