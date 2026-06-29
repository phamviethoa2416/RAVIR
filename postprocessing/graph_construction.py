from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    binary_closing,
)
from skimage.measure import label as sk_label
from skimage.morphology import disk, remove_small_objects, skeletonize

from postprocessing.utils import (
    NEIGHBOR_OFFSETS,
    branch_point_mask,
    endpoint_mask,
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


def prune_skeleton(
    skeleton: np.ndarray,
    min_branch_length: int = 5,
    max_iterations: int = 10,
) -> np.ndarray:
    if min_branch_length <= 0:
        return skeleton.copy()

    skel = skeleton.astype(bool).copy()
    H, W = skel.shape

    for _ in range(max_iterations):
        endpoints = endpoint_mask(skel)
        if not endpoints.any():
            break

        branch_points = branch_point_mask(skel)
        to_remove = np.zeros_like(skel)

        for y, x in np.argwhere(endpoints):
            path = [(int(y), int(x))]
            prev_y, prev_x = -1, -1
            cy, cx = int(y), int(x)

            remove_path = False

            for _ in range(min_branch_length):
                next_pixel = None
                hit_branch_point = False

                for dy, dx in NEIGHBOR_OFFSETS:
                    ny, nx = cy + dy, cx + dx

                    if ny < 0 or nx < 0 or ny >= H or nx >= W:
                        continue
                    if not skel[ny, nx]:
                        continue
                    if ny == prev_y and nx == prev_x:
                        continue

                    if branch_points[ny, nx]:
                        hit_branch_point = True
                        break

                    if next_pixel is None:
                        next_pixel = (ny, nx)

                if hit_branch_point:
                    remove_path = True
                    break

                if next_pixel is None:
                    if len(path) < min_branch_length:
                        remove_path = True
                    break

                prev_y, prev_x = cy, cx
                cy, cx = next_pixel
                path.append((cy, cx))

            if remove_path:
                for py, px in path:
                    to_remove[py, px] = True

        if not to_remove.any():
            break

        skel &= ~to_remove

    return skel


def compute_branch_directions(
    branch_pixels: list[tuple[int, int]] | np.ndarray,
    node_a_yx: tuple[float, float] | None = None,
    node_b_yx: tuple[float, float] | None = None,
    n_pixels_for_direction: int = 5,
) -> tuple[float, float]:
    pixels = np.asarray(branch_pixels, dtype=np.int32)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape[0] == 0:
        return 0.0, 0.0

    k = int(max(2, min(n_pixels_for_direction, pixels.shape[0])))
    head = pixels[:k]
    tail = pixels[-k:]

    origin_a = (
        np.asarray(node_a_yx, dtype=np.float64)
        if node_a_yx is not None
        else head[0].astype(np.float64)
    )

    origin_b = (
        np.asarray(node_b_yx, dtype=np.float64)
        if node_b_yx is not None
        else tail[-1].astype(np.float64)
    )

    dy_a, dx_a = head[-1].astype(np.float64) - origin_a
    angle_a = float(np.arctan2(dy_a, dx_a))
    dy_b, dx_b = tail[0].astype(np.float64) - origin_b
    angle_b = float(np.arctan2(dy_b, dx_b))

    return angle_a, angle_b


def assign_vessel_pixels(
    vessel_mask: np.ndarray,
    branch_labels: np.ndarray,
    dilation_radius: int = 2,
) -> np.ndarray:
    vessel_mask = vessel_mask.astype(bool)
    segmentation = branch_labels.astype(np.int32).copy()
    segmentation[~vessel_mask] = 0

    if dilation_radius > 0:
        selem = disk(dilation_radius)
        dilated = np.zeros_like(segmentation)
        n_branches = int(segmentation.max())
        for branch_id in range(1, n_branches + 1):
            m = (
                binary_dilation(segmentation == branch_id, structure=selem)
                & vessel_mask
            )
            dilated[m & (dilated == 0)] = branch_id
        segmentation = dilated

    unlabeled = (segmentation == 0) & vessel_mask
    if unlabeled.any() and (segmentation > 0).any():
        _, nearest_idx = distance_transform_edt(
            segmentation == 0,
            return_distances=True,
            return_indices=True,
        )
        filled = segmentation[nearest_idx[0], nearest_idx[1]]
        segmentation = np.where(unlabeled, filled, segmentation)

    segmentation[~vessel_mask] = 0
    return segmentation.astype(np.int32)


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
            clean_mask, max_size=min_mask_object_size - 1, connectivity=2
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
                skeleton.astype(bool), max_size=min_component_size - 1, connectivity=2
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
        # Rebuild node labels after demotion.
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

    for cid in range(1, n_components + 1):
        component = comp_labels == cid

        touching: dict[int, set[tuple[int, int]]] = {}
        ys, xs = np.where(component)
        for y, x in zip(ys, xs):
            per_pixel_nodes: set[int] = set()
            for dy, dx in NEIGHBOR_OFFSETS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nl = int(node_labels[ny, nx])
                    if nl > 0:
                        per_pixel_nodes.add(nl)
            for nl in per_pixel_nodes:
                touching.setdefault(nl, set()).add((int(y), int(x)))

        touching_ids = sorted(touching.keys())
        if len(touching_ids) >= 2:
            node_a_id = touching_ids[0] - 1
            node_b_id = touching_ids[1] - 1
        elif len(touching_ids) == 1:
            only_node = touching_ids[0] - 1
            node_a_id = only_node
            node_b_id = only_node if len(touching[touching_ids[0]]) >= 2 else None
        else:
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
            component[y0, x0] = False
            node_a_id = synth_id - 1
            node_b_id = synth_id - 1

        preferred_start_node = touching_ids[0] if touching_ids else None
        pys, pxs = np.where(component)
        if len(pys) == 0:
            ordered: list[tuple[int, int]] = []
        elif len(pys) == 1:
            ordered = [(int(pys[0]), int(pxs[0]))]
        else:
            start: tuple[int, int] | None = None
            fallback: tuple[int, int] | None = None
            for py, px in zip(pys, pxs):
                touches: set[int] = set()
                for dy, dx in NEIGHBOR_OFFSETS:
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < H and 0 <= nx < W and node_labels[ny, nx] > 0:
                        touches.add(int(node_labels[ny, nx]))
                if preferred_start_node is not None and preferred_start_node in touches:
                    start = (int(py), int(px))
                    break
                if fallback is None and touches:
                    fallback = (int(py), int(px))
            if start is None:
                start = fallback if fallback is not None else (int(pys[0]), int(pxs[0]))

            ordered = [start]
            visited = np.zeros_like(component)
            visited[start] = True
            cy_walk, cx_walk = start
            while True:
                nxt: tuple[int, int] | None = None
                for dy, dx in NEIGHBOR_OFFSETS:
                    ny, nx = cy_walk + dy, cx_walk + dx
                    if (
                        0 <= ny < H
                        and 0 <= nx < W
                        and component[ny, nx]
                        and not visited[ny, nx]
                    ):
                        nxt = (ny, nx)
                        break
                if nxt is None:
                    break
                ordered.append(nxt)
                visited[nxt] = True
                cy_walk, cx_walk = nxt

        if not ordered:
            continue

        branch_id = len(branches) + 1
        branch_labels[component] = branch_id

        node_a_pos = nodes[node_a_id]["position"] if node_a_id is not None else None
        node_b_pos = nodes[node_b_id]["position"] if node_b_id is not None else None
        angle_a, angle_b = compute_branch_directions(
            ordered,
            node_a_yx=node_a_pos,
            node_b_yx=node_b_pos,
            n_pixels_for_direction=num_pixels_for_direction,
        )

        if ordered:
            ys_w, xs_w = zip(*ordered)
            widths_arr = dist_map[list(ys_w), list(xs_w)] * 2.0
            branch_width = float(np.median(widths_arr))
        else:
            branch_width = 0.0

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
