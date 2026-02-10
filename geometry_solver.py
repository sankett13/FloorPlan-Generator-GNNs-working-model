"""
Constrained geometric layout solver for floorplan rooms.

Exposes:
  - solve_layout(...)
  - OverConstrainedLayoutError

This is a penalty-based optimization using scipy.optimize.minimize.
"""
from typing import Dict, Tuple, List
import numpy as np
import networkx as nx
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from scipy.optimize import minimize


class OverConstrainedLayoutError(Exception):
    """Raised when the layout solver cannot find a feasible solution."""
    pass


def _make_polygons_from_vars(vars_vec: np.ndarray, ids: List[str]):
    """Convert variable vector to dict of shapely Polygons.

    vars layout: [x0,y0,w0,h0, x1,y1,w1,h1, ...]
    ids: list of room identifiers in same order.
    """
    polys = {}
    for i, rid in enumerate(ids):
        x = vars_vec[4 * i + 0]
        y = vars_vec[4 * i + 1]
        w = max(0.01, vars_vec[4 * i + 2])
        h = max(0.01, vars_vec[4 * i + 3])
        polys[rid] = box(x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)
    return polys


def _total_overlap_area(polys: Dict[str, Polygon]):
    """Compute sum of pairwise intersection areas."""
    keys = list(polys.keys())
    total = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = polys[keys[i]].intersection(polys[keys[j]])
            if not inter.is_empty:
                total += inter.area
    return float(total)


def _boundary_violation_area(polys: Dict[str, Polygon], boundary_w: float, boundary_h: float):
    """Area of polygons outside the boundary box (0,0)-(w,h)."""
    boundary = box(0.0, 0.0, boundary_w, boundary_h)
    total = 0.0
    for p in polys.values():
        outside = p.difference(boundary)
        if not outside.is_empty:
            total += outside.area
    return float(total)


def _adjacency_violation(polys: Dict[str, Polygon], adjacency_graph: nx.Graph):
    """Measure adjacency violation: sum of squared distances between adjacent rooms' exteriors.

    If two rooms intersect or share edge, distance = 0.
    Otherwise use shapely.distance between exteriors.
    """
    total = 0.0
    for u, v in adjacency_graph.edges():
        if u not in polys or v not in polys:
            continue
        d = polys[u].distance(polys[v])
        total += d * d
    return float(total)


def solve_layout(
    centroids: Dict[str, np.ndarray],
    target_sizes: Dict[str, Tuple[float, float]],
    adjacency_graph: nx.Graph,
    boundary: Tuple[float, float],
    max_iter: int = 500,
):
    """Solve constrained layout via penalty-based continuous optimization.

    Parameters
    - centroids: mapping id -> (x,y) or list/np.array of centroids
    - target_sizes: mapping id -> (w_target, h_target)
    - adjacency_graph: networkx.Graph with node ids matching centroids keys
    - boundary: (width, height)

    Returns mapping id -> shapely.geometry.Polygon on success.
    Raises OverConstrainedLayoutError if optimizer fails to converge.
    """
    # Convert input centroids to ordered lists
    ids = list(centroids.keys())
    n = len(ids)
    boundary_w, boundary_h = boundary

    # Initial variable vector
    x0 = np.zeros(4 * n, dtype=float)
    for i, rid in enumerate(ids):
        c = np.array(centroids[rid])
        w_t, h_t = target_sizes.get(rid, (1.0, 1.0))
        x0[4 * i + 0] = float(c[0])
        x0[4 * i + 1] = float(c[1])
        x0[4 * i + 2] = float(w_t)
        x0[4 * i + 3] = float(h_t)

    # Bounds: keep centers inside boundary and sizes positive and reasonably bounded
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.0, boundary_w))
        # y bounds
        bounds.append((0.0, boundary_h))
        # width bounds (0.2m .. boundary_w)
        w_t = max(0.2, x0[4 * i + 2])
        bounds.append((0.2, min(boundary_w, max(w_t * 3.0, 0.5))))
        # height bounds
        h_t = max(0.2, x0[4 * i + 3])
        bounds.append((0.2, min(boundary_h, max(h_t * 3.0, 0.5))))

    # Objective weights (defaults as requested)
    w_overlap = 1000.0
    w_size = 1.0
    w_adj = 50.0
    w_boundary = 500.0

    # Precompute target arrays
    w_targets = np.zeros(n)
    h_targets = np.zeros(n)
    for i, rid in enumerate(ids):
        wt, ht = target_sizes.get(rid, (x0[4 * i + 2], x0[4 * i + 3]))
        w_targets[i] = float(wt)
        h_targets[i] = float(ht)

    def objective(vars_vec):
        polys = _make_polygons_from_vars(vars_vec, ids)

        overlap_area = _total_overlap_area(polys)
        # size deviation
        sizes = np.array([vars_vec[4 * i + 2:4 * i + 4] for i in range(n)])
        size_dev = np.sum((sizes[:, 0] - w_targets) ** 2 + (sizes[:, 1] - h_targets) ** 2)
        adj_violation = _adjacency_violation(polys, adjacency_graph)
        boundary_violation = _boundary_violation_area(polys, boundary_w, boundary_h)

        total = (
            w_overlap * overlap_area
            + w_size * size_dev
            + w_adj * adj_violation
            + w_boundary * boundary_violation
        )
        return total

    # Run optimizer
    res = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-6}
    )

    if not res.success:
        # Optimization did not converge
        raise OverConstrainedLayoutError(f"Layout solver failed: {res.message}")

    final_polys = _make_polygons_from_vars(res.x, ids)

    # Final safety: return polygons; caller may check overlap and fallback if needed
    return final_polys
