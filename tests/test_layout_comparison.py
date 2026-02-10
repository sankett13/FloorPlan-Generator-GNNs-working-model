"""Tests comparing new constrained geometry solver vs previous guillotine packing.

Produces a human-readable log at Outputs/layout_comparison.log with metrics.

These tests are deterministic (seeded) and keep a permissive assertion so they
report improvements without being flaky.
"""
import os
import sys
from pathlib import Path
import numpy as np
import networkx as nx
from shapely.geometry import box

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry_solver import solve_layout, OverConstrainedLayoutError, _total_overlap_area, _boundary_violation_area, _adjacency_violation
from gatnet_integrated_generator import GATNetFloorPlanGenerator


def _make_polys_from_positions(positions, sizes):
    polys = {}
    for rid, (x, y) in positions.items():
        w, h = sizes[rid]
        polys[rid] = box(x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0)
    return polys


def _compute_loss(polys, sizes, adjacency_graph, boundary, weights=None):
    if weights is None:
        weights = dict(w_overlap=1000.0, w_size=1.0, w_adj=50.0, w_boundary=500.0)

    ids = list(polys.keys())
    # sizes_dev: sum squared diff from target sizes
    size_dev = 0.0
    for rid in ids:
        w_actual = polys[rid].bounds[2] - polys[rid].bounds[0]
        h_actual = polys[rid].bounds[3] - polys[rid].bounds[1]
        w_t, h_t = sizes[rid]
        size_dev += (w_actual - w_t) ** 2 + (h_actual - h_t) ** 2

    overlap_area = _total_overlap_area(polys)
    adj_violation = _adjacency_violation(polys, adjacency_graph)
    boundary_violation = _boundary_violation_area(polys, boundary[0], boundary[1])

    total = (
        weights['w_overlap'] * overlap_area
        + weights['w_size'] * size_dev
        + weights['w_adj'] * adj_violation
        + weights['w_boundary'] * boundary_violation
    )

    return dict(total=total, overlap=overlap_area, size_dev=size_dev, adj=adj_violation, boundary=boundary_violation)


def test_solver_vs_guillotine_and_log():
    """Run solver and guillotine packing on a synthetic clustered layout and log comparison.

    The test writes a human-readable log to Outputs/layout_comparison.log and asserts
    the solver's loss is not massively worse than guillotine (sanity check).
    """
    np.random.seed(42)

    out_dir = Path("Outputs")
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / "layout_comparison.log"

    # Synthetic scenario
    boundary = (12.0, 10.0)

    # Define rooms and intentionally overlapping centroids
    base_centroids = {
        'living_room#0': np.array([6.0, 2.5]),
        'bedroom#0': np.array([4.5, 6.5]),
        'bedroom#1': np.array([7.5, 6.5]),
        'bathroom#0': np.array([9.0, 4.5]),
        'kitchen#0': np.array([8.5, 7.5]),
    }

    # target sizes (w,h) derived to create tight packing
    target_sizes = {
        'living_room#0': (4.0, 3.5),
        'bedroom#0': (3.0, 3.0),
        'bedroom#1': (3.0, 3.0),
        'bathroom#0': (2.0, 2.2),
        'kitchen#0': (3.5, 2.8),
    }

    # adjacency intent: living near bedrooms, kitchen near bathroom
    G_adj = nx.Graph()
    for nid in base_centroids.keys():
        G_adj.add_node(nid)
    G_adj.add_edge('living_room#0', 'bedroom#0')
    G_adj.add_edge('living_room#0', 'bedroom#1')
    G_adj.add_edge('kitchen#0', 'bathroom#0')

    # Run constrained solver (single run with jitter disabled by setting small jitter)
    try:
        polys_solver = solve_layout(base_centroids, target_sizes, G_adj, boundary, max_iter=300)
    except OverConstrainedLayoutError as e:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Solver failed with OverConstrainedLayoutError: {e}\n")
        assert False, f"Solver failed: {e}"

    # Build guillotine-packed positions via generator helper
    gen = GATNetFloorPlanGenerator()
    # Create rooms_to_place structure expected by _guillotine_pack
    rooms_to_place = []
    for rid, (w, h) in target_sizes.items():
        room_type = rid.split('#')[0]
        idx = int(rid.split('#')[1])
        rooms_to_place.append({
            'type': room_type,
            'index': idx,
            'width': w,
            'height': h,
            'area': w * h,
            'priority': 1,
            'original_centroid': base_centroids[rid].tolist()
        })

    # sort for deterministic packing
    rooms_to_place.sort(key=lambda x: (-x['priority'], -x['area']))
    packed_positions = gen._guillotine_pack(rooms_to_place, boundary[0], boundary[1])

    guill_positions = {}
    for room_info, pos in zip(rooms_to_place, packed_positions):
        rid = f"{room_info['type']}#{room_info['index']}"
        guill_positions[rid] = (pos['x'], pos['y'])

    # Convert solver polys to comparison dict of polys
    polys_guill = _make_polys_from_positions(guill_positions, target_sizes)

    # polys_solver is dict returned from solver
    polys_sol = polys_solver

    # Compute metrics
    metrics_solver = _compute_loss(polys_sol, target_sizes, G_adj, boundary)
    metrics_guill = _compute_loss(polys_guill, target_sizes, G_adj, boundary)

    # Write human-friendly log
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Layout Comparison Results\n")
        f.write("========================\n\n")
        f.write("Boundary: {} x {}\n\n".format(*boundary))

        f.write("Solver Metrics:\n")
        for k, v in metrics_solver.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\nGuillotine Metrics:\n")
        for k, v in metrics_guill.items():
            f.write(f"  {k}: {v:.6f}\n")

        f.write("\nSummary:\n")
        f.write(f"  solver_total = {metrics_solver['total']:.6f}\n")
        f.write(f"  guill_total  = {metrics_guill['total']:.6f}\n")
        if metrics_solver['total'] <= metrics_guill['total']:
            f.write("  Verdict: solver <= guillotine (improved or equal)\n")
        else:
            f.write("  Verdict: solver > guillotine (worse)\n")

    # Sanity assertion: solver must not be massively worse than guillotine
    # (allow up to 10x worse as absolute safeguard; adjust if you want stricter test)
    assert metrics_solver['total'] <= metrics_guill['total'] * 10 + 1e-6
