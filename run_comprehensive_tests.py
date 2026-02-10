"""Comprehensive test suite comparing geometry solver vs guillotine packing.

This produces detailed reports at:
- Outputs/layout_comparison.log (detailed metrics)
- Outputs/test_summary.txt (formatted summary)
"""
import sys
import traceback
from pathlib import Path
import numpy as np
import networkx as nx
from shapely.geometry import box
from datetime import datetime

# Add tests to path
sys.path.insert(0, str(Path(__file__).parent / 'tests'))

from test_layout_comparison import (
    test_solver_vs_guillotine_and_log,
    _make_polys_from_positions,
    _compute_loss,
)

from geometry_solver import solve_layout, OverConstrainedLayoutError
from gatnet_integrated_generator import GATNetFloorPlanGenerator


def run_scenario_1():
    """Scenario 1: Tight 5-room layout (original test)."""
    np.random.seed(42)
    
    boundary = (12.0, 10.0)
    base_centroids = {
        'living_room#0': np.array([6.0, 2.5]),
        'bedroom#0': np.array([4.5, 6.5]),
        'bedroom#1': np.array([7.5, 6.5]),
        'bathroom#0': np.array([9.0, 4.5]),
        'kitchen#0': np.array([8.5, 7.5]),
    }
    target_sizes = {
        'living_room#0': (4.0, 3.5),
        'bedroom#0': (3.0, 3.0),
        'bedroom#1': (3.0, 3.0),
        'bathroom#0': (2.0, 2.2),
        'kitchen#0': (3.5, 2.8),
    }
    
    G_adj = nx.Graph()
    for nid in base_centroids.keys():
        G_adj.add_node(nid)
    G_adj.add_edge('living_room#0', 'bedroom#0')
    G_adj.add_edge('living_room#0', 'bedroom#1')
    G_adj.add_edge('kitchen#0', 'bathroom#0')
    
    return "tight_5room", boundary, base_centroids, target_sizes, G_adj


def run_scenario_2():
    """Scenario 2: Larger 8-room apartment with complex adjacencies."""
    np.random.seed(123)
    
    boundary = (15.0, 12.0)
    base_centroids = {
        'living_room#0': np.array([7.5, 3.0]),
        'bedroom#0': np.array([3.0, 8.0]),
        'bedroom#1': np.array([7.5, 8.0]),
        'bedroom#2': np.array([12.0, 8.0]),
        'bathroom#0': np.array([6.0, 5.0]),
        'bathroom#1': np.array([12.0, 5.0]),
        'kitchen#0': np.array([1.5, 5.0]),
        'dining#0': np.array([4.0, 2.0]),
    }
    target_sizes = {
        'living_room#0': (5.0, 4.5),
        'bedroom#0': (3.5, 3.5),
        'bedroom#1': (3.5, 3.5),
        'bedroom#2': (3.5, 3.5),
        'bathroom#0': (2.0, 2.2),
        'bathroom#1': (2.0, 2.2),
        'kitchen#0': (3.0, 3.0),
        'dining#0': (4.0, 3.5),
    }
    
    G_adj = nx.Graph()
    for nid in base_centroids.keys():
        G_adj.add_node(nid)
    # Connect adjacent rooms
    G_adj.add_edge('living_room#0', 'dining#0')
    G_adj.add_edge('living_room#0', 'bedroom#1')
    G_adj.add_edge('living_room#0', 'bathroom#0')
    G_adj.add_edge('kitchen#0', 'dining#0')
    G_adj.add_edge('kitchen#0', 'bathroom#0')
    G_adj.add_edge('bedroom#0', 'bathroom#0')
    G_adj.add_edge('bedroom#1', 'bathroom#0')
    G_adj.add_edge('bedroom#2', 'bathroom#1')
    
    return "complex_8room", boundary, base_centroids, target_sizes, G_adj


def run_scenario_3():
    """Scenario 3: Small studio-like layout."""
    np.random.seed(456)
    
    boundary = (8.0, 7.0)
    base_centroids = {
        'living_room#0': np.array([3.5, 2.0]),
        'bedroom#0': np.array([2.0, 5.0]),
        'bathroom#0': np.array([6.0, 5.0]),
        'kitchen#0': np.array([6.0, 2.0]),
    }
    target_sizes = {
        'living_room#0': (3.5, 2.5),
        'bedroom#0': (2.5, 2.5),
        'bathroom#0': (1.8, 2.0),
        'kitchen#0': (2.5, 2.0),
    }
    
    G_adj = nx.Graph()
    for nid in base_centroids.keys():
        G_adj.add_node(nid)
    G_adj.add_edge('living_room#0', 'bedroom#0')
    G_adj.add_edge('living_room#0', 'kitchen#0')
    G_adj.add_edge('kitchen#0', 'bathroom#0')
    
    return "studio_4room", boundary, base_centroids, target_sizes, G_adj


def test_scenario(scenario_name, boundary, base_centroids, target_sizes, G_adj):
    """Test a single scenario and return metrics."""
    try:
        # Run solver
        polys_solver = solve_layout(base_centroids, target_sizes, G_adj, boundary, max_iter=300)
    except OverConstrainedLayoutError as e:
        return {"error": str(e), "scenario": scenario_name}
    
    # Run guillotine
    gen = GATNetFloorPlanGenerator()
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
    
    rooms_to_place.sort(key=lambda x: (-x['priority'], -x['area']))
    packed_positions = gen._guillotine_pack(rooms_to_place, boundary[0], boundary[1])
    
    guill_positions = {}
    for room_info, pos in zip(rooms_to_place, packed_positions):
        rid = f"{room_info['type']}#{room_info['index']}"
        guill_positions[rid] = (pos['x'], pos['y'])
    
    polys_guill = _make_polys_from_positions(guill_positions, target_sizes)
    
    # Compute metrics
    metrics_solver = _compute_loss(polys_solver, target_sizes, G_adj, boundary)
    metrics_guill = _compute_loss(polys_guill, target_sizes, G_adj, boundary)
    
    return {
        "scenario": scenario_name,
        "boundary": boundary,
        "num_rooms": len(target_sizes),
        "solver": metrics_solver,
        "guillotine": metrics_guill,
    }


def main():
    """Run all test scenarios and generate reports."""
    output_dir = Path("Outputs")
    output_dir.mkdir(exist_ok=True)
    
    log_path = output_dir / "layout_comparison.log"
    summary_path = output_dir / "test_summary.txt"
    
    # Scenarios to test
    scenarios = [
        run_scenario_1(),
        run_scenario_2(),
        run_scenario_3(),
    ]
    
    all_results = []
    
    print("Running comprehensive layout tests...")
    print("=" * 60)
    
    for scenario_data in scenarios:
        scenario_name, boundary, base_centroids, target_sizes, G_adj = scenario_data
        print(f"Testing: {scenario_name}...", end=" ")
        
        result = test_scenario(scenario_name, boundary, base_centroids, target_sizes, G_adj)
        all_results.append(result)
        
        if "error" in result:
            print(f"❌ FAILED: {result['error']}")
        else:
            print("✅ PASSED")
    
    # Write detailed log
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DETAILED LAYOUT COMPARISON TEST RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        for result in all_results:
            if "error" in result:
                f.write(f"\n{result['scenario']}: FAILED\n")
                f.write(f"  Error: {result['error']}\n")
                continue
            
            f.write(f"\n{'─' * 70}\n")
            f.write(f"Scenario: {result['scenario']}\n")
            f.write(f"Boundary: {result['boundary'][0]:.1f}m × {result['boundary'][1]:.1f}m\n")
            f.write(f"Rooms: {result['num_rooms']}\n")
            f.write(f"{'─' * 70}\n\n")
            
            f.write("SOLVER (New Constrained Optimizer):\n")
            for k, v in result['solver'].items():
                f.write(f"  {k:20s}: {v:12.6f}\n")
            
            f.write("\nGUILLOTINE PACKING (Baseline):\n")
            for k, v in result['guillotine'].items():
                f.write(f"  {k:20s}: {v:12.6f}\n")
            
            # Compare
            improvement = (result['guillotine']['total'] - result['solver']['total']) / max(result['guillotine']['total'], 1.0) * 100
            f.write(f"\nImprovement: {improvement:+.1f}% (solver is {'better' if improvement > 0 else 'worse'})\n")
    
    # Write summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("TEST SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("Test Results Overview:\n")
        f.write("-" * 70 + "\n")
        
        passed = sum(1 for r in all_results if "error" not in r)
        failed = sum(1 for r in all_results if "error" in r)
        
        f.write(f"Total Scenarios: {len(all_results)}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {failed}\n\n")
        
        # Detailed results
        f.write("Individual Scenario Results:\n")
        f.write("-" * 70 + "\n\n")
        
        for result in all_results:
            if "error" in result:
                f.write(f"❌ {result['scenario']}: FAILED\n")
                f.write(f"   Error: {result['error']}\n\n")
            else:
                improvement = (result['guillotine']['total'] - result['solver']['total']) / max(result['guillotine']['total'], 1.0) * 100
                verdict = "✅ IMPROVED" if improvement > 0 else "⚠️  WORSE" if improvement < -1 else "≈ EQUAL"
                
                f.write(f"{verdict} {result['scenario']}\n")
                f.write(f"   {result['num_rooms']} rooms, {result['boundary'][0]:.0f}×{result['boundary'][1]:.0f}m\n")
                f.write(f"   Improvement: {improvement:+.1f}%\n")
                f.write(f"   Solver Loss:      {result['solver']['total']:10.2f}\n")
                f.write(f"   Guillotine Loss:  {result['guillotine']['total']:10.2f}\n\n")
        
        # Summary stats
        f.write("-" * 70 + "\n")
        f.write("Summary Statistics:\n")
        f.write("-" * 70 + "\n\n")
        
        total_improvement = 0
        count = 0
        for result in all_results:
            if "error" not in result:
                improvement = (result['guillotine']['total'] - result['solver']['total']) / max(result['guillotine']['total'], 1.0) * 100
                total_improvement += improvement
                count += 1
        
        avg_improvement = total_improvement / count if count > 0 else 0
        f.write(f"Average Improvement across all scenarios: {avg_improvement:+.1f}%\n\n")
        
        if avg_improvement > 0:
            f.write("✅ VERDICT: Solver is BETTER than guillotine packing on average!\n")
        elif avg_improvement > -1:
            f.write("≈ VERDICT: Solver is roughly EQUAL to guillotine packing.\n")
        else:
            f.write("⚠️  VERDICT: Solver is WORSE than guillotine packing on average.\n")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    with open(summary_path, 'r') as f:
        print(f.read())
    
    print(f"\n✅ Detailed log written to: {log_path}")
    print(f"✅ Summary written to: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)
