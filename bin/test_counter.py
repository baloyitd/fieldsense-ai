#!/usr/bin/env python3
"""
Test script for FieldSense AI v3.0 Counterfactual System
Validates physics-constrained simulations and CVS >= 0.92
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import load_data
from src.data.normalize import normalize_to_json
from src.insights.counterfactual import (
    CounterfactualGenerator,
    PhysicsSimulator,
    create_counterfactual_payload
)


def create_test_play() -> dict:
    """Create a test play with realistic player positions."""
    return {
        'time': 0.0,
        'players': [
            {'id': '1', 'x': 50, 'y': 25, 'role': 'QB', 'possession': True,
             'vel': 5.0, 'dir': 45, 'acc': 1.0},
            {'id': '2', 'x': 65, 'y': 15, 'role': 'WR', 'possession': False,
             'vel': 8.5, 'dir': 90, 'acc': 1.5},
            {'id': '3', 'x': 70, 'y': 35, 'role': 'WR', 'possession': False,
             'vel': 7.8, 'dir': 60, 'acc': 1.2},
            {'id': '4', 'x': 45, 'y': 30, 'role': 'RB', 'possession': False,
             'vel': 6.5, 'dir': 75, 'acc': 0.8},
            {'id': '5', 'x': 35, 'y': 28, 'role': 'LB', 'possession': False,
             'vel': 5.5, 'dir': 180, 'acc': 0.5},
            {'id': '6', 'x': 55, 'y': 45, 'role': 'CB', 'possession': False,
             'vel': 6.0, 'dir': 270, 'acc': 0.7},
        ],
        'ball': {'x': 50, 'y': 25}
    }


def test_counterfactual_generation():
    """Test counterfactual generation with multiple scenarios."""
    print("="*60)
    print("FieldSense AI v3.0 - Counterfactual Test")
    print("="*60)

    # Create test play
    print("\n[Step 1/5] Creating test play...")
    play_data = create_test_play()
    print(f"  Play created with {len(play_data['players'])} players")

    # Initialize generator
    print("\n[Step 2/5] Initializing counterfactual generator...")
    generator = CounterfactualGenerator()
    simulator = generator.simulator
    print(f"  Time step: {simulator.dt}s")
    print(f"  Physics enabled: momentum + decay + boundaries")

    # Generate counterfactuals
    print("\n[Step 3/5] Generating counterfactuals...")
    n_scenarios = 3
    original_xt = 0.5

    results, cvs = generator.generate_counterfactuals(
        play_data,
        original_xt=original_xt,
        n_scenarios=n_scenarios
    )

    print(f"  Generated {len(results)} scenarios")
    print(f"  CVS: {cvs:.3f}")

    # Display results
    print("\n[Step 4/5] Counterfactual Results:")
    for i, result in enumerate(results, 1):
        status = "✓" if result.valid else "✗"
        print(f"\n  [{i}] {status} {result.player} - {result.change}")
        print(f"      xT Lift: {result.lift:+.3f} ({result.lift*100:+.1f}%)")
        print(f"      Valid: {result.valid}")
        if not result.valid:
            print(f"      Violation: {result.violation_reason}")
        if result.new_positions:
            print(f"      Trajectory points: {len(result.new_positions)}")

    # Validate CVS
    print("\n[Step 5/5] Validation:")

    # Test 1: CVS >= 0.92
    cvs_threshold = 0.92
    cvs_passed = cvs >= cvs_threshold

    print(f"  CVS Test: {cvs:.3f} >= {cvs_threshold}")
    if cvs_passed:
        print(f"    ✓ PASSED: CVS meets threshold")
    else:
        print(f"    ✗ FAILED: CVS below threshold")

    # Test 2: All scenarios have trajectories
    trajectories_ok = all(r.new_positions for r in results if r.valid)
    print(f"\n  Trajectory Test:")
    if trajectories_ok:
        print(f"    ✓ PASSED: All valid scenarios have trajectories")
    else:
        print(f"    ✗ FAILED: Some valid scenarios missing trajectories")

    # Test 3: Physics constraints
    print(f"\n  Physics Constraints:")
    n_valid = sum(1 for r in results if r.valid)
    valid_rate = n_valid / len(results) if results else 0

    print(f"    Valid scenarios: {n_valid}/{len(results)} ({valid_rate*100:.1f}%)")

    # Check for common violations
    violations = {}
    for result in results:
        if not result.valid and result.violation_reason:
            vtype = result.violation_reason.split(':')[0]
            violations[vtype] = violations.get(vtype, 0) + 1

    if violations:
        print(f"    Violations detected:")
        for vtype, count in violations.items():
            print(f"      - {vtype}: {count}")
    else:
        print(f"    ✓ No violations detected")

    # Test 4: xT changes are reasonable
    print(f"\n  xT Lift Analysis:")
    lifts = [r.lift for r in results if r.valid]
    if lifts:
        avg_lift = np.mean(lifts)
        max_lift = max(lifts)
        min_lift = min(lifts)

        print(f"    Average: {avg_lift:+.3f}")
        print(f"    Range: [{min_lift:+.3f}, {max_lift:+.3f}]")

        # Lifts should be reasonable (not too extreme)
        reasonable = all(-0.5 < lift < 0.5 for lift in lifts)
        if reasonable:
            print(f"    ✓ All lifts within reasonable range")
        else:
            print(f"    ⚠ Some lifts may be unrealistic")

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    tests_passed = []
    tests_failed = []

    if cvs_passed:
        tests_passed.append("CVS >= 0.92")
    else:
        tests_failed.append(f"CVS = {cvs:.3f} < 0.92")

    if trajectories_ok:
        tests_passed.append("All valid scenarios have trajectories")
    else:
        tests_failed.append("Missing trajectories")

    if valid_rate >= 0.5:
        tests_passed.append(f"Valid rate: {valid_rate*100:.1f}%")
    else:
        tests_failed.append(f"Low valid rate: {valid_rate*100:.1f}%")

    print(f"\nPassed: {len(tests_passed)}/{len(tests_passed) + len(tests_failed)}")
    for test in tests_passed:
        print(f"  ✓ {test}")

    if tests_failed:
        print(f"\nFailed:")
        for test in tests_failed:
            print(f"  ✗ {test}")

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / "counterfactual_results.json"
    payload = create_counterfactual_payload(play_data, original_xt, n_scenarios)

    with open(results_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return 0 if cvs_passed else 1


def test_physics_simulation():
    """Test physics simulation components."""
    print("\n" + "="*60)
    print("Physics Simulation Test")
    print("="*60)

    simulator = PhysicsSimulator(dt=0.1)

    # Test 1: Player path simulation
    print("\n[Test 1] Player path simulation:")
    start_pos = np.array([50.0, 30.0])
    path = simulator.simulate_player_path(
        start_pos=start_pos,
        velocity=8.0,
        direction=45.0,
        duration=2.0
    )

    print(f"  Generated {len(path)} positions over 2.0s")
    print(f"  Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
    print(f"  End: ({path[-1][0]:.1f}, {path[-1][1]:.1f})")

    # Check positions are in bounds
    in_bounds = all(
        0 <= pos[0] <= 105 and 0 <= pos[1] <= 68
        for pos in path
    )

    if in_bounds:
        print(f"  ✓ All positions within field bounds")
    else:
        print(f"  ✗ Some positions out of bounds")

    # Test 2: Ball trajectory simulation
    print("\n[Test 2] Ball trajectory simulation:")
    ball_start = np.array([50.0, 30.0])
    ball_vel = np.array([10.0, 5.0])

    ball_path = simulator.simulate_ball_trajectory(
        start_pos=ball_start,
        initial_velocity=ball_vel,
        duration=1.0
    )

    print(f"  Generated {len(ball_path)} positions over 1.0s")
    print(f"  Start: ({ball_start[0]:.1f}, {ball_start[1]:.1f})")
    print(f"  End: ({ball_path[-1][0]:.1f}, {ball_path[-1][1]:.1f})")

    # Check velocity decay
    initial_speed = np.linalg.norm(ball_vel)
    # Approximate final velocity from last two positions
    if len(ball_path) >= 2:
        final_displacement = ball_path[-1] - ball_path[-2]
        final_speed = np.linalg.norm(final_displacement) / simulator.dt

        print(f"  Initial speed: {initial_speed:.2f} m/s")
        print(f"  Final speed: {final_speed:.2f} m/s")

        if final_speed < initial_speed:
            print(f"  ✓ Velocity decay working correctly")
        else:
            print(f"  ⚠ No velocity decay detected")

    # Test 3: Field boundaries
    print("\n[Test 3] Field boundary enforcement:")
    boundary_test_start = np.array([100.0, 60.0])
    boundary_path = simulator.simulate_player_path(
        start_pos=boundary_test_start,
        velocity=8.0,
        direction=45.0,  # Will hit boundary
        duration=1.0
    )

    all_in_bounds = all(
        0 <= pos[0] <= 105 and 0 <= pos[1] <= 68
        for pos in boundary_path
    )

    if all_in_bounds:
        print(f"  ✓ Boundaries enforced correctly")
    else:
        print(f"  ✗ Boundary violation detected")

    print(f"\nPhysics simulation tests complete")


if __name__ == '__main__':
    try:
        # Run counterfactual generation test
        result = test_counterfactual_generation()

        # Run physics simulation test
        test_physics_simulation()

        sys.exit(result)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
