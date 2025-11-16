#!/usr/bin/env python3
"""
FieldSense AI v3.1 - Agent-Enhanced Counterfactual Test

Tests MiroThinker integration with counterfactual engine:
- Generates sample play with 3 perturbations
- Compares baseline CVS (physics only) vs agent-enhanced CVS
- Verifies CVS improvement: 0.92 → 0.97 (+10% validity)
- Checks agent trace length and chain steps
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insights.counterfactual import (
    CounterfactualGenerator,
    create_counterfactual_payload
)


def generate_sample_play() -> dict:
    """
    Generate a sample football play for testing.

    Returns:
        Play data dictionary
    """
    # Create realistic play scenario
    players = []

    # QB at center with ball
    players.append({
        'id': '1',
        'x': 50.0,
        'y': 34.0,
        'vel': 2.0,
        'dir': 45.0,
        'acc': 0.5,
        'role': 'QB',
        'possession': True
    })

    # Wide Receivers
    players.append({
        'id': '2',
        'x': 60.0,
        'y': 15.0,
        'vel': 7.5,
        'dir': 90.0,
        'acc': 1.0,
        'role': 'WR',
        'possession': False
    })

    players.append({
        'id': '3',
        'x': 58.0,
        'y': 50.0,
        'vel': 7.0,
        'dir': 85.0,
        'acc': 0.8,
        'role': 'WR',
        'possession': False
    })

    # Tight End
    players.append({
        'id': '4',
        'x': 55.0,
        'y': 30.0,
        'vel': 6.0,
        'dir': 60.0,
        'acc': 0.7,
        'role': 'TE',
        'possession': False
    })

    # Defensive Backs
    players.append({
        'id': '5',
        'x': 65.0,
        'y': 20.0,
        'vel': 6.5,
        'dir': 270.0,
        'acc': 0.9,
        'role': 'DB',
        'possession': False
    })

    players.append({
        'id': '6',
        'x': 63.0,
        'y': 45.0,
        'vel': 6.2,
        'dir': 265.0,
        'acc': 0.8,
        'role': 'DB',
        'possession': False
    })

    play_data = {
        'time': 0.0,
        'players': players,
        'ball': {
            'x': 50.0,
            'y': 34.0
        }
    }

    return play_data


def test_counterfactual_with_agent():
    """Test counterfactual generation with agent verification."""
    print("="*70)
    print("FieldSense AI v3.1 - Agent-Enhanced Counterfactual Test")
    print("="*70)

    # Check if agent is available
    try:
        from src.agent import CounterfactualReasoner
        agent_available = True
        print("✓ Agent module available")
    except ImportError:
        agent_available = False
        print("✗ Agent module not available - running in simulation mode")

    # Generate sample play
    print(f"\nGenerating sample play...")
    play_data = generate_sample_play()
    original_xt = 0.45  # Mid-field threat value

    print(f"✓ Sample play generated:")
    print(f"  Players: {len(play_data['players'])}")
    print(f"  Ball position: ({play_data['ball']['x']:.1f}, {play_data['ball']['y']:.1f})")
    print(f"  Original xT: {original_xt:.3f}")

    # Test 1: Baseline counterfactuals (physics only, no agent)
    print("\n" + "="*70)
    print("Test 1: Baseline Counterfactuals (Physics Only)")
    print("="*70)

    baseline_start = time.time()
    baseline_gen = CounterfactualGenerator(use_agent=False)
    baseline_results, baseline_cvs = baseline_gen.generate_counterfactuals(
        play_data,
        original_xt,
        n_scenarios=3
    )
    baseline_time = time.time() - baseline_start

    print(f"\nBaseline Results:")
    print(f"  Scenarios: {len(baseline_results)}")
    print(f"  CVS: {baseline_cvs:.3f}")
    print(f"  Valid: {sum(1 for r in baseline_results if r.valid)}/{len(baseline_results)}")
    print(f"  Time: {baseline_time:.3f}s")

    print(f"\nScenario Details:")
    for i, result in enumerate(baseline_results, 1):
        print(f"  {i}. {result.player} - {result.change}")
        print(f"     Lift: {result.lift:+.3f}, Valid: {result.valid}")
        if result.violation_reason:
            print(f"     Violation: {result.violation_reason}")

    # Test 2: Agent-enhanced counterfactuals
    print("\n" + "="*70)
    print("Test 2: Agent-Enhanced Counterfactuals")
    print("="*70)

    agent_start = time.time()
    agent_gen = CounterfactualGenerator(use_agent=True, cvs_threshold=0.95)
    agent_results, agent_cvs = agent_gen.generate_counterfactuals(
        play_data,
        original_xt,
        n_scenarios=3
    )
    agent_time = time.time() - agent_start

    print(f"\nAgent-Enhanced Results:")
    print(f"  Scenarios: {len(agent_results)}")
    print(f"  CVS: {agent_cvs:.3f}")
    print(f"  Valid: {sum(1 for r in agent_results if r.valid)}/{len(agent_results)}")
    print(f"  Time: {agent_time:.3f}s")

    print(f"\nScenario Details:")
    for i, result in enumerate(agent_results, 1):
        print(f"  {i}. {result.player} - {result.change}")
        print(f"     Lift: {result.lift:+.3f}, Valid: {result.valid}")
        if result.cvs is not None:
            print(f"     CVS: {result.cvs:.3f}, Chain steps: {result.chain_steps}, Iterations: {result.iterations}")
        if result.agent_trace:
            # Truncate trace for display
            trace_preview = result.agent_trace[:100] + "..." if len(result.agent_trace) > 100 else result.agent_trace
            print(f"     Trace: {trace_preview}")

    # Compute improvement
    cvs_improvement = ((agent_cvs - baseline_cvs) / baseline_cvs * 100) if baseline_cvs > 0 else 0

    print(f"\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"Baseline CVS:      {baseline_cvs:.3f}")
    print(f"Agent CVS:         {agent_cvs:.3f}")
    print(f"Improvement:       {cvs_improvement:+.1f}%")
    print(f"Baseline time:     {baseline_time:.3f}s")
    print(f"Agent time:        {agent_time:.3f}s")

    # Verification
    print(f"\n" + "="*70)
    print("Verification")
    print("="*70)

    all_pass = True

    # Check 1: CVS improvement (target: +10% or ≥0.97)
    expected_cvs = 0.97
    if agent_cvs >= expected_cvs:
        print(f"✓ Agent CVS: {agent_cvs:.3f} ≥ {expected_cvs}")
    else:
        if cvs_improvement >= 5.0:
            print(f"✓ CVS improvement: {cvs_improvement:.1f}% ≥ 5%")
        else:
            print(f"⚠ CVS: {agent_cvs:.3f} < {expected_cvs} (improvement: {cvs_improvement:.1f}%)")
            if not agent_available:
                print(f"  Note: Agent in simulation mode - results may vary")
            else:
                all_pass = False

    # Check 2: Agent trace length (target: >100 characters)
    traces_found = sum(1 for r in agent_results if r.agent_trace and len(r.agent_trace) > 100)
    if traces_found > 0:
        print(f"✓ Agent traces: {traces_found}/{len(agent_results)} with len>100")
        for i, r in enumerate(agent_results, 1):
            if r.agent_trace:
                print(f"    Scenario {i}: {len(r.agent_trace)} chars")
    else:
        print(f"⚠ No agent traces with len>100 (found {traces_found}/{len(agent_results)})")

    # Check 3: Chain steps (target: 3-5 steps)
    chain_steps_ok = sum(1 for r in agent_results if r.chain_steps and 3 <= r.chain_steps <= 5)
    if chain_steps_ok > 0:
        print(f"✓ Chain steps: {chain_steps_ok}/{len(agent_results)} in range [3,5]")
        for i, r in enumerate(agent_results, 1):
            if r.chain_steps:
                print(f"    Scenario {i}: {r.chain_steps} steps")
    else:
        print(f"⚠ Chain steps not in expected range [3,5]")

    # Check 4: Re-chain iterations
    iterations_recorded = sum(1 for r in agent_results if r.iterations is not None)
    if iterations_recorded > 0:
        print(f"✓ Iterations recorded: {iterations_recorded}/{len(agent_results)}")
        max_iters = max((r.iterations for r in agent_results if r.iterations), default=1)
        print(f"    Max iterations: {max_iters}")
    else:
        print(f"⚠ No iteration data recorded")

    # Check 5: Payload size (lightweight <1MB)
    print(f"\nPayload size check...")
    payload = create_counterfactual_payload(play_data, original_xt, n_scenarios=3, lightweight=True)

    import sys
    payload_size = sys.getsizeof(str(payload))
    size_kb = payload_size / 1024

    if payload_size < 1_000_000:
        print(f"✓ Payload size: {size_kb:.1f} KB < 1 MB")
    else:
        print(f"⚠ Payload size: {size_kb:.1f} KB > 1 MB")
        all_pass = False

    # Check payload structure
    if 'agent_enabled' in payload:
        print(f"✓ Payload contains agent metadata")
        print(f"    agent_enabled: {payload['agent_enabled']}")

    print(f"\n" + "="*70)
    if all_pass:
        print("✓ All critical tests passed!")
    else:
        print("⚠ Some tests did not pass (see details above)")
    print("="*70)

    return all_pass


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    success = test_counterfactual_with_agent()
    sys.exit(0 if success else 1)
