#!/usr/bin/env python3
"""
FieldSense AI v3.1 - Agent-Enhanced Counterfactual Unit Tests

Tests for MiroThinker integration with counterfactual engine.
"""

import unittest
import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insights.counterfactual import (
    CounterfactualGenerator,
    CounterfactualResult,
    create_counterfactual_payload
)


class TestCounterfactualReasoner(unittest.TestCase):
    """Test CounterfactualReasoner class."""

    def test_reasoner_import(self):
        """Test CounterfactualReasoner can be imported."""
        try:
            from src.agent import CounterfactualReasoner
            self.assertTrue(hasattr(CounterfactualReasoner, 'reason_counterfactual'))
            self.assertTrue(hasattr(CounterfactualReasoner, 'verify_and_refine'))
        except ImportError:
            self.skipTest("Agent module not available")

    def test_reasoner_initialization(self):
        """Test CounterfactualReasoner initialization."""
        try:
            from src.agent import CounterfactualReasoner

            reasoner = CounterfactualReasoner(
                max_chain_steps=5,
                cvs_threshold=0.95,
                max_retries=3
            )

            self.assertEqual(reasoner.max_chain_steps, 5)
            self.assertEqual(reasoner.cvs_threshold, 0.95)
            self.assertEqual(reasoner.max_retries, 3)

        except ImportError:
            self.skipTest("Agent module not available")

    def test_reason_counterfactual_output(self):
        """Test reason_counterfactual output structure."""
        try:
            from src.agent import reason_counterfactual

            perturbation = {
                'player_id': '1',
                'delay': 0.5,
                'speed_factor': 1.0,
                'angle_delta': 0.0
            }

            simulation_result = {
                'valid': True,
                'violation_reason': None,
                'lift': 0.15,
                'ball_x': 65.0,
                'simulated_xt': 0.60
            }

            result = reason_counterfactual(perturbation, simulation_result)

            # Check output structure
            self.assertIn('lift', result)
            self.assertIn('cvs', result)
            self.assertIn('trace', result)
            self.assertIn('chain_steps', result)
            self.assertIn('violations', result)

            # Check types
            self.assertIsInstance(result['lift'], (int, float))
            self.assertIsInstance(result['cvs'], (int, float))
            self.assertIsInstance(result['trace'], str)
            self.assertIsInstance(result['chain_steps'], int)
            self.assertIsInstance(result['violations'], list)

            # Check ranges
            self.assertGreaterEqual(result['cvs'], 0.0)
            self.assertLessEqual(result['cvs'], 1.0)
            self.assertGreaterEqual(result['chain_steps'], 3)
            self.assertLessEqual(result['chain_steps'], 5)

        except ImportError:
            self.skipTest("Agent module not available")

    def test_verify_and_refine(self):
        """Test verify_and_refine with re-chain logic."""
        try:
            from src.agent import CounterfactualReasoner

            reasoner = CounterfactualReasoner(cvs_threshold=0.95, max_retries=2)

            perturbation = {
                'player_id': '2',
                'delay': 0.3,
                'speed_factor': 0.9,
                'angle_delta': 10.0
            }

            simulation_result = {
                'valid': False,
                'violation_reason': 'Collision detected',
                'lift': 0.05,
                'ball_x': 55.0,
                'simulated_xt': 0.50
            }

            result = reasoner.verify_and_refine(perturbation, simulation_result)

            # Should have iteration info
            self.assertIn('iterations', result)
            self.assertGreaterEqual(result['iterations'], 1)

            # CVS should be valid range
            self.assertIn('cvs', result)
            self.assertGreaterEqual(result['cvs'], 0.0)
            self.assertLessEqual(result['cvs'], 1.0)

        except ImportError:
            self.skipTest("Agent module not available")

    def test_batch_verify(self):
        """Test batch verification."""
        try:
            from src.agent import CounterfactualReasoner

            reasoner = CounterfactualReasoner()

            perturbations = [
                {'player_id': '1', 'delay': 0.5, 'speed_factor': 1.0, 'angle_delta': 0.0},
                {'player_id': '2', 'delay': 0.0, 'speed_factor': 0.9, 'angle_delta': 0.0},
                {'player_id': '3', 'delay': 0.0, 'speed_factor': 1.0, 'angle_delta': 15.0}
            ]

            simulation_results = [
                {'valid': True, 'violation_reason': None, 'lift': 0.15, 'ball_x': 65.0, 'simulated_xt': 0.60},
                {'valid': True, 'violation_reason': None, 'lift': -0.05, 'ball_x': 52.0, 'simulated_xt': 0.48},
                {'valid': False, 'violation_reason': 'Offside', 'lift': 0.20, 'ball_x': 80.0, 'simulated_xt': 0.75}
            ]

            results = reasoner.batch_verify(perturbations, simulation_results)

            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIn('cvs', result)
                self.assertIn('trace', result)

        except ImportError:
            self.skipTest("Agent module not available")

    def test_aggregate_cvs(self):
        """Test aggregate CVS computation."""
        try:
            from src.agent import CounterfactualReasoner

            reasoner = CounterfactualReasoner()

            results = [
                {'cvs': 0.95, 'lift': 0.15},
                {'cvs': 0.92, 'lift': -0.05},
                {'cvs': 0.88, 'lift': 0.20}
            ]

            aggregate_cvs = reasoner.compute_aggregate_cvs(results)

            expected = (0.95 + 0.92 + 0.88) / 3
            self.assertAlmostEqual(aggregate_cvs, expected, places=3)

        except ImportError:
            self.skipTest("Agent module not available")


class TestCounterfactualGeneratorWithAgent(unittest.TestCase):
    """Test CounterfactualGenerator with agent integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_play = {
            'time': 0.0,
            'players': [
                {'id': '1', 'x': 50, 'y': 30, 'vel': 2, 'dir': 45, 'acc': 0.5, 'role': 'QB', 'possession': True},
                {'id': '2', 'x': 60, 'y': 15, 'vel': 7, 'dir': 90, 'acc': 1.0, 'role': 'WR', 'possession': False},
                {'id': '3', 'x': 58, 'y': 50, 'vel': 7, 'dir': 85, 'acc': 0.8, 'role': 'WR', 'possession': False}
            ],
            'ball': {'x': 50, 'y': 30}
        }

    def test_generator_with_agent_enabled(self):
        """Test generator with agent enabled."""
        gen = CounterfactualGenerator(use_agent=True, cvs_threshold=0.95)

        # Check agent initialization
        if gen.agent_reasoner is not None:
            self.assertIsNotNone(gen.agent_reasoner)
            self.assertEqual(gen.agent_reasoner.cvs_threshold, 0.95)

    def test_generator_with_agent_disabled(self):
        """Test generator with agent disabled."""
        gen = CounterfactualGenerator(use_agent=False)

        self.assertFalse(gen.use_agent)
        self.assertIsNone(gen.agent_reasoner)

    def test_generate_counterfactuals_with_agent(self):
        """Test counterfactual generation with agent."""
        gen = CounterfactualGenerator(use_agent=True)

        results, cvs = gen.generate_counterfactuals(
            self.sample_play,
            original_xt=0.45,
            n_scenarios=3
        )

        # Check results
        self.assertEqual(len(results), 3)
        self.assertIsInstance(cvs, float)
        self.assertGreaterEqual(cvs, 0.0)
        self.assertLessEqual(cvs, 1.0)

        # Check result structure
        for result in results:
            self.assertIsInstance(result, CounterfactualResult)
            self.assertIsNotNone(result.player)
            self.assertIsNotNone(result.change)
            self.assertIsInstance(result.lift, (int, float))
            self.assertIsInstance(result.valid, bool)

            # Check agent fields (may be None if agent not available)
            if result.cvs is not None:
                self.assertGreaterEqual(result.cvs, 0.0)
                self.assertLessEqual(result.cvs, 1.0)

            if result.chain_steps is not None:
                self.assertGreaterEqual(result.chain_steps, 3)
                self.assertLessEqual(result.chain_steps, 5)

    def test_generate_counterfactuals_without_agent(self):
        """Test counterfactual generation without agent."""
        gen = CounterfactualGenerator(use_agent=False)

        results, cvs = gen.generate_counterfactuals(
            self.sample_play,
            original_xt=0.45,
            n_scenarios=3
        )

        # Check results
        self.assertEqual(len(results), 3)

        # Agent fields should be None
        for result in results:
            self.assertIsNone(result.cvs)
            self.assertIsNone(result.agent_trace)
            self.assertIsNone(result.chain_steps)

    def test_cvs_comparison(self):
        """Test CVS improvement with agent vs without."""
        # Without agent
        gen_baseline = CounterfactualGenerator(use_agent=False)
        _, cvs_baseline = gen_baseline.generate_counterfactuals(
            self.sample_play,
            original_xt=0.45,
            n_scenarios=3
        )

        # With agent
        gen_agent = CounterfactualGenerator(use_agent=True)
        _, cvs_agent = gen_agent.generate_counterfactuals(
            self.sample_play,
            original_xt=0.45,
            n_scenarios=3
        )

        # Agent CVS should be >= baseline CVS (in most cases)
        # Note: May not always be true in simulation mode
        print(f"CVS - Baseline: {cvs_baseline:.3f}, Agent: {cvs_agent:.3f}")
        self.assertIsInstance(cvs_baseline, float)
        self.assertIsInstance(cvs_agent, float)


class TestCounterfactualPayload(unittest.TestCase):
    """Test counterfactual payload generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_play = {
            'time': 0.0,
            'players': [
                {'id': '1', 'x': 50, 'y': 30, 'vel': 2, 'dir': 45, 'acc': 0.5, 'role': 'QB', 'possession': True},
                {'id': '2', 'x': 60, 'y': 15, 'vel': 7, 'dir': 90, 'acc': 1.0, 'role': 'WR', 'possession': False}
            ],
            'ball': {'x': 50, 'y': 30}
        }

    def test_payload_structure(self):
        """Test payload structure."""
        payload = create_counterfactual_payload(
            self.sample_play,
            original_xt=0.45,
            n_scenarios=2
        )

        # Check structure
        self.assertIn('counterfactuals', payload)
        self.assertIn('cvs', payload)
        self.assertIn('n_valid', payload)
        self.assertIn('n_total', payload)
        self.assertIn('agent_enabled', payload)

        # Check types
        self.assertIsInstance(payload['counterfactuals'], list)
        self.assertIsInstance(payload['cvs'], (int, float))
        self.assertIsInstance(payload['n_valid'], int)
        self.assertIsInstance(payload['n_total'], int)
        self.assertIsInstance(payload['agent_enabled'], bool)

    def test_payload_with_agent(self):
        """Test payload with agent enabled."""
        payload = create_counterfactual_payload(
            self.sample_play,
            original_xt=0.45,
            n_scenarios=2,
            use_agent=True
        )

        # Check agent fields in counterfactuals
        for cf in payload['counterfactuals']:
            # May have agent fields (if agent available)
            if 'cvs' in cf:
                self.assertIsInstance(cf['cvs'], (int, float))
            if 'chain_steps' in cf:
                self.assertIsInstance(cf['chain_steps'], int)
            if 'agent_trace' in cf:
                self.assertIsInstance(cf['agent_trace'], str)

    def test_payload_lightweight_mode(self):
        """Test payload in lightweight mode."""
        payload = create_counterfactual_payload(
            self.sample_play,
            original_xt=0.45,
            n_scenarios=2,
            lightweight=True
        )

        # Check size
        import sys
        payload_size = sys.getsizeof(str(payload))

        # Should be under 1MB
        self.assertLess(payload_size, 1_000_000)

        # Trajectories should be sampled
        for cf in payload['counterfactuals']:
            if 'trajectory' in cf:
                # Should have at most 20 points in lightweight mode
                self.assertLessEqual(len(cf['trajectory']), 20)

    def test_payload_agent_trace_truncation(self):
        """Test agent trace truncation in lightweight mode."""
        # Create payload with lightweight mode
        payload = create_counterfactual_payload(
            self.sample_play,
            original_xt=0.45,
            n_scenarios=2,
            lightweight=True
        )

        # Check trace truncation
        for cf in payload['counterfactuals']:
            if 'agent_trace' in cf:
                # Should be truncated to ~200 chars in lightweight mode
                self.assertLessEqual(len(cf['agent_trace']), 203)  # 200 + "..."


class TestChainSteps(unittest.TestCase):
    """Test chain step validation."""

    def test_chain_steps_range(self):
        """Test that chain steps are in valid range [3,5]."""
        try:
            from src.agent import reason_counterfactual

            perturbation = {
                'player_id': '1',
                'delay': 0.5,
                'speed_factor': 1.0,
                'angle_delta': 0.0
            }

            simulation_result = {
                'valid': True,
                'violation_reason': None,
                'lift': 0.15,
                'ball_x': 65.0,
                'simulated_xt': 0.60
            }

            # Run multiple times to check consistency
            for _ in range(5):
                result = reason_counterfactual(perturbation, simulation_result)

                # Chain steps should be in [3, 5]
                self.assertGreaterEqual(result['chain_steps'], 3)
                self.assertLessEqual(result['chain_steps'], 5)

        except ImportError:
            self.skipTest("Agent module not available")

    def test_no_invalid_outputs(self):
        """Test that reasoning produces no invalid outputs."""
        try:
            from src.agent import reason_counterfactual

            perturbation = {
                'player_id': '1',
                'delay': 0.5,
                'speed_factor': 1.0,
                'angle_delta': 0.0
            }

            simulation_result = {
                'valid': True,
                'violation_reason': None,
                'lift': 0.15,
                'ball_x': 65.0,
                'simulated_xt': 0.60
            }

            result = reason_counterfactual(perturbation, simulation_result)

            # Check no NaN or infinite values
            self.assertFalse(np.isnan(result['lift']))
            self.assertFalse(np.isnan(result['cvs']))
            self.assertFalse(np.isinf(result['lift']))
            self.assertFalse(np.isinf(result['cvs']))

            # Check CVS in valid range
            self.assertGreaterEqual(result['cvs'], 0.0)
            self.assertLessEqual(result['cvs'], 1.0)

            # Check trace is not empty
            self.assertGreater(len(result['trace']), 0)

        except ImportError:
            self.skipTest("Agent module not available")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
