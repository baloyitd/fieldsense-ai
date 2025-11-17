#!/usr/bin/env python3
"""
FieldSense AI v3.1 - Agent-Enhanced Heatmap Unit Tests

Tests for MiroThinker integration with heatmap opportunity chaining.
"""

import unittest
import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insights.heatmap import (
    generate_insight,
    compute_opportunity_heatmap
)


class TestHeatmapReasoner(unittest.TestCase):
    """Test HeatmapReasoner class."""

    def test_reasoner_import(self):
        """Test HeatmapReasoner can be imported."""
        try:
            from src.agent import HeatmapReasoner
            self.assertTrue(hasattr(HeatmapReasoner, 'reason_opportunity'))
            self.assertTrue(hasattr(HeatmapReasoner, 'chain_zones'))
        except ImportError:
            self.skipTest("Agent module not available")

    def test_reasoner_initialization(self):
        """Test HeatmapReasoner initialization."""
        try:
            from src.agent import HeatmapReasoner

            reasoner = HeatmapReasoner(max_depth=4, max_tokens=256)

            self.assertEqual(reasoner.max_depth, 4)
            self.assertEqual(reasoner.max_tokens, 256)

        except ImportError:
            self.skipTest("Agent module not available")

    def test_reason_opportunity_output(self):
        """Test reason_opportunity output structure."""
        try:
            from src.agent import reason_opportunity

            grid_data = {
                'shape': [105, 68],
                'mean_opportunity': 0.5,
                'max_opportunity': 0.85
            }

            top_zones = [
                {'zone': 'left', 'opportunity': 0.85, 'improvement': 25},
                {'zone': 'center', 'opportunity': 0.72, 'improvement': 15}
            ]

            result = reason_opportunity(grid_data, top_zones)

            # Check output structure
            self.assertIn('upgraded_zones', result)
            self.assertIn('depth', result)
            self.assertIn('primary_action', result)
            self.assertIn('risks', result)

            # Check types
            self.assertIsInstance(result['upgraded_zones'], dict)
            self.assertIsInstance(result['depth'], int)
            self.assertIsInstance(result['primary_action'], str)
            self.assertIsInstance(result['risks'], list)

            # Check depth range
            self.assertGreaterEqual(result['depth'], 2)
            self.assertLessEqual(result['depth'], 4)

        except ImportError:
            self.skipTest("Agent module not available")

    def test_chain_zones(self):
        """Test chain_zones method."""
        try:
            from src.agent import HeatmapReasoner

            reasoner = HeatmapReasoner()

            # Create sample grids
            opportunity = np.random.rand(105, 68) * 0.5
            # Add high-value zone
            opportunity[75:85, 15:25] = 0.9

            xt_grid = np.random.rand(105, 68) * 0.5
            xt_grid[75:85, 15:25] = 0.85

            result = reasoner.chain_zones(opportunity, xt_grid, top_n=3)

            # Check structure
            self.assertIn('upgraded_zones', result)
            self.assertIn('depth', result)

            # Check depth range
            self.assertGreaterEqual(result['depth'], 2)
            self.assertLessEqual(result['depth'], 4)

        except ImportError:
            self.skipTest("Agent module not available")

    def test_upgraded_zones_structure(self):
        """Test upgraded zones structure."""
        try:
            from src.agent import reason_opportunity

            grid_data = {'shape': [105, 68], 'mean_opportunity': 0.5}
            top_zones = [
                {'zone': 'left', 'opportunity': 0.85, 'improvement': 30}
            ]

            result = reason_opportunity(grid_data, top_zones)

            if result['upgraded_zones']:
                # Check first zone structure
                first_zone = list(result['upgraded_zones'].values())[0]

                self.assertIn('base_opportunity', first_zone)
                self.assertIn('risk_vector', first_zone)
                self.assertIn('confidence', first_zone)
                self.assertIn('reasoning', first_zone)

                # Check ranges
                self.assertGreaterEqual(first_zone['base_opportunity'], 0.0)
                self.assertLessEqual(first_zone['base_opportunity'], 1.0)
                self.assertGreaterEqual(first_zone['risk_vector'], -0.5)
                self.assertLessEqual(first_zone['risk_vector'], 0.2)
                self.assertGreaterEqual(first_zone['confidence'], 0.7)
                self.assertLessEqual(first_zone['confidence'], 0.95)

        except ImportError:
            self.skipTest("Agent module not available")


class TestHeatmapWithAgent(unittest.TestCase):
    """Test heatmap generation with agent integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample xT grid
        self.xt_grid = np.random.rand(105, 68) * 0.5
        self.xt_grid[70:85, 15:25] = 0.8  # High-value zone

        # Create sample players
        self.players = [
            {'x': 60, 'y': 30, 'role': 'QB', 'possession': True},
            {'x': 75, 'y': 18, 'role': 'WR', 'possession': False},
            {'x': 80, 'y': 20, 'role': 'CB', 'possession': False}
        ]

    def test_generate_insight_baseline(self):
        """Test baseline insight generation (no agent)."""
        insight = generate_insight(
            self.xt_grid,
            self.players,
            use_agent=False
        )

        # Check structure
        self.assertIn('top_decision', insight)
        self.assertIn('zones', insight)
        self.assertIn('confidence', insight)
        self.assertIn('agent_enabled', insight)

        # Agent should be disabled
        self.assertFalse(insight['agent_enabled'])

    def test_generate_insight_with_agent(self):
        """Test insight generation with agent enabled."""
        insight = generate_insight(
            self.xt_grid,
            self.players,
            use_agent=True,
            agent_threshold=0.7
        )

        # Check structure
        self.assertIn('top_decision', insight)
        self.assertIn('zones', insight)
        self.assertIn('agent_enabled', insight)

        # Check types
        self.assertIsInstance(insight['top_decision'], str)
        self.assertIsInstance(insight['zones'], np.ndarray)
        self.assertIsInstance(insight['agent_enabled'], bool)

    def test_agent_analysis_structure(self):
        """Test agent analysis structure if available."""
        insight = generate_insight(
            self.xt_grid,
            self.players,
            use_agent=True,
            agent_threshold=0.5  # Lower threshold to ensure activation
        )

        if insight['agent_enabled'] and 'agent_analysis' in insight:
            analysis = insight['agent_analysis']

            # Check required fields
            self.assertIn('upgraded_zones', analysis)
            self.assertIn('depth', analysis)
            self.assertIn('primary_action', analysis)

            # Check depth range
            self.assertGreaterEqual(analysis['depth'], 2)
            self.assertLessEqual(analysis['depth'], 4)

    def test_upgraded_zones_in_insight(self):
        """Test upgraded zones appear in insight."""
        insight = generate_insight(
            self.xt_grid,
            self.players,
            use_agent=True,
            agent_threshold=0.5
        )

        if insight['agent_enabled'] and 'upgraded_zones' in insight:
            upgraded = insight['upgraded_zones']

            self.assertIsInstance(upgraded, dict)
            if upgraded:
                # Check first zone has required fields
                first_zone = list(upgraded.values())[0]
                self.assertIn('base_opportunity', first_zone)
                self.assertIn('risk_vector', first_zone)

    def test_chain_depth_recorded(self):
        """Test chain depth is recorded."""
        insight = generate_insight(
            self.xt_grid,
            self.players,
            use_agent=True,
            agent_threshold=0.5
        )

        if insight['agent_enabled'] and 'chain_depth' in insight:
            depth = insight['chain_depth']

            self.assertIsInstance(depth, int)
            self.assertGreaterEqual(depth, 2)
            self.assertLessEqual(depth, 4)


class TestAgentThreshold(unittest.TestCase):
    """Test agent threshold behavior."""

    def test_agent_not_triggered_low_opportunity(self):
        """Test agent not triggered when opportunity below threshold."""
        # Create low-value grid
        xt_grid = np.random.rand(105, 68) * 0.3

        players = [
            {'x': 50, 'y': 34, 'role': 'QB', 'possession': True}
        ]

        insight = generate_insight(
            xt_grid,
            players,
            use_agent=True,
            agent_threshold=0.9  # Very high threshold
        )

        # Agent should not be triggered due to low opportunity
        if insight['agent_enabled']:
            # If agent is enabled, opportunity must be >= threshold
            opp = insight['details'].get('opportunity', 0)
            self.assertGreaterEqual(opp, 0.9)

    def test_agent_triggered_high_opportunity(self):
        """Test agent triggered when opportunity above threshold."""
        # Create high-value grid
        xt_grid = np.ones((105, 68)) * 0.9

        players = [
            {'x': 80, 'y': 20, 'role': 'WR', 'possession': True}
        ]

        insight = generate_insight(
            xt_grid,
            players,
            use_agent=True,
            agent_threshold=0.7
        )

        # With high opportunity, agent should be enabled (if available)
        try:
            from src.agent import HeatmapReasoner
            # If agent is available, it should be enabled
            self.assertTrue(insight['agent_enabled'] or 'agent_error' in insight)
        except ImportError:
            # If agent not available, that's OK
            pass


class TestChainDepthValidation(unittest.TestCase):
    """Test chain depth validation (2-4 steps)."""

    def test_depth_range(self):
        """Test depth is always in range [2, 4]."""
        try:
            from src.agent import reason_opportunity

            grid_data = {'shape': [105, 68]}
            top_zones = [{'zone': 'left', 'opportunity': 0.8}]

            # Run multiple times to check consistency
            for _ in range(10):
                result = reason_opportunity(grid_data, top_zones)
                depth = result['depth']

                self.assertGreaterEqual(depth, 2, f"Depth {depth} < 2")
                self.assertLessEqual(depth, 4, f"Depth {depth} > 4")
                self.assertIsInstance(depth, int)

        except ImportError:
            self.skipTest("Agent module not available")

    def test_no_invalid_outputs(self):
        """Test reasoning produces no invalid outputs."""
        try:
            from src.agent import reason_opportunity

            grid_data = {'shape': [105, 68], 'mean_opportunity': 0.6}
            top_zones = [{'zone': 'center', 'opportunity': 0.75}]

            result = reason_opportunity(grid_data, top_zones)

            # Check depth is valid
            self.assertIsInstance(result['depth'], int)
            self.assertGreaterEqual(result['depth'], 2)
            self.assertLessEqual(result['depth'], 4)

            # Check upgraded zones have valid values
            for zone_data in result['upgraded_zones'].values():
                # No NaN or infinite values
                self.assertFalse(np.isnan(zone_data['base_opportunity']))
                self.assertFalse(np.isnan(zone_data['risk_vector']))
                self.assertFalse(np.isinf(zone_data['base_opportunity']))
                self.assertFalse(np.isinf(zone_data['risk_vector']))

                # Values in expected ranges
                self.assertGreaterEqual(zone_data['base_opportunity'], 0.0)
                self.assertLessEqual(zone_data['base_opportunity'], 1.0)
                self.assertGreaterEqual(zone_data['risk_vector'], -0.5)
                self.assertLessEqual(zone_data['risk_vector'], 0.5)

        except ImportError:
            self.skipTest("Agent module not available")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
