#!/usr/bin/env python3
"""
FieldSense AI v3.1 - Agent-Assisted Calibration Unit Tests

Tests for agent integration with LoRA calibration.
"""

import unittest
import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.calibrate import (
    compute_entropy,
    identify_target_zone,
    PlayDataset
)


class TestEntropyComputation(unittest.TestCase):
    """Test entropy computation."""

    def test_compute_entropy_shape(self):
        """Test entropy computation output shape."""
        import torch

        # Create dummy predictions (B=4, H=105, W=68)
        predictions = torch.randn(4, 105, 68)

        entropy = compute_entropy(predictions)

        # Should return (B,) tensor
        self.assertEqual(entropy.shape, (4,))

    def test_compute_entropy_range(self):
        """Test entropy values are in [0, 1] range."""
        import torch

        predictions = torch.randn(10, 105, 68)
        entropy = compute_entropy(predictions)

        # All values should be in [0, 1]
        self.assertTrue(torch.all(entropy >= 0))
        self.assertTrue(torch.all(entropy <= 1))

    def test_compute_entropy_uniform(self):
        """Test entropy of uniform distribution is high."""
        import torch

        # Uniform predictions (high entropy)
        predictions = torch.ones(2, 105, 68) * 0.5

        entropy = compute_entropy(predictions)

        # Uniform distribution should have relatively high entropy
        self.assertGreater(entropy[0].item(), 0.5)

    def test_compute_entropy_peaked(self):
        """Test entropy of peaked distribution is low."""
        import torch

        # Create peaked distribution (low entropy)
        predictions = torch.zeros(2, 105, 68)
        predictions[:, 50, 34] = 10.0  # Single high peak

        entropy = compute_entropy(predictions)

        # Peaked distribution should have lower entropy
        self.assertLess(entropy[0].item(), 0.5)


class TestZoneIdentification(unittest.TestCase):
    """Test zone identification."""

    def test_left_attack_zone(self):
        """Test left attack zone identification."""
        frame = {
            'ball': {'x': 70.0, 'y': 15.0},
            'players': []
        }

        zone = identify_target_zone(frame)
        self.assertEqual(zone, 'left_attack')

    def test_right_attack_zone(self):
        """Test right attack zone identification."""
        frame = {
            'ball': {'x': 70.0, 'y': 50.0},
            'players': []
        }

        zone = identify_target_zone(frame)
        self.assertEqual(zone, 'right_attack')

    def test_center_attack_zone(self):
        """Test center attack zone identification."""
        frame = {
            'ball': {'x': 70.0, 'y': 34.0},
            'players': []
        }

        zone = identify_target_zone(frame)
        self.assertEqual(zone, 'center_attack')

    def test_left_defense_zone(self):
        """Test left defense zone identification."""
        frame = {
            'ball': {'x': 30.0, 'y': 15.0},
            'players': []
        }

        zone = identify_target_zone(frame)
        self.assertEqual(zone, 'left_defense')

    def test_right_defense_zone(self):
        """Test right defense zone identification."""
        frame = {
            'ball': {'x': 30.0, 'y': 50.0},
            'players': []
        }

        zone = identify_target_zone(frame)
        self.assertEqual(zone, 'right_defense')

    def test_zone_from_players(self):
        """Test zone identification from player positions."""
        frame = {
            'ball': None,
            'players': [
                {'x': 70, 'y': 15, 'possession': True},
                {'x': 75, 'y': 20, 'possession': False}
            ]
        }

        zone = identify_target_zone(frame)
        # Should be left_attack based on possession player
        self.assertEqual(zone, 'left_attack')

    def test_zone_default(self):
        """Test default zone when no data."""
        frame = {
            'ball': None,
            'players': []
        }

        zone = identify_target_zone(frame)
        self.assertEqual(zone, 'center_attack')


class TestPlayDataset(unittest.TestCase):
    """Test PlayDataset with sample weighting."""

    def setUp(self):
        """Set up test fixtures."""
        self.frames = [
            {
                'time': 0.0,
                'players': [
                    {'id': '1', 'x': 50, 'y': 30, 'vel': 2, 'dir': 45,
                     'acc': 0.5, 'role': 'QB', 'possession': True}
                ],
                'ball': {'x': 50, 'y': 30}
            },
            {
                'time': 0.1,
                'players': [
                    {'id': '1', 'x': 52, 'y': 32, 'vel': 2, 'dir': 45,
                     'acc': 0.5, 'role': 'QB', 'possession': True}
                ],
                'ball': {'x': 52, 'y': 32}
            }
        ]

    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = PlayDataset(self.frames)

        self.assertEqual(len(dataset), 2)
        self.assertIsNotNone(dataset.labels)
        self.assertEqual(len(dataset.labels), 2)

    def test_dataset_with_weights(self):
        """Test dataset with custom weights."""
        weights = np.array([1.5, 0.8])
        dataset = PlayDataset(self.frames, sample_weights=weights)

        np.testing.assert_array_equal(dataset.sample_weights, weights)

    def test_dataset_default_weights(self):
        """Test dataset creates default weights."""
        dataset = PlayDataset(self.frames)

        # Default weights should be all 1.0
        np.testing.assert_array_equal(dataset.sample_weights, np.ones(2))

    def test_dataset_getitem(self):
        """Test dataset __getitem__."""
        dataset = PlayDataset(self.frames)

        frame_tensor, target_tensor = dataset[0]

        # Check shapes
        self.assertEqual(frame_tensor.shape[0], 11)  # 11 features
        self.assertEqual(target_tensor.shape, (105, 68))

    def test_get_weights(self):
        """Test get_weights method."""
        weights = np.array([1.2, 1.8])
        dataset = PlayDataset(self.frames, sample_weights=weights)

        retrieved_weights = dataset.get_weights()
        np.testing.assert_array_equal(retrieved_weights, weights)

    def test_update_weights(self):
        """Test update_weights method."""
        dataset = PlayDataset(self.frames)

        new_weights = np.array([2.0, 0.5])
        dataset.update_weights(new_weights)

        np.testing.assert_array_equal(dataset.sample_weights, new_weights)

    def test_update_weights_mismatch(self):
        """Test update_weights with mismatched length."""
        dataset = PlayDataset(self.frames)

        # Try to update with wrong length
        wrong_weights = np.array([1.0, 2.0, 3.0])
        dataset.update_weights(wrong_weights)

        # Weights should remain unchanged
        np.testing.assert_array_equal(dataset.sample_weights, np.ones(2))


class TestAgentIntegration(unittest.TestCase):
    """Test agent integration with calibration."""

    def test_agent_availability(self):
        """Test agent module availability."""
        try:
            from src.agent import CalibrationReasoner
            agent_available = True
        except ImportError:
            agent_available = False

        # Just check that we can determine availability
        self.assertIsInstance(agent_available, bool)

    def test_calibration_reasoner_import(self):
        """Test CalibrationReasoner can be imported."""
        try:
            from src.agent import CalibrationReasoner

            # Check that class exists
            self.assertTrue(hasattr(CalibrationReasoner, 'reason_calibration'))
            self.assertTrue(hasattr(CalibrationReasoner, 'compute_sample_weights'))

        except ImportError:
            # Agent module not available - skip test
            self.skipTest("Agent module not available")

    def test_reason_calibration_function(self):
        """Test reason_calibration convenience function."""
        try:
            from src.agent import reason_calibration

            # Create dummy high-entropy plays
            plays = [
                {
                    'frame': {'ball': {'x': 70, 'y': 15}, 'players': []},
                    'entropy': 0.85,
                    'target_zone': 'left_attack'
                },
                {
                    'frame': {'ball': {'x': 70, 'y': 18}, 'players': []},
                    'entropy': 0.90,
                    'target_zone': 'left_attack'
                }
            ]

            # Call reasoning
            weights = reason_calibration(plays, entropy_threshold=0.8)

            # Check output structure
            self.assertIsInstance(weights, dict)
            self.assertIn('left_attack', weights)
            self.assertIn('right_attack', weights)
            self.assertIn('center_attack', weights)

            # Check weight values are reasonable
            for key, val in weights.items():
                if key != 'reasoning':
                    self.assertIsInstance(val, (int, float))
                    self.assertGreaterEqual(val, 0.5)
                    self.assertLessEqual(val, 2.0)

        except ImportError:
            self.skipTest("Agent module not available")

    def test_compute_sample_weights(self):
        """Test compute_sample_weights method."""
        try:
            from src.agent import CalibrationReasoner

            reasoner = CalibrationReasoner()

            # Create plays with different zones
            plays = [
                {'ball': {'x': 70, 'y': 15}, 'players': []},  # left_attack
                {'ball': {'x': 70, 'y': 50}, 'players': []},  # right_attack
                {'ball': {'x': 70, 'y': 34}, 'players': []}   # center_attack
            ]

            # Zone weights (emphasize left)
            zone_weights = {
                'left_attack': 1.8,
                'right_attack': 1.0,
                'center_attack': 1.2,
                'left_defense': 1.0,
                'right_defense': 1.0
            }

            # Compute weights
            weights = reasoner.compute_sample_weights(plays, zone_weights)

            # Check output
            self.assertEqual(len(weights), 3)
            self.assertTrue(np.all(weights > 0))

            # Mean should be normalized to 1.0
            self.assertAlmostEqual(np.mean(weights), 1.0, places=5)

        except ImportError:
            self.skipTest("Agent module not available")


class TestCalibrationWithAgent(unittest.TestCase):
    """Test calibration function with agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.frames = [
            {
                'time': i * 0.1,
                'players': [
                    {'id': '1', 'x': 50 + i, 'y': 30, 'vel': 2, 'dir': 45,
                     'acc': 0.5, 'role': 'QB', 'possession': True}
                ],
                'ball': {'x': 50 + i, 'y': 30}
            }
            for i in range(10)
        ]

    def test_calibration_params(self):
        """Test calibration function accepts agent parameters."""
        from src.model.calibrate import calibrate_model
        from src.model.adapter import create_adapter_model

        # Check function signature
        import inspect
        sig = inspect.signature(calibrate_model)

        # Check new parameters exist
        self.assertIn('use_agent', sig.parameters)
        self.assertIn('entropy_threshold', sig.parameters)
        self.assertIn('agent_time_budget', sig.parameters)

        # Check defaults
        self.assertEqual(sig.parameters['use_agent'].default, True)
        self.assertEqual(sig.parameters['entropy_threshold'].default, 0.8)
        self.assertEqual(sig.parameters['agent_time_budget'].default, 10.0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
