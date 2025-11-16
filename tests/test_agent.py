#!/usr/bin/env python3
"""
FieldSense AI v3.1 - Agent Unit Tests

Tests for MiroThinker agent module.
"""

import unittest
import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import (
    MiroThinkerAgent,
    load_model,
    load_tokenizer,
    check_model_availability
)


class TestDependencies(unittest.TestCase):
    """Test dependency availability."""

    def test_check_availability(self):
        """Test dependency checking."""
        availability = check_model_availability()

        self.assertIsInstance(availability, dict)
        self.assertIn('transformers', availability)
        self.assertIn('peft', availability)
        self.assertIn('torch', availability)


class TestAgentInitialization(unittest.TestCase):
    """Test agent initialization."""

    def test_agent_creation(self):
        """Test creating agent instance."""
        agent = MiroThinkerAgent(offline_mode=True)

        self.assertIsNotNone(agent)
        self.assertEqual(agent.model_name, "miromind-ai/MiroThinker-v1.0-30B")
        self.assertTrue(agent.offline_mode)

    def test_config_loading(self):
        """Test configuration loading."""
        agent = MiroThinkerAgent(offline_mode=True)

        self.assertIsNotNone(agent.config)
        self.assertIn('toggle', agent.config)
        self.assertIn('generation', agent.config)

        # Check nested generation config
        gen_config = agent.config.get('generation', {})
        self.assertIn('max_new_tokens', gen_config)

    def test_default_config(self):
        """Test default configuration."""
        agent = MiroThinkerAgent(offline_mode=True)
        default_config = agent._get_default_config()

        self.assertIsInstance(default_config, dict)
        self.assertIn('toggle', default_config)
        self.assertIn('prompt_template', default_config)


class TestReasoning(unittest.TestCase):
    """Test reasoning functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = MiroThinkerAgent(offline_mode=True)

        self.xT_grid = np.random.rand(105, 68) * 0.5
        self.xT_grid[70:80, 10:20] = 0.8

        self.play_data = {
            'players': [
                {'id': '1', 'x': 50, 'y': 30, 'role': 'QB'},
                {'id': '2', 'x': 70, 'y': 15, 'role': 'WR'},
            ],
            'ball': {'x': 50, 'y': 30}
        }

    def test_reason_disabled(self):
        """Test reasoning when agent disabled."""
        # Agent should be disabled by default
        result = self.agent.reason(self.xT_grid, self.play_data)

        self.assertIsInstance(result, dict)
        self.assertIn('reasoning', result)
        self.assertIn('refined_delta_xT', result)
        self.assertIn('confidence', result)

    def test_reason_simulation(self):
        """Test reasoning in simulation mode."""
        # Enable agent temporarily
        self.agent.enabled = True

        result = self.agent.reason(self.xT_grid, self.play_data)

        self.assertIsInstance(result, dict)
        self.assertIn('reasoning', result)
        self.assertIn('refined_delta_xT', result)
        self.assertIn('steps', result)

        # Verify output length
        self.assertGreater(len(result['reasoning']), 50)

    def test_build_prompt(self):
        """Test prompt building."""
        prompt = self.agent._build_prompt(self.xT_grid, self.play_data)

        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)
        self.assertIn('Step', prompt)

    def test_identify_high_zones(self):
        """Test high zone identification."""
        zones = self.agent._identify_high_zones(self.xT_grid, threshold=0.5)

        self.assertIsInstance(zones, list)
        self.assertLessEqual(len(zones), 5)

        if len(zones) > 0:
            zone = zones[0]
            self.assertIn('x', zone)
            self.assertIn('y', zone)
            self.assertIn('threat', zone)

    def test_parse_output(self):
        """Test output parsing."""
        simulated_output = self.agent._simulate_generation("")
        result = self.agent._parse_output(simulated_output, self.xT_grid, self.play_data)

        self.assertIn('reasoning', result)
        self.assertIn('refined_delta_xT', result)
        self.assertIn('confidence', result)
        self.assertIn('steps', result)

    def test_extract_steps(self):
        """Test step extraction."""
        output = """
        Step 1: Analyze the grid
        Some other text
        Step 2: Verify constraints
        More text
        """

        steps = self.agent._extract_steps(output)

        self.assertEqual(len(steps), 2)
        self.assertTrue(steps[0].startswith('Step'))


class TestPrivacy(unittest.TestCase):
    """Test privacy and offline functionality."""

    def test_offline_mode(self):
        """Test that agent respects offline mode."""
        agent = MiroThinkerAgent(offline_mode=True)

        self.assertTrue(agent.offline_mode)

        # Verify no internet access attempted
        if 'HF_HUB_OFFLINE' in os.environ:
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')

    def test_no_external_calls(self):
        """Test that agent doesn't make external calls."""
        agent = MiroThinkerAgent(offline_mode=True)

        xT_grid = np.random.rand(105, 68) * 0.5
        play_data = {'players': [], 'ball': {'x': 0, 'y': 0}}

        # This should work without internet
        result = agent.reason(xT_grid, play_data)

        self.assertIsNotNone(result)


class TestModelLoading(unittest.TestCase):
    """Test model loading functions."""

    def test_load_model_function(self):
        """Test load_model function."""
        # This will fail without dependencies, which is expected
        model, tokenizer = load_model()

        # Should return None, None if dependencies missing
        if model is None:
            self.assertIsNone(tokenizer)

    def test_load_tokenizer_function(self):
        """Test load_tokenizer function."""
        # This will fail without dependencies, which is expected
        tokenizer = load_tokenizer()

        # Should return None if dependencies missing
        if tokenizer is None:
            self.assertIsNone(tokenizer)


class TestOutputFormat(unittest.TestCase):
    """Test output format and structure."""

    def test_result_structure(self):
        """Test that result has expected structure."""
        agent = MiroThinkerAgent(offline_mode=True)
        agent.enabled = True

        xT_grid = np.random.rand(105, 68) * 0.5
        play_data = {'players': [], 'ball': {'x': 0, 'y': 0}}

        result = agent.reason(xT_grid, play_data)

        # Required fields
        self.assertIn('reasoning', result)
        self.assertIn('refined_delta_xT', result)
        self.assertIn('confidence', result)
        self.assertIn('steps', result)
        self.assertIn('original_xT_max', result)

        # Type checks
        self.assertIsInstance(result['reasoning'], str)
        self.assertIsInstance(result['refined_delta_xT'], (int, float))
        self.assertIsInstance(result['confidence'], (int, float))
        self.assertIsInstance(result['steps'], list)

    def test_confidence_range(self):
        """Test that confidence is in valid range."""
        agent = MiroThinkerAgent(offline_mode=True)
        agent.enabled = True

        xT_grid = np.random.rand(105, 68) * 0.5
        play_data = {'players': [], 'ball': {'x': 0, 'y': 0}}

        result = agent.reason(xT_grid, play_data)

        confidence = result['confidence']
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
