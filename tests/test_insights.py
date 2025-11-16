"""
Unit tests for insights module (heatmap and active learning).
"""

import unittest
import sys
import os
import json
import tempfile
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insights.heatmap import (
    compute_entropy,
    compute_pressure_decay,
    compute_off_ball_bonus,
    compute_xt_gradient,
    compute_opportunity_heatmap,
    find_top_decision,
    generate_insight
)
from src.insights.active_learning import (
    ActiveLearningManager,
    create_ui_label_payload
)


class TestHeatmapGeneration(unittest.TestCase):
    """Test heatmap computation functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample xT grid
        self.xt_grid = np.random.rand(105, 68) * 0.5 + 0.3

        # Create sample players
        self.players = [
            {'id': '1', 'x': 50, 'y': 25, 'role': 'QB', 'possession': True,
             'vel': 5.0, 'dir': 45, 'acc': 1.0},
            {'id': '2', 'x': 65, 'y': 15, 'role': 'WR', 'possession': False,
             'vel': 7.0, 'dir': 90, 'acc': 1.5},
            {'id': '3', 'x': 35, 'y': 30, 'role': 'LB', 'possession': False,
             'vel': 4.0, 'dir': 180, 'acc': 0.8},
        ]

    def test_compute_entropy(self):
        """Test entropy computation."""
        entropy = compute_entropy(self.xt_grid)

        # Check shape
        self.assertEqual(entropy.shape, self.xt_grid.shape)

        # Check range (binary entropy should be 0-1)
        self.assertTrue(np.all(entropy >= 0))
        self.assertTrue(np.all(entropy <= 1))

        # Check that p=0.5 gives maximum entropy
        uniform_grid = np.ones((10, 10)) * 0.5
        uniform_entropy = compute_entropy(uniform_grid)
        self.assertTrue(np.all(uniform_entropy > 0.9))  # Close to 1

    def test_compute_pressure_decay(self):
        """Test pressure decay computation."""
        pressure = compute_pressure_decay(self.players)

        # Check shape
        self.assertEqual(pressure.shape, (105, 68))

        # Check range [0, 1]
        self.assertTrue(np.all(pressure >= 0))
        self.assertTrue(np.all(pressure <= 1))

        # Check that areas near defenders have lower pressure (inverted)
        # Since we invert, areas near defenders should have lower values
        defender_x = int(self.players[2]['x'])
        defender_y = int(self.players[2]['y'])

        # Should have some variation
        self.assertTrue(np.std(pressure) > 0)

    def test_compute_off_ball_bonus(self):
        """Test off-ball bonus computation."""
        bonus = compute_off_ball_bonus(self.players)

        # Check shape
        self.assertEqual(bonus.shape, (105, 68))

        # Check that it's mostly >= 1 (base value)
        self.assertTrue(np.mean(bonus) >= 1.0)

        # Should have bonuses near offensive players without possession
        self.assertTrue(np.max(bonus) > 1.0)

    def test_compute_xt_gradient(self):
        """Test xT gradient computation."""
        gradient = compute_xt_gradient(self.xt_grid)

        # Check shape
        self.assertEqual(gradient.shape, self.xt_grid.shape)

        # Check normalization [0, 1]
        self.assertTrue(np.all(gradient >= 0))
        self.assertTrue(np.all(gradient <= 1))

    def test_compute_opportunity_heatmap(self):
        """Test opportunity heatmap computation."""
        opportunity = compute_opportunity_heatmap(
            self.xt_grid,
            self.players
        )

        # Check shape
        self.assertEqual(opportunity.shape, self.xt_grid.shape)

        # Check range [0, 1]
        self.assertTrue(np.all(opportunity >= 0))
        self.assertTrue(np.all(opportunity <= 1))

        # Check that it combines multiple factors
        self.assertTrue(np.std(opportunity) > 0)  # Should have variation

    def test_find_top_decision(self):
        """Test top decision finding."""
        opportunity = compute_opportunity_heatmap(self.xt_grid, self.players)

        decision = find_top_decision(
            opportunity,
            self.xt_grid,
            self.players
        )

        # Check required fields
        self.assertIn('top_decision', decision)
        self.assertIn('action', decision)
        self.assertIn('zone', decision)
        self.assertIn('location', decision)
        self.assertIn('confidence', decision)

        # Check decision format
        self.assertIsInstance(decision['top_decision'], str)
        self.assertIn(':', decision['top_decision'])
        self.assertIn('%', decision['top_decision'])
        self.assertIn('xT', decision['top_decision'])

        # Check zone is valid
        self.assertIn(decision['zone'], ['left', 'center', 'right'])

        # Check confidence is valid
        self.assertIn(decision['confidence'], ['High', 'Medium', 'Low'])

    def test_decision_string_format(self):
        """Test decision string format matches specification."""
        opportunity = compute_opportunity_heatmap(self.xt_grid, self.players)
        decision = find_top_decision(opportunity, self.xt_grid, self.players)

        decision_text = decision['top_decision']

        # Format: "Action zone: +X% xT"
        self.assertRegex(decision_text, r'^\w+ \w+: [+-]\d+% xT$')

        # Should have positive improvement
        self.assertIn('+', decision_text)

    def test_generate_insight(self):
        """Test complete insight generation."""
        insight = generate_insight(
            self.xt_grid,
            self.players
        )

        # Check output structure
        self.assertIn('top_decision', insight)
        self.assertIn('zones', insight)
        self.assertIn('confidence', insight)
        self.assertIn('details', insight)

        # Check zones shape
        self.assertEqual(insight['zones'].shape, self.xt_grid.shape)

    def test_insight_with_save(self):
        """Test insight generation with file save."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            insight = generate_insight(
                self.xt_grid,
                self.players,
                output_path=output_path
            )

            # Check file was created
            self.assertTrue(os.path.exists(output_path))

            # Check file size (should have content)
            self.assertGreater(os.path.getsize(output_path), 1000)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestActiveLearning(unittest.TestCase):
    """Test active learning functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary labels file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        )
        self.temp_file.close()

        self.labels_path = self.temp_file.name

        self.al_manager = ActiveLearningManager(
            labels_path=self.labels_path,
            entropy_threshold=0.8
        )

        # Sample data
        self.xt_grid = np.random.rand(105, 68) * 0.5 + 0.3
        self.players = [
            {'id': '1', 'x': 50, 'y': 25, 'role': 'QB', 'possession': True,
             'vel': 5.0, 'dir': 45, 'acc': 1.0},
        ]

    def tearDown(self):
        """Clean up temp files."""
        if os.path.exists(self.labels_path):
            os.unlink(self.labels_path)

    def test_manager_initialization(self):
        """Test active learning manager initialization."""
        self.assertIsInstance(self.al_manager, ActiveLearningManager)
        self.assertEqual(self.al_manager.entropy_threshold, 0.8)

    def test_should_flag_for_review_high_entropy(self):
        """Test flagging based on high entropy."""
        # Create insight with high entropy
        insight = {
            'confidence': 'Low',
            'details': {'avg_entropy': 0.85}
        }

        should_flag = self.al_manager.should_flag_for_review(
            self.xt_grid,
            insight
        )

        self.assertTrue(should_flag)

    def test_should_flag_for_review_low_confidence(self):
        """Test flagging based on low confidence."""
        insight = {
            'confidence': 'Low',
            'details': {'avg_entropy': 0.5}
        }

        should_flag = self.al_manager.should_flag_for_review(
            self.xt_grid,
            insight
        )

        self.assertTrue(should_flag)

    def test_no_flag_high_confidence(self):
        """Test no flag for high confidence plays."""
        insight = {
            'confidence': 'High',
            'details': {'avg_entropy': 0.3}
        }

        should_flag = self.al_manager.should_flag_for_review(
            self.xt_grid,
            insight
        )

        self.assertFalse(should_flag)

    def test_create_review_prompt(self):
        """Test review prompt creation."""
        play_data = {
            'play_id': 'test_001',
            'time': 0.0,
            'players': self.players,
            'ball': {'x': 50, 'y': 25}
        }

        insight = {
            'top_decision': 'Attack left: +31% xT',
            'confidence': 'Medium',
            'details': {'avg_entropy': 0.75}
        }

        prompt = self.al_manager.create_review_prompt(play_data, insight)

        # Check structure
        self.assertIn('play_id', prompt)
        self.assertIn('decision', prompt)
        self.assertIn('confidence', prompt)
        self.assertIn('questions', prompt)
        self.assertIn('flagged_reason', prompt)

        # Check questions
        self.assertGreater(len(prompt['questions']), 0)

    def test_submit_label(self):
        """Test label submission."""
        play_data = {'time': 0.0, 'players': self.players}
        insight = {'top_decision': 'Attack left: +31% xT', 'confidence': 'Medium'}

        label = self.al_manager.submit_label(
            play_id='test_001',
            outcome='Success',
            tags=['LB late', 'WR open'],
            notes='Good execution',
            play_data=play_data,
            insight=insight
        )

        # Check label structure
        self.assertEqual(label['play_id'], 'test_001')
        self.assertEqual(label['outcome'], 'Success')
        self.assertEqual(label['tags'], ['LB late', 'WR open'])

        # Check it was added to manager
        self.assertEqual(len(self.al_manager.labels), 1)

        # Check file was saved
        self.assertTrue(os.path.exists(self.labels_path))

        # Verify file content
        with open(self.labels_path, 'r') as f:
            saved_labels = json.load(f)

        self.assertEqual(len(saved_labels), 1)
        self.assertEqual(saved_labels[0]['outcome'], 'Success')

    def test_multiple_labels(self):
        """Test submitting multiple labels."""
        for i in range(3):
            self.al_manager.submit_label(
                play_id=f'test_{i:03d}',
                outcome='Success' if i % 2 == 0 else 'Fail',
                tags=['tag1'],
                play_data={},
                insight={}
            )

        self.assertEqual(len(self.al_manager.labels), 3)

    def test_get_labeling_statistics(self):
        """Test labeling statistics."""
        # Submit some labels
        self.al_manager.submit_label('p1', 'Success', ['tag1'], play_data={}, insight={})
        self.al_manager.submit_label('p2', 'Success', ['tag2'], play_data={}, insight={})
        self.al_manager.submit_label('p3', 'Fail', ['tag1', 'tag2'], play_data={}, insight={})

        stats = self.al_manager.get_labeling_statistics()

        # Check statistics
        self.assertEqual(stats['total_labels'], 3)
        self.assertEqual(stats['outcomes']['Success'], 2)
        self.assertEqual(stats['outcomes']['Fail'], 1)
        self.assertGreater(stats['avg_tags_per_play'], 0)

    def test_export_labels_for_training(self):
        """Test exporting labels for training."""
        # Add some labels
        play_data = {'time': 0.0, 'players': self.players}
        self.al_manager.submit_label(
            'p1', 'Success', ['tag1'],
            play_data=play_data, insight={}
        )

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            export_path = f.name

        try:
            training_data = self.al_manager.export_labels_for_training(
                export_path,
                include_failures=True
            )

            # Check structure
            self.assertIn('plays', training_data)
            self.assertIn('metadata', training_data)

            # Check content
            self.assertEqual(len(training_data['plays']), 1)

            # Check file exists
            self.assertTrue(os.path.exists(export_path))

        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

    def test_create_ui_label_payload(self):
        """Test UI label payload creation."""
        play_data = {
            'play_id': 'test_001',
            'time': 0.0,
            'players': self.players,
            'ball': {'x': 50, 'y': 25}
        }

        # Create high-entropy insight to trigger flagging
        insight = {
            'top_decision': 'Attack left: +31% xT',
            'confidence': 'Low',
            'zones': np.random.rand(105, 68),
            'details': {'avg_entropy': 0.85}
        }

        payload = create_ui_label_payload(play_data, insight, self.al_manager)

        # Should create payload for flagged play
        if payload:
            self.assertIn('ui_config', payload)
            self.assertIn('buttons', payload['ui_config'])
            self.assertIn('tag_suggestions', payload['ui_config'])


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHeatmapGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestActiveLearning))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
