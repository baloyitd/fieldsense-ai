"""
Unit tests for counterfactual simulation and physics.
"""

import unittest
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insights.counterfactual import (
    PhysicsSimulator,
    CounterfactualGenerator,
    Perturbation,
    create_counterfactual_payload,
    FIELD_LENGTH,
    FIELD_WIDTH
)


class TestPhysicsSimulator(unittest.TestCase):
    """Test physics simulation."""

    def setUp(self):
        """Set up test fixtures."""
        self.simulator = PhysicsSimulator(dt=0.1)

    def test_simulator_initialization(self):
        """Test simulator initializes correctly."""
        self.assertEqual(self.simulator.dt, 0.1)
        self.assertEqual(self.simulator.gravity, 9.81)
        self.assertGreater(self.simulator.ball_mass, 0)

    def test_player_path_simulation(self):
        """Test player path simulation."""
        start_pos = np.array([50.0, 30.0])
        path = self.simulator.simulate_player_path(
            start_pos=start_pos,
            velocity=5.0,
            direction=45.0,
            duration=1.0
        )

        # Check path exists
        self.assertGreater(len(path), 0)

        # Check first position matches start
        np.testing.assert_array_almost_equal(path[0], start_pos, decimal=1)

        # Check player moved
        self.assertGreater(np.linalg.norm(path[-1] - path[0]), 0)

    def test_player_path_in_bounds(self):
        """Test player paths stay within field bounds."""
        start_pos = np.array([50.0, 30.0])
        path = self.simulator.simulate_player_path(
            start_pos=start_pos,
            velocity=8.0,
            direction=90.0,
            duration=2.0
        )

        # All positions should be in bounds
        for pos in path:
            self.assertGreaterEqual(pos[0], 0)
            self.assertLessEqual(pos[0], FIELD_LENGTH)
            self.assertGreaterEqual(pos[1], 0)
            self.assertLessEqual(pos[1], FIELD_WIDTH)

    def test_player_delay_perturbation(self):
        """Test delay perturbation."""
        start_pos = np.array([50.0, 30.0])
        pert = Perturbation(player_id='1', delay=0.5)

        path = self.simulator.simulate_player_path(
            start_pos=start_pos,
            velocity=5.0,
            direction=0.0,
            duration=1.0,
            perturbation=pert
        )

        # During delay, position should not change much
        delay_steps = int(0.5 / self.simulator.dt)
        if delay_steps > 0 and delay_steps < len(path):
            dist_during_delay = np.linalg.norm(path[delay_steps] - path[0])
            self.assertLess(dist_during_delay, 1.0)  # Should move very little

    def test_speed_perturbation(self):
        """Test speed perturbation."""
        start_pos = np.array([50.0, 30.0])

        # Normal speed path
        path_normal = self.simulator.simulate_player_path(
            start_pos=start_pos,
            velocity=8.0,
            direction=0.0,
            duration=1.0
        )

        # Slower speed path
        pert = Perturbation(player_id='1', speed_factor=0.5)
        path_slow = self.simulator.simulate_player_path(
            start_pos=start_pos,
            velocity=8.0,
            direction=0.0,
            duration=1.0,
            perturbation=pert
        )

        # Slower path should cover less distance
        dist_normal = np.linalg.norm(path_normal[-1] - path_normal[0])
        dist_slow = np.linalg.norm(path_slow[-1] - path_slow[0])

        self.assertLess(dist_slow, dist_normal)

    def test_angle_perturbation(self):
        """Test angle perturbation."""
        start_pos = np.array([50.0, 34.0])

        # Path at 0 degrees (moving right)
        path_0 = self.simulator.simulate_player_path(
            start_pos=start_pos,
            velocity=5.0,
            direction=0.0,
            duration=1.0
        )

        # Path at 90 degrees (moving up)
        pert = Perturbation(player_id='1', angle_delta=90.0)
        path_90 = self.simulator.simulate_player_path(
            start_pos=start_pos,
            velocity=5.0,
            direction=0.0,
            duration=1.0,
            perturbation=pert
        )

        # Y displacement should be different
        dy_0 = path_0[-1][1] - path_0[0][1]
        dy_90 = path_90[-1][1] - path_90[0][1]

        self.assertNotAlmostEqual(dy_0, dy_90, places=1)

    def test_ball_trajectory(self):
        """Test ball trajectory simulation."""
        start_pos = np.array([50.0, 30.0])
        initial_vel = np.array([10.0, 5.0])

        trajectory = self.simulator.simulate_ball_trajectory(
            start_pos=start_pos,
            initial_velocity=initial_vel,
            duration=1.0
        )

        # Check trajectory exists
        self.assertGreater(len(trajectory), 0)

        # Check ball moved
        self.assertGreater(np.linalg.norm(trajectory[-1] - trajectory[0]), 0)

    def test_ball_decay(self):
        """Test ball velocity decays over time."""
        start_pos = np.array([50.0, 30.0])
        initial_vel = np.array([10.0, 0.0])

        trajectory = self.simulator.simulate_ball_trajectory(
            start_pos=start_pos,
            initial_velocity=initial_vel,
            duration=2.0
        )

        # Approximate velocities from consecutive positions
        if len(trajectory) >= 3:
            # Early velocity
            early_disp = trajectory[1] - trajectory[0]
            early_speed = np.linalg.norm(early_disp) / self.simulator.dt

            # Late velocity
            late_disp = trajectory[-1] - trajectory[-2]
            late_speed = np.linalg.norm(late_disp) / self.simulator.dt

            # Speed should decrease due to decay
            self.assertLess(late_speed, early_speed)

    def test_ball_boundaries(self):
        """Test ball respects field boundaries."""
        start_pos = np.array([100.0, 30.0])
        # Velocity pointing out of bounds
        initial_vel = np.array([20.0, 0.0])

        trajectory = self.simulator.simulate_ball_trajectory(
            start_pos=start_pos,
            initial_velocity=initial_vel,
            duration=1.0
        )

        # All positions should be in bounds
        for pos in trajectory:
            self.assertGreaterEqual(pos[0], 0)
            self.assertLessEqual(pos[0], FIELD_LENGTH)
            self.assertGreaterEqual(pos[1], 0)
            self.assertLessEqual(pos[1], FIELD_WIDTH)


class TestCounterfactualGenerator(unittest.TestCase):
    """Test counterfactual generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = CounterfactualGenerator()

        self.play_data = {
            'time': 0.0,
            'players': [
                {'id': '1', 'x': 50, 'y': 25, 'role': 'QB', 'possession': True,
                 'vel': 5.0, 'dir': 45, 'acc': 1.0},
                {'id': '2', 'x': 65, 'y': 15, 'role': 'WR', 'possession': False,
                 'vel': 8.5, 'dir': 90, 'acc': 1.5},
                {'id': '3', 'x': 70, 'y': 35, 'role': 'Forward', 'possession': False,
                 'vel': 7.0, 'dir': 60, 'acc': 1.2},
            ],
            'ball': {'x': 50, 'y': 25}
        }

    def test_generator_initialization(self):
        """Test generator initializes."""
        self.assertIsNotNone(self.generator.simulator)

    def test_generate_perturbations(self):
        """Test perturbation generation."""
        perts = self.generator.generate_perturbations(self.play_data, n_perturbations=2)

        # Should generate perturbations
        self.assertGreater(len(perts), 0)

        # Perturbations should target offensive players
        for pert in perts:
            self.assertIn(pert.player_id, ['2', '3'])  # WR or Forward

    def test_simulate_counterfactual(self):
        """Test counterfactual simulation."""
        pert = Perturbation(player_id='2', delay=0.5)

        simulated = self.generator.simulate_counterfactual(
            self.play_data,
            pert,
            duration=1.0
        )

        # Should return simulated play
        self.assertIn('players', simulated)
        self.assertIn('ball', simulated)

        # Should have trajectory
        self.assertIn('trajectory', simulated)

    def test_validate_scenario_valid(self):
        """Test validation of valid scenario."""
        simulated = self.play_data.copy()

        is_valid, violation = self.generator.validate_scenario(
            simulated,
            self.play_data
        )

        self.assertTrue(is_valid)
        self.assertIsNone(violation)

    def test_validate_scenario_out_of_bounds(self):
        """Test validation catches out of bounds."""
        simulated = self.play_data.copy()
        simulated['ball']['x'] = -10  # Out of bounds

        is_valid, violation = self.generator.validate_scenario(
            simulated,
            self.play_data
        )

        self.assertFalse(is_valid)
        self.assertIsNotNone(violation)
        self.assertIn('bounds', violation.lower())

    def test_validate_scenario_collision(self):
        """Test validation catches collisions."""
        simulated = self.play_data.copy()
        # Move two players to same position
        simulated['players'][0]['x'] = 50
        simulated['players'][0]['y'] = 25
        simulated['players'][1]['x'] = 50
        simulated['players'][1]['y'] = 25

        is_valid, violation = self.generator.validate_scenario(
            simulated,
            self.play_data
        )

        self.assertFalse(is_valid)
        self.assertIsNotNone(violation)
        self.assertIn('collision', violation.lower())

    def test_compute_delta_xt(self):
        """Test delta xT computation."""
        delta = self.generator.compute_delta_xt(0.5, 0.7)
        self.assertAlmostEqual(delta, 0.2)

        delta_neg = self.generator.compute_delta_xt(0.7, 0.5)
        self.assertAlmostEqual(delta_neg, -0.2)

    def test_compute_cvs(self):
        """Test CVS computation."""
        from src.insights.counterfactual import CounterfactualResult

        results = [
            CounterfactualResult(player='P1', change='test', lift=0.1, valid=True),
            CounterfactualResult(player='P2', change='test', lift=0.2, valid=True),
            CounterfactualResult(player='P3', change='test', lift=0.3, valid=False),
        ]

        cvs = self.generator.compute_cvs(results)
        self.assertAlmostEqual(cvs, 2/3, places=2)

    def test_cvs_all_valid(self):
        """Test CVS with all valid scenarios."""
        from src.insights.counterfactual import CounterfactualResult

        results = [
            CounterfactualResult(player='P1', change='test', lift=0.1, valid=True),
            CounterfactualResult(player='P2', change='test', lift=0.2, valid=True),
            CounterfactualResult(player='P3', change='test', lift=0.3, valid=True),
        ]

        cvs = self.generator.compute_cvs(results)
        self.assertEqual(cvs, 1.0)

    def test_generate_counterfactuals(self):
        """Test full counterfactual generation."""
        results, cvs = self.generator.generate_counterfactuals(
            self.play_data,
            original_xt=0.5,
            n_scenarios=3
        )

        # Should generate results (up to n_scenarios, limited by available offensive players)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

        # CVS should be between 0 and 1
        self.assertGreaterEqual(cvs, 0.0)
        self.assertLessEqual(cvs, 1.0)

        # Results should have required fields
        for result in results:
            self.assertIsNotNone(result.player)
            self.assertIsNotNone(result.change)
            self.assertIsInstance(result.lift, float)
            self.assertIsInstance(result.valid, bool)

    def test_create_counterfactual_payload(self):
        """Test payload creation for UI."""
        payload = create_counterfactual_payload(
            self.play_data,
            original_xt=0.5,
            n_scenarios=3
        )

        # Check payload structure
        self.assertIn('counterfactuals', payload)
        self.assertIn('cvs', payload)
        self.assertIn('n_valid', payload)
        self.assertIn('n_total', payload)

        # Check counts match (may be limited by available offensive players)
        self.assertGreater(payload['n_total'], 0)
        self.assertEqual(len(payload['counterfactuals']), payload['n_total'])


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicsSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestCounterfactualGenerator))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
