"""
Unit tests for data ingestion and normalization modules.
"""

import unittest
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import (
    load_data,
    load_statsbomb_json,
    load_wyscout_csv,
    load_custom_csv
)
from src.data.normalize import (
    normalize_dataframe,
    normalize_to_json,
    scale_coordinates,
    auto_detect_roles,
    impute_missing_values
)


class TestDataIngestion(unittest.TestCase):
    """Test data ingestion functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.base_dir = Path(__file__).parent.parent
        cls.sample_dir = cls.base_dir / "assets" / "sample_data"

    def test_load_statsbomb_json(self):
        """Test STATSBomb JSON loading."""
        file_path = self.sample_dir / "statsbomb_sample.json"
        df = load_statsbomb_json(str(file_path))

        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Check required columns exist
        required_cols = ['match_id', 'event_id', 'x', 'y', 'velocity',
                        'direction', 'acceleration', 'possession', 'role']
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_load_wyscout_csv(self):
        """Test Wyscout CSV loading."""
        file_path = self.sample_dir / "wyscout_sample.csv"
        df = load_wyscout_csv(str(file_path))

        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Check required columns exist
        required_cols = ['x', 'y', 'velocity', 'direction', 'acceleration']
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_load_custom_csv(self):
        """Test Custom CSV loading."""
        file_path = self.sample_dir / "custom_sample.csv"
        df = load_custom_csv(str(file_path))

        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Check required columns exist
        required_cols = ['x', 'y', 'velocity', 'direction', 'acceleration']
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_auto_load_detection(self):
        """Test automatic format detection."""
        # Test JSON
        json_file = self.sample_dir / "statsbomb_sample.json"
        df_json = load_data(str(json_file))
        self.assertIsInstance(df_json, pd.DataFrame)

        # Test CSV
        csv_file = self.sample_dir / "custom_sample.csv"
        df_csv = load_data(str(csv_file))
        self.assertIsInstance(df_csv, pd.DataFrame)


class TestDataNormalization(unittest.TestCase):
    """Test data normalization functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.base_dir = Path(__file__).parent.parent
        cls.sample_dir = cls.base_dir / "assets" / "sample_data"

    def test_scale_coordinates(self):
        """Test coordinate scaling."""
        x, y = scale_coordinates(60.0, 40.0, source_length=120.0, source_width=80.0)

        # Should scale to half (105m / 120m * 60 = 52.5)
        self.assertAlmostEqual(x, 52.5, places=1)
        self.assertAlmostEqual(y, 34.0, places=1)

    def test_impute_missing_values(self):
        """Test missing value imputation."""
        # Create DataFrame with NaNs
        df = pd.DataFrame({
            'x': [50.0, np.nan, 60.0],
            'y': [30.0, 35.0, np.nan],
            'velocity': [5.0, np.nan, 6.0],
            'direction': [45.0, np.nan, 90.0],
            'acceleration': [1.0, np.nan, 2.0],
            'possession': [True, np.nan, False],
            'role': ['QB', np.nan, 'WR']
        })

        df_imputed = impute_missing_values(df)

        # Check no NaNs remain in key columns
        self.assertFalse(df_imputed['velocity'].isna().any())
        self.assertFalse(df_imputed['direction'].isna().any())
        self.assertFalse(df_imputed['acceleration'].isna().any())

        # Check imputed values are reasonable
        self.assertEqual(df_imputed['velocity'].iloc[1], 0.0)

    def test_auto_detect_roles(self):
        """Test role auto-detection."""
        # Create sample data with distinct position clusters
        df = pd.DataFrame({
            'x': [20, 25, 50, 55, 80, 85],  # Three clusters
            'y': [30, 35, 30, 35, 30, 35]
        })

        roles = auto_detect_roles(df, n_clusters=3)

        # Should return Series of same length
        self.assertEqual(len(roles), len(df))

        # Should contain role names
        unique_roles = roles.unique()
        self.assertGreater(len(unique_roles), 0)

    def test_normalize_dataframe(self):
        """Test DataFrame normalization to unified schema."""
        # Load sample data
        file_path = self.sample_dir / "statsbomb_sample.json"
        df = load_statsbomb_json(str(file_path))

        # Normalize
        frames = normalize_dataframe(df)

        # Check output structure
        self.assertIsInstance(frames, list)
        self.assertGreater(len(frames), 0)

        # Check frame structure
        frame = frames[0]
        self.assertIn('time', frame)
        self.assertIn('players', frame)
        self.assertIn('ball', frame)

        # Check players structure
        self.assertIsInstance(frame['players'], list)
        if len(frame['players']) > 0:
            player = frame['players'][0]
            required_keys = ['id', 'x', 'y', 'vel', 'dir', 'acc', 'role', 'possession']
            for key in required_keys:
                self.assertIn(key, player, f"Missing key: {key}")

            # Check no NaNs in player data
            self.assertFalse(np.isnan(player['x']))
            self.assertFalse(np.isnan(player['y']))
            self.assertFalse(np.isnan(player['vel']))

    def test_normalize_to_json(self):
        """Test normalization to JSON format."""
        # Load sample data
        file_path = self.sample_dir / "custom_sample.csv"
        df = load_custom_csv(str(file_path))

        # Normalize
        output = normalize_to_json(df)

        # Check output structure
        self.assertIsInstance(output, dict)
        self.assertIn('frames', output)
        self.assertIsInstance(output['frames'], list)
        self.assertGreater(len(output['frames']), 0)

    def test_normalized_output_shape(self):
        """Test that normalized output has correct shape and no NaNs."""
        for filename in ['statsbomb_sample.json', 'wyscout_sample.csv', 'custom_sample.csv']:
            with self.subTest(filename=filename):
                file_path = self.sample_dir / filename
                df = load_data(str(file_path))
                frames = normalize_dataframe(df)

                # Assert is list of dicts
                self.assertIsInstance(frames, list)
                for frame in frames:
                    self.assertIsInstance(frame, dict)

                    # Check all player values are valid
                    for player in frame['players']:
                        self.assertFalse(np.isnan(player['x']),
                                       f"NaN in x for {filename}")
                        self.assertFalse(np.isnan(player['y']),
                                       f"NaN in y for {filename}")
                        self.assertFalse(np.isnan(player['vel']),
                                       f"NaN in vel for {filename}")

    def test_coordinate_clipping(self):
        """Test that coordinates are clipped to field boundaries."""
        # Create data with out-of-bounds coordinates
        df = pd.DataFrame({
            'x': [-10, 50, 150],  # Out of bounds on both ends
            'y': [-5, 34, 100],
            'velocity': [0, 0, 0],
            'direction': [0, 0, 0],
            'acceleration': [0, 0, 0],
            'possession': [False, False, False],
            'role': ['Unknown', 'Unknown', 'Unknown'],
            'player_id': [1, 2, 3],
            'timestamp': [0.0, 0.0, 0.0]
        })

        frames = normalize_dataframe(df)

        # Check all coordinates are within bounds
        for frame in frames:
            for player in frame['players']:
                self.assertGreaterEqual(player['x'], 0)
                self.assertLessEqual(player['x'], 105)
                self.assertGreaterEqual(player['y'], 0)
                self.assertLessEqual(player['y'], 68)


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.base_dir = Path(__file__).parent.parent
        cls.sample_dir = cls.base_dir / "assets" / "sample_data"

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from load to normalized output."""
        for filename in ['statsbomb_sample.json', 'wyscout_sample.csv', 'custom_sample.csv']:
            with self.subTest(filename=filename):
                file_path = self.sample_dir / filename

                # Load
                df = load_data(str(file_path))
                self.assertGreater(len(df), 0)

                # Normalize
                output = normalize_to_json(df)
                self.assertIn('frames', output)
                self.assertGreater(len(output['frames']), 0)

                # Verify no NaNs
                for frame in output['frames']:
                    for player in frame['players']:
                        for key in ['x', 'y', 'vel', 'dir', 'acc']:
                            value = player[key]
                            self.assertFalse(
                                np.isnan(value),
                                f"NaN found in {key} for {filename}"
                            )


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataIngestion))
    suite.addTests(loader.loadTestsFromTestCase(TestDataNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
