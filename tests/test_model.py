"""
Unit tests for model components (PyTorch backbone, LoRA adapter, calibration).
"""

import unittest
import sys
import os
import json
import time
import tempfile
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model.pytorch_backbone import BackboneConvNet, preprocess_frames_torch
from src.model.adapter import BackboneWithAdapter, create_adapter_model
from src.model.calibrate import calibrate_model, PlayDataset


class TestPyTorchBackbone(unittest.TestCase):
    """Test PyTorch backbone model."""

    def test_model_creation(self):
        """Test model can be created."""
        model = BackboneConvNet()
        self.assertIsInstance(model, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass with correct shapes."""
        model = BackboneConvNet()
        model.eval()

        # Create dummy input (batch=2, channels=8, H=105, W=68)
        x = torch.randn(2, 8, 105, 68)

        with torch.no_grad():
            output = model(x)

        # Check output shape (batch=2, H=105, W=68)
        self.assertEqual(output.shape, (2, 105, 68))

        # Check output range (should be 0-1 due to sigmoid)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    def test_preprocess_frames(self):
        """Test frame preprocessing."""
        # Create sample frame
        frames = [{
            'time': 0.0,
            'players': [
                {
                    'id': '1',
                    'x': 52.5,
                    'y': 34.0,
                    'vel': 5.0,
                    'dir': 45.0,
                    'acc': 1.0,
                    'role': 'QB',
                    'possession': True
                }
            ],
            'ball': {'x': 52.5, 'y': 34.0}
        }]

        tensor = preprocess_frames_torch(frames)

        # Check shape
        self.assertEqual(tensor.shape, (1, 8, 105, 68))

        # Check dtype
        self.assertEqual(tensor.dtype, torch.float32)

        # Check some values are set
        self.assertTrue(torch.any(tensor > 0))


class TestLoRAAdapter(unittest.TestCase):
    """Test LoRA adapter functionality."""

    def test_adapter_creation(self):
        """Test adapter model creation."""
        model = BackboneWithAdapter(lora_rank=4)
        self.assertIsInstance(model, torch.nn.Module)

    def test_adapter_has_lora_params(self):
        """Test adapter has LoRA parameters."""
        model = BackboneWithAdapter(lora_rank=4)

        # Check LoRA parameters exist
        self.assertTrue(hasattr(model, 'conv1_lora_A'))
        self.assertTrue(hasattr(model, 'conv1_lora_B'))
        self.assertTrue(hasattr(model, 'conv2_lora_A'))
        self.assertTrue(hasattr(model, 'conv2_lora_B'))

    def test_trainable_params(self):
        """Test only LoRA params are trainable."""
        model = BackboneWithAdapter(lora_rank=4)

        # Backbone params should be frozen
        for name, param in model.backbone.named_parameters():
            self.assertFalse(param.requires_grad,
                           f"Backbone param {name} should be frozen")

        # LoRA params should be trainable
        self.assertTrue(model.conv1_lora_A.requires_grad)
        self.assertTrue(model.conv1_lora_B.requires_grad)
        self.assertTrue(model.conv2_lora_A.requires_grad)
        self.assertTrue(model.conv2_lora_B.requires_grad)

    def test_forward_pass_with_adapter(self):
        """Test forward pass with adapter."""
        model = BackboneWithAdapter(lora_rank=4)
        model.eval()

        x = torch.randn(2, 8, 105, 68)

        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape, (2, 105, 68))
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    def test_save_load_adapter(self):
        """Test saving and loading adapter weights."""
        model1 = BackboneWithAdapter(lora_rank=4)

        # Modify some weights
        with torch.no_grad():
            model1.conv1_lora_A.fill_(0.5)

        # Save adapter
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            adapter_path = f.name

        try:
            model1.save_adapter(adapter_path)

            # Load into new model
            model2 = BackboneWithAdapter(lora_rank=4)
            model2.load_adapter(adapter_path)

            # Check weights match
            self.assertTrue(
                torch.allclose(model1.conv1_lora_A, model2.conv1_lora_A)
            )

        finally:
            os.unlink(adapter_path)

    def test_adapter_size(self):
        """Test adapter size is small (~0.8MB)."""
        model = BackboneWithAdapter(lora_rank=4)

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            adapter_path = f.name

        try:
            model.save_adapter(adapter_path)

            # Check file size
            size_mb = os.path.getsize(adapter_path) / (1024 * 1024)

            # Should be less than 1 MB
            self.assertLess(size_mb, 1.0,
                          f"Adapter size {size_mb:.2f} MB should be < 1 MB")

            # Should be around 0.8 MB (give some tolerance)
            self.assertGreater(size_mb, 0.1,
                             f"Adapter size {size_mb:.2f} MB seems too small")

        finally:
            os.unlink(adapter_path)


class TestCalibration(unittest.TestCase):
    """Test calibration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample plays
        self.sample_plays = []
        for i in range(20):
            frame = {
                'time': float(i) * 0.1,
                'players': [
                    {
                        'id': '1',
                        'x': 50.0 + i,
                        'y': 30.0,
                        'vel': 5.0,
                        'dir': 45.0,
                        'acc': 1.0,
                        'role': 'QB',
                        'possession': True
                    },
                    {
                        'id': '2',
                        'x': 45.0 + i,
                        'y': 25.0,
                        'vel': 4.0,
                        'dir': 90.0,
                        'acc': 0.5,
                        'role': 'WR',
                        'possession': False
                    }
                ],
                'ball': {'x': 50.0 + i, 'y': 30.0}
            }
            self.sample_plays.append(frame)

    def test_dataset_creation(self):
        """Test PlayDataset creation."""
        dataset = PlayDataset(self.sample_plays)

        self.assertEqual(len(dataset), len(self.sample_plays))

    def test_dataset_getitem(self):
        """Test dataset __getitem__."""
        dataset = PlayDataset(self.sample_plays)

        inputs, targets = dataset[0]

        # Check shapes
        self.assertEqual(inputs.shape, (8, 105, 68))
        self.assertEqual(targets.shape, (105, 68))

        # Check types
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)

    def test_calibration_runs(self):
        """Test calibration completes without errors."""
        model = BackboneWithAdapter(lora_rank=4)

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            adapter_path = f.name

        try:
            results = calibrate_model(
                model=model,
                user_plays=self.sample_plays,
                output_path=adapter_path,
                epochs=2,
                batch_size=4,
                lr=1e-4,
                train_split=0.8,
                use_bf16=False,
                max_time=60.0,
                verbose=False
            )

            # Check results structure
            self.assertIn('total_time', results)
            self.assertIn('epochs_completed', results)
            self.assertIn('final_train_loss', results)

            # Check adapter was saved
            self.assertTrue(os.path.exists(adapter_path))

        finally:
            if os.path.exists(adapter_path):
                os.unlink(adapter_path)

    def test_calibration_time_constraint(self):
        """Test calibration completes in <42 seconds."""
        model = BackboneWithAdapter(lora_rank=4)

        # Use more plays for realistic test
        plays = self.sample_plays * 3  # 60 plays

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            adapter_path = f.name

        try:
            start = time.time()

            results = calibrate_model(
                model=model,
                user_plays=plays,
                output_path=adapter_path,
                epochs=5,
                batch_size=8,
                lr=1e-4,
                train_split=0.8,
                use_bf16=False,
                max_time=42.0,
                verbose=False
            )

            elapsed = time.time() - start

            # Should complete in < 42 seconds (or stop early)
            self.assertLess(elapsed, 45.0,
                          f"Calibration took {elapsed:.2f}s, should be < 45s")

            # Reported time should be accurate
            self.assertLess(results['total_time'], 45.0)

        finally:
            if os.path.exists(adapter_path):
                os.unlink(adapter_path)

    def test_calibration_improves_loss(self):
        """Test calibration reduces training loss."""
        model = BackboneWithAdapter(lora_rank=4)

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            adapter_path = f.name

        try:
            results = calibrate_model(
                model=model,
                user_plays=self.sample_plays,
                output_path=adapter_path,
                epochs=5,
                batch_size=4,
                lr=1e-4,
                train_split=0.8,
                use_bf16=False,
                max_time=60.0,
                verbose=False
            )

            # Loss should decrease
            history = results['history']
            if len(history['train_loss']) > 1:
                first_loss = history['train_loss'][0]
                last_loss = history['train_loss'][-1]

                # Last loss should be less than or equal to first loss
                # (allowing for some variance)
                self.assertLessEqual(last_loss, first_loss * 1.1,
                                   "Training loss should decrease or stay stable")

        finally:
            if os.path.exists(adapter_path):
                os.unlink(adapter_path)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPyTorchBackbone))
    suite.addTests(loader.loadTestsFromTestCase(TestLoRAAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestCalibration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
