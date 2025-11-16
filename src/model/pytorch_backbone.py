"""
PyTorch backbone model for FieldSense AI v3.0
Equivalent to ONNX model but trainable with LoRA
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List


class BackboneConvNet(nn.Module):
    """
    PyTorch implementation of the xT-Grid backbone.
    Equivalent to the ONNX model for compatibility.
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 16):
        """
        Initialize backbone ConvNet.

        Args:
            in_channels: Number of input feature channels (default: 8)
            hidden_channels: Number of hidden channels (default: 16)
        """
        super().__init__()

        # Conv1: (batch, 8, 105, 68) -> (batch, 16, 105, 68)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )

        # ReLU activation
        self.relu1 = nn.ReLU()

        # Conv2: (batch, 16, 105, 68) -> (batch, 1, 105, 68)
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            padding=0,
            stride=1
        )

        # Sigmoid for probability output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 8, 105, 68)

        Returns:
            xT grid of shape (batch, 105, 68)
        """
        # Conv1 + ReLU
        x = self.conv1(x)
        x = self.relu1(x)

        # Conv2
        x = self.conv2(x)

        # Sigmoid
        x = self.sigmoid(x)

        # Remove channel dimension: (batch, 1, 105, 68) -> (batch, 105, 68)
        x = x.squeeze(1)

        return x

    def load_from_onnx_weights(self, onnx_path: str):
        """
        Load weights from ONNX model (if available).

        Args:
            onnx_path: Path to ONNX model file
        """
        try:
            import onnx
            import onnxruntime as ort

            # Load ONNX model
            onnx_model = onnx.load(onnx_path)

            # Extract weights from initializers
            weights = {}
            for initializer in onnx_model.graph.initializer:
                weights[initializer.name] = onnx.numpy_helper.to_array(initializer)

            # Map ONNX weights to PyTorch parameters
            if 'conv1_weight' in weights:
                self.conv1.weight.data = torch.from_numpy(weights['conv1_weight'])
            if 'conv1_bias' in weights:
                self.conv1.bias.data = torch.from_numpy(weights['conv1_bias'])
            if 'conv2_weight' in weights:
                self.conv2.weight.data = torch.from_numpy(weights['conv2_weight'])
            if 'conv2_bias' in weights:
                self.conv2.bias.data = torch.from_numpy(weights['conv2_bias'])

            print(f"Loaded weights from ONNX model: {onnx_path}")

        except Exception as e:
            print(f"Warning: Could not load ONNX weights: {e}")
            print("Using random initialization instead")


def preprocess_frames_torch(frames: List[Dict[str, Any]]) -> torch.Tensor:
    """
    Convert normalized frames to PyTorch tensor format.

    Args:
        frames: List of frame dictionaries from normalized data

    Returns:
        Torch tensor of shape [batch_size, 8, 105, 68]
    """
    batch = []

    for frame in frames:
        # Create feature maps: 8 channels x 105 x 68
        feature_map = np.zeros((8, 105, 68), dtype=np.float32)

        for player in frame['players']:
            x = int(np.clip(player['x'], 0, 104))
            y = int(np.clip(player['y'], 0, 67))

            # Player density
            feature_map[0, x, y] += 1.0

            # Velocity components
            vel = player['vel']
            dir_rad = np.radians(player['dir'])
            feature_map[1, x, y] = vel * np.cos(dir_rad)  # velocity_x
            feature_map[2, x, y] = vel * np.sin(dir_rad)  # velocity_y

            # Acceleration
            feature_map[3, x, y] = player['acc']

            # Possession
            if player['possession']:
                feature_map[4, x, y] = 1.0

            # Role encoding
            role = player.get('role', 'Unknown')
            if 'Defend' in role or 'QB' in role:
                feature_map[5, x, y] = 1.0
            elif 'Mid' in role or 'RB' in role:
                feature_map[6, x, y] = 1.0
            elif 'Forward' in role or 'WR' in role or 'TE' in role:
                feature_map[7, x, y] = 1.0

        batch.append(feature_map)

    return torch.from_numpy(np.array(batch, dtype=np.float32))
