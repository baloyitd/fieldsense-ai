"""
Backbone model inference module for FieldSense AI v3.0
Loads ONNX model and performs xT grid inference
"""

import numpy as np
import onnxruntime as ort
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class BackboneModel:
    """
    xT-Grid + Temporal ConvNet backbone model for field threat prediction.
    """

    def __init__(self, model_path: str):
        """
        Initialize the backbone model.

        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.load_model()

    def load_model(self):
        """Load ONNX model into inference session."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Create inference session
        self.session = ort.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider']
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"Loaded model from {self.model_path}")
        print(f"Input: {self.input_name}, shape: {self.session.get_inputs()[0].shape}")
        print(f"Output: {self.output_name}, shape: {self.session.get_outputs()[0].shape}")

    def preprocess_frames(self, frames: List[Dict[str, Any]]) -> np.ndarray:
        """
        Convert normalized frames to model input format.

        Args:
            frames: List of frame dictionaries from normalized data

        Returns:
            NumPy array of shape [batch_size, features, 105, 68]
        """
        batch = []

        for frame in frames:
            # Create feature maps: 105x68 grid with 8 feature channels
            # Features: player_density, velocity_x, velocity_y, acceleration,
            #           possession, role_defensive, role_midfield, role_offensive
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

                # Role encoding (one-hot-ish)
                role = player.get('role', 'Unknown')
                if 'Defend' in role or 'QB' in role:
                    feature_map[5, x, y] = 1.0
                elif 'Mid' in role or 'RB' in role:
                    feature_map[6, x, y] = 1.0
                elif 'Forward' in role or 'WR' in role or 'TE' in role:
                    feature_map[7, x, y] = 1.0

            batch.append(feature_map)

        return np.array(batch, dtype=np.float32)

    def infer(self, frames: List[Dict[str, Any]]) -> np.ndarray:
        """
        Run inference on normalized frames.

        Args:
            frames: List of frame dictionaries

        Returns:
            xT grid predictions of shape [batch_size, 105, 68]
        """
        # Preprocess frames to model input format
        input_data = self.preprocess_frames(frames)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )

        return outputs[0]

    def infer_from_json(self, normalized_data: Dict[str, Any]) -> np.ndarray:
        """
        Run inference on normalized JSON data.

        Args:
            normalized_data: Dictionary with 'frames' key

        Returns:
            xT grid predictions
        """
        frames = normalized_data.get('frames', [])
        return self.infer(frames)


def save_heatmap(
    xt_grid: np.ndarray,
    output_path: str,
    title: str = "xT Heatmap",
    frame_idx: int = 0
):
    """
    Save xT grid as heatmap PNG.

    Args:
        xt_grid: xT predictions of shape [batch, 105, 68] or [105, 68]
        output_path: Path to save PNG file
        title: Plot title
        frame_idx: Frame index to visualize if batch
    """
    # Get single frame if batch
    if len(xt_grid.shape) == 3:
        grid = xt_grid[frame_idx]
    else:
        grid = xt_grid

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap with green-gold gradient
    im = ax.imshow(
        grid.T,  # Transpose for correct orientation
        cmap='YlGn',  # Yellow-Green colormap
        aspect='auto',
        origin='lower',
        extent=[0, 105, 0, 68],
        vmin=0,
        vmax=1
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Expected Threat (xT)', rotation=270, labelpad=20)

    # Labels and title
    ax.set_xlabel('Field Length (m)')
    ax.set_ylabel('Field Width (m)')
    ax.set_title(title)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to {output_path}")


def generate_heatmap_from_file(
    data_file: str,
    model_path: str,
    output_path: str,
    title: Optional[str] = None
):
    """
    End-to-end: Load data, normalize, infer, save heatmap.

    Args:
        data_file: Path to input data file
        model_path: Path to ONNX model
        output_path: Path to save heatmap PNG
        title: Optional plot title
    """
    from ..data.normalize import load_and_normalize

    # Load and normalize data
    print(f"Loading data from {data_file}...")
    normalized_data = load_and_normalize(data_file)
    print(f"Loaded {len(normalized_data['frames'])} frames")

    # Load model and infer
    print(f"Loading model from {model_path}...")
    model = BackboneModel(model_path)

    print("Running inference...")
    xt_grid = model.infer_from_json(normalized_data)
    print(f"Generated xT grid: shape={xt_grid.shape}")

    # Save heatmap
    if title is None:
        title = f"xT Heatmap - {Path(data_file).stem}"

    save_heatmap(xt_grid, output_path, title)

    return xt_grid
