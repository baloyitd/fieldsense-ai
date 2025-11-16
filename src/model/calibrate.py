"""
Team-specific calibration engine for FieldSense AI v3.0
Fine-tunes LoRA adapter on user plays in <42 seconds
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from .adapter import BackboneWithAdapter, create_adapter_model
from .pytorch_backbone import preprocess_frames_torch


class PlayDataset(Dataset):
    """Dataset for user plays."""

    def __init__(self, frames: List[Dict[str, Any]], labels: Optional[List[np.ndarray]] = None):
        """
        Initialize dataset.

        Args:
            frames: List of normalized frame dictionaries
            labels: Optional list of target xT grids (if None, uses self-supervised)
        """
        self.frames = frames
        self.labels = labels

        # If no labels, create synthetic targets based on possession zones
        if self.labels is None:
            self.labels = self._create_synthetic_targets()

    def _create_synthetic_targets(self) -> List[np.ndarray]:
        """Create synthetic xT targets from play data."""
        targets = []

        for frame in self.frames:
            # Create target grid (105x68)
            target = np.zeros((105, 68), dtype=np.float32)

            # Higher xT near ball and attacking players
            for player in frame['players']:
                x = int(np.clip(player['x'], 0, 104))
                y = int(np.clip(player['y'], 0, 67))

                # Base threat from player position
                threat = 0.5

                # Higher threat if player has possession
                if player['possession']:
                    threat = 0.9

                # Higher threat for offensive roles moving forward
                role = player.get('role', '')
                if 'Forward' in role or 'WR' in role or 'TE' in role:
                    threat += 0.1

                # Spread threat with Gaussian kernel
                for dx in range(-10, 11):
                    for dy in range(-5, 6):
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < 105 and 0 <= ny < 68:
                            dist = np.sqrt(dx**2 + dy**2)
                            gaussian = np.exp(-dist**2 / 20.0)
                            target[nx, ny] = max(target[nx, ny], threat * gaussian)

            targets.append(target)

        return targets

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Preprocess single frame
        frame_tensor = preprocess_frames_torch([self.frames[idx]])
        target_tensor = torch.from_numpy(self.labels[idx]).float()

        return frame_tensor.squeeze(0), target_tensor


def calibrate_model(
    model: BackboneWithAdapter,
    user_plays: List[Dict[str, Any]],
    output_path: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    train_split: float = 0.8,
    use_bf16: bool = True,
    max_time: float = 42.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Calibrate model on user plays with LoRA fine-tuning.

    Args:
        model: BackboneWithAdapter model
        user_plays: List of normalized frame dictionaries
        output_path: Path to save adapter weights
        epochs: Number of training epochs (default: 5)
        batch_size: Batch size (default: 8)
        lr: Learning rate (default: 1e-4)
        train_split: Train/val split ratio (default: 0.8)
        use_bf16: Use BF16 mixed precision (default: True)
        max_time: Maximum calibration time in seconds (default: 42.0)
        verbose: Print progress (default: True)

    Returns:
        Dictionary with calibration metrics
    """
    start_time = time.time()

    if verbose:
        print("="*60)
        print("FieldSense AI v3.0 - Team Calibration")
        print("="*60)
        print(f"User plays: {len(user_plays)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Split data
    n_train = int(len(user_plays) * train_split)
    train_frames = user_plays[:n_train]
    val_frames = user_plays[n_train:]

    if verbose:
        print(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    # Create datasets
    train_dataset = PlayDataset(train_frames)
    val_dataset = PlayDataset(val_frames) if len(val_frames) > 0 else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    # Setup optimizer (only LoRA parameters)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01
    )

    # Loss function
    criterion = nn.MSELoss()

    # Mixed precision (BF16 if available, else FP16)
    scaler = None
    use_amp = use_bf16 and torch.cuda.is_available()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_times': []
    }

    model.train()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_time:
            if verbose:
                print(f"\nReached time limit ({max_time}s), stopping early")
            break

        # Training
        train_loss = 0.0
        n_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()

            if use_amp and scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            # Check time limit
            if time.time() - start_time > max_time:
                break

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            model.train()

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_times'].append(epoch_time)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"time={epoch_time:.2f}s")

    # Save adapter
    model.save_adapter(output_path)

    total_time = time.time() - start_time

    # Summary
    results = {
        'total_time': total_time,
        'epochs_completed': len(history['train_loss']),
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else 0,
        'n_train': len(train_frames),
        'n_val': len(val_frames),
        'adapter_path': output_path,
        'history': history
    }

    if verbose:
        print("="*60)
        print(f"Calibration complete in {total_time:.2f}s")
        print(f"Adapter saved to: {output_path}")
        print("="*60)

    return results


def calibrate_from_json(
    json_path: str,
    model_path: str,
    output_path: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Calibrate model from JSON file containing user plays.

    Args:
        json_path: Path to JSON file with normalized plays
        model_path: Path to base ONNX model
        output_path: Path to save adapter
        **kwargs: Additional arguments for calibrate_model

    Returns:
        Calibration results
    """
    # Load user plays
    with open(json_path, 'r') as f:
        data = json.load(f)

    user_plays = data.get('frames', [])

    print(f"Loaded {len(user_plays)} plays from {json_path}")

    # Create model with adapter
    model = create_adapter_model(
        onnx_path=model_path,
        lora_rank=kwargs.get('lora_rank', 4)
    )

    # Calibrate
    results = calibrate_model(
        model=model,
        user_plays=user_plays,
        output_path=output_path,
        **kwargs
    )

    return results


def load_calibrated_model(
    base_model_path: str,
    adapter_path: str,
    lora_rank: int = 4
) -> BackboneWithAdapter:
    """
    Load model with calibrated adapter.

    Args:
        base_model_path: Path to base ONNX model
        adapter_path: Path to adapter weights
        lora_rank: LoRA rank

    Returns:
        Model with loaded adapter
    """
    # Create model
    model = create_adapter_model(
        onnx_path=base_model_path,
        lora_rank=lora_rank
    )

    # Load adapter
    model.load_adapter(adapter_path)

    model.eval()

    return model
