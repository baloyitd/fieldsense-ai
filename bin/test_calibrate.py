#!/usr/bin/env python3
"""
Test script for LoRA calibration in FieldSense AI v3.0.
Simulates team-specific adaptation in <42 seconds.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import load_data
from src.data.normalize import normalize_to_json
from src.model.adapter import create_adapter_model
from src.model.calibrate import calibrate_model
from src.model.pytorch_backbone import preprocess_frames_torch
from src.model.backbone import save_heatmap
import torch


def generate_synthetic_user_plays(n_plays: int = 50) -> list:
    """
    Generate synthetic user plays for testing.
    Simulates team-specific patterns (e.g., left-side attack preference).

    Args:
        n_plays: Number of plays to generate

    Returns:
        List of normalized frame dictionaries
    """
    plays = []

    for play_idx in range(n_plays):
        frame = {
            'time': float(play_idx) * 0.1,
            'players': [],
            'ball': {'x': 52.5, 'y': 34.0}
        }

        # Create team-specific pattern: prefer left side (y < 34)
        # This simulates a team that attacks more on the left
        n_players = np.random.randint(8, 12)

        for player_idx in range(n_players):
            # Bias towards left side (lower y values)
            if np.random.rand() < 0.7:  # 70% on left side
                y = np.random.uniform(10, 34)
            else:
                y = np.random.uniform(34, 58)

            x = np.random.uniform(30, 90)

            # Random player attributes
            player = {
                'id': str(player_idx),
                'x': float(x),
                'y': float(y),
                'vel': float(np.random.uniform(2, 8)),
                'dir': float(np.random.uniform(0, 360)),
                'acc': float(np.random.uniform(0, 3)),
                'role': np.random.choice(['QB', 'WR', 'RB', 'TE']),
                'possession': player_idx == 0  # First player has possession
            }

            frame['players'].append(player)

            # Update ball position to possession player
            if player['possession']:
                frame['ball']['x'] = player['x']
                frame['ball']['y'] = player['y']

        plays.append(frame)

    return plays


def compute_heatmap_diff(heatmap1: np.ndarray, heatmap2: np.ndarray) -> dict:
    """
    Compute difference metrics between two heatmaps.

    Args:
        heatmap1: First heatmap (H, W)
        heatmap2: Second heatmap (H, W)

    Returns:
        Dictionary with difference metrics
    """
    diff = heatmap2 - heatmap1

    # Overall metrics
    mean_diff = np.mean(diff)
    max_diff = np.max(np.abs(diff))
    mse = np.mean(diff**2)

    # Zone-specific metrics (left vs right)
    left_zone = diff[:, :34]  # Left half
    right_zone = diff[:, 34:]  # Right half

    metrics = {
        'mean_diff': float(mean_diff),
        'max_diff': float(max_diff),
        'mse': float(mse),
        'left_zone_mean': float(np.mean(left_zone)),
        'right_zone_mean': float(np.mean(right_zone)),
        'left_zone_max': float(np.max(left_zone)),
        'right_zone_max': float(np.max(right_zone))
    }

    return metrics


def main():
    """Main test function."""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "assets" / "backbone.onnx"
    output_dir = base_dir / "outputs"
    adapter_path = output_dir / "my_team_adapter.bin"

    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("FieldSense AI v3.0 - LoRA Calibration Test")
    print("="*60)

    # Step 1: Generate synthetic user plays
    print("\n[Step 1/5] Generating synthetic user plays...")
    n_plays = 50
    user_plays = generate_synthetic_user_plays(n_plays)
    print(f"Generated {len(user_plays)} plays with left-side attack bias")

    # Save user plays
    user_plays_path = output_dir / "user_plays.json"
    with open(user_plays_path, 'w') as f:
        json.dump({'frames': user_plays}, f, indent=2)
    print(f"Saved to: {user_plays_path}")

    # Step 2: Create base model and get baseline prediction
    print("\n[Step 2/5] Creating base model and baseline prediction...")
    base_model = create_adapter_model(
        onnx_path=str(model_path),
        lora_rank=4
    )

    # Get baseline prediction on first play
    test_play = [user_plays[0]]
    test_input = preprocess_frames_torch(test_play)

    base_model.eval()
    with torch.no_grad():
        baseline_output = base_model(test_input).numpy()

    print(f"Baseline xT range: [{baseline_output.min():.4f}, {baseline_output.max():.4f}]")

    # Save baseline heatmap
    baseline_heatmap_path = output_dir / "before_calibration_heatmap.png"
    save_heatmap(
        baseline_output,
        str(baseline_heatmap_path),
        title="Before Calibration - Baseline xT"
    )

    # Step 3: Calibrate model
    print("\n[Step 3/5] Calibrating model on user plays...")
    print(f"Target: <42 seconds")

    calibration_start = time.time()

    results = calibrate_model(
        model=base_model,
        user_plays=user_plays,
        output_path=str(adapter_path),
        epochs=5,
        batch_size=8,
        lr=1e-4,
        train_split=0.8,
        use_bf16=False,  # CPU mode
        max_time=42.0,
        verbose=True
    )

    calibration_time = results['total_time']

    print(f"\nCalibration completed in {calibration_time:.2f}s")

    # Check time constraint
    if calibration_time < 42.0:
        print(f"✓ Time constraint met (<42s)")
    else:
        print(f"✗ Time constraint exceeded (>42s)")

    # Step 4: Load calibrated model and re-infer
    print("\n[Step 4/5] Testing calibrated model...")

    # Reload model with adapter
    calibrated_model = create_adapter_model(
        onnx_path=str(model_path),
        lora_rank=4
    )
    calibrated_model.load_adapter(str(adapter_path))
    calibrated_model.eval()

    # Re-infer same play
    with torch.no_grad():
        calibrated_output = calibrated_model(test_input).numpy()

    print(f"Calibrated xT range: [{calibrated_output.min():.4f}, {calibrated_output.max():.4f}]")

    # Save calibrated heatmap
    calibrated_heatmap_path = output_dir / "after_calibration_heatmap.png"
    save_heatmap(
        calibrated_output,
        str(calibrated_heatmap_path),
        title="After Calibration - Team-Adapted xT"
    )

    # Step 5: Compare heatmaps
    print("\n[Step 5/5] Comparing heatmaps...")

    diff_metrics = compute_heatmap_diff(
        baseline_output[0],
        calibrated_output[0]
    )

    print(f"Difference metrics:")
    print(f"  Overall mean diff: {diff_metrics['mean_diff']:.4f}")
    print(f"  Overall max diff: {diff_metrics['max_diff']:.4f}")
    print(f"  MSE: {diff_metrics['mse']:.6f}")
    print(f"  Left zone mean diff: {diff_metrics['left_zone_mean']:.4f}")
    print(f"  Right zone mean diff: {diff_metrics['right_zone_mean']:.4f}")

    # Create difference heatmap
    diff_heatmap = calibrated_output[0] - baseline_output[0]
    diff_heatmap_path = output_dir / "calibration_diff_heatmap.png"

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        diff_heatmap.T,
        cmap='RdBu_r',  # Red-Blue diverging
        aspect='auto',
        origin='lower',
        extent=[0, 105, 0, 68],
        vmin=-0.2,
        vmax=0.2
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('xT Difference (Calibrated - Baseline)', rotation=270, labelpad=20)
    ax.set_xlabel('Field Length (m)')
    ax.set_ylabel('Field Width (m)')
    ax.set_title('Calibration Impact: xT Difference Heatmap')
    ax.axhline(y=34, color='white', linestyle='--', alpha=0.5, label='Field Center')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(diff_heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Difference heatmap saved to: {diff_heatmap_path}")

    # Check adapter size
    adapter_size_mb = adapter_path.stat().st_size / (1024 * 1024)
    print(f"\nAdapter size: {adapter_size_mb:.2f} MB")

    if adapter_size_mb < 1.0:
        print(f"✓ Adapter size constraint met (<1 MB)")
    else:
        print(f"⚠ Adapter size: {adapter_size_mb:.2f} MB")

    # Verification assertions
    print("\n" + "="*60)
    print("Verification Results")
    print("="*60)

    passed_checks = []
    failed_checks = []

    # Check 1: Calibration time
    if calibration_time < 42.0:
        passed_checks.append(f"Calibration time: {calibration_time:.2f}s < 42s")
    else:
        failed_checks.append(f"Calibration time: {calibration_time:.2f}s >= 42s")

    # Check 2: Adapter size
    if adapter_size_mb < 1.0:
        passed_checks.append(f"Adapter size: {adapter_size_mb:.2f} MB < 1 MB")
    else:
        failed_checks.append(f"Adapter size: {adapter_size_mb:.2f} MB >= 1 MB")

    # Check 3: Team zone improvement (left side should have higher xT after calibration)
    if diff_metrics['left_zone_mean'] > 0.01:  # At least 0.01 improvement on left
        passed_checks.append(f"Left zone improvement: {diff_metrics['left_zone_mean']:.4f} > 0.01")
    else:
        failed_checks.append(f"Left zone improvement: {diff_metrics['left_zone_mean']:.4f} <= 0.01")

    # Check 4: Maximum difference > 0.05
    if diff_metrics['max_diff'] > 0.05:
        passed_checks.append(f"Max xT change: {diff_metrics['max_diff']:.4f} > 0.05")
    else:
        failed_checks.append(f"Max xT change: {diff_metrics['max_diff']:.4f} <= 0.05")

    # Check 5: Validation loss stable (no catastrophic forgetting)
    val_loss_stable = results['final_val_loss'] < 0.1  # Arbitrary threshold
    if val_loss_stable:
        passed_checks.append(f"Val loss stable: {results['final_val_loss']:.4f} < 0.1")
    else:
        failed_checks.append(f"Val loss unstable: {results['final_val_loss']:.4f} >= 0.1")

    print(f"\nPassed: {len(passed_checks)}/{len(passed_checks) + len(failed_checks)}")
    for check in passed_checks:
        print(f"  ✓ {check}")

    if failed_checks:
        print(f"\nFailed:")
        for check in failed_checks:
            print(f"  ✗ {check}")

    print("\nOutputs:")
    print(f"  - User plays: {user_plays_path}")
    print(f"  - Adapter: {adapter_path}")
    print(f"  - Baseline heatmap: {baseline_heatmap_path}")
    print(f"  - Calibrated heatmap: {calibrated_heatmap_path}")
    print(f"  - Difference heatmap: {diff_heatmap_path}")

    return 0 if len(failed_checks) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
