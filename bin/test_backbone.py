#!/usr/bin/env python3
"""
Test script for FieldSense AI v3.0 backbone model.
Tests data pipeline and inference on all sample formats.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import load_data
from src.data.normalize import normalize_to_json
from src.model.backbone import BackboneModel, save_heatmap


def test_single_file(data_file: str, model: BackboneModel, output_dir: Path):
    """
    Test single data file through full pipeline.

    Args:
        data_file: Path to data file
        model: Loaded backbone model
        output_dir: Directory to save outputs
    """
    file_name = Path(data_file).stem
    print(f"\n{'='*60}")
    print(f"Testing: {file_name}")
    print(f"{'='*60}")

    start_time = time.time()

    # Step 1: Load data
    print(f"[1/4] Loading data from {data_file}...")
    df = load_data(data_file)
    print(f"  Loaded {len(df)} rows")

    # Step 2: Normalize
    print(f"[2/4] Normalizing data...")
    normalized_data = normalize_to_json(df)
    n_frames = len(normalized_data['frames'])
    print(f"  Created {n_frames} frames")

    # Save sample normalized JSON
    sample_json_path = output_dir / f"{file_name}_normalized.json"
    with open(sample_json_path, 'w') as f:
        # Save just first frame as sample
        sample = {
            'frames': normalized_data['frames'][:1]
        }
        json.dump(sample, f, indent=2)
    print(f"  Sample normalized JSON: {sample_json_path}")

    # Step 3: Run inference
    print(f"[3/4] Running backbone inference...")
    xt_grid = model.infer_from_json(normalized_data)
    print(f"  Output shape: {xt_grid.shape}")
    print(f"  xT range: [{xt_grid.min():.4f}, {xt_grid.max():.4f}]")

    # Step 4: Save heatmap
    print(f"[4/4] Generating heatmap...")
    heatmap_path = output_dir / f"{file_name}_heatmap.png"
    save_heatmap(xt_grid, str(heatmap_path), title=f"xT Heatmap - {file_name}")

    elapsed = time.time() - start_time
    print(f"\n  ✓ Completed in {elapsed:.2f}s ({elapsed/n_frames:.3f}s per frame)")
    print(f"  Output: {heatmap_path}")

    return normalized_data, xt_grid


def main():
    """Main test function."""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    sample_dir = base_dir / "assets" / "sample_data"
    model_path = base_dir / "assets" / "backbone.onnx"
    output_dir = base_dir / "outputs"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("FieldSense AI v3.0 - Backbone Model Test")
    print("="*60)

    # Load model once
    print(f"\nLoading backbone model from {model_path}...")
    model = BackboneModel(str(model_path))

    # Test files
    test_files = [
        sample_dir / "statsbomb_sample.json",
        sample_dir / "wyscout_sample.csv",
        sample_dir / "custom_sample.csv"
    ]

    # Verify files exist
    for f in test_files:
        if not f.exists():
            print(f"ERROR: Sample file not found: {f}")
            return 1

    # Test each file
    results = []
    for test_file in test_files:
        try:
            normalized, xt_grid = test_single_file(
                str(test_file),
                model,
                output_dir
            )
            results.append({
                'file': test_file.name,
                'success': True,
                'n_frames': len(normalized['frames']),
                'xt_shape': xt_grid.shape
            })
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'file': test_file.name,
                'success': False,
                'error': str(e)
            })

    # Create baseline heatmap from first file
    print(f"\n{'='*60}")
    print("Creating baseline heatmap...")
    print(f"{'='*60}")

    baseline_file = test_files[0]
    df = load_data(str(baseline_file))
    normalized = normalize_to_json(df)
    xt_grid = model.infer_from_json(normalized)

    baseline_path = output_dir / "baseline_heatmap.png"
    save_heatmap(xt_grid, str(baseline_path), title="Baseline xT Heatmap")
    print(f"Baseline heatmap: {baseline_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if r['success'])
    print(f"Passed: {success_count}/{len(results)}")

    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['file']}")
        if result['success']:
            print(f"    Frames: {result['n_frames']}, xT shape: {result['xt_shape']}")

    print(f"\nAll outputs saved to: {output_dir}")

    # Print sample normalized JSON
    print(f"\n{'='*60}")
    print("Sample Normalized JSON (first frame):")
    print(f"{'='*60}")
    sample_frame = normalized['frames'][0]
    print(json.dumps({'frame': sample_frame}, indent=2)[:500] + "...")

    return 0 if success_count == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
