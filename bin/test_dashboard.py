#!/usr/bin/env python3
"""
Test script for FieldSense AI v3.0 Dashboard
Generates insights and prepares data for Tauri UI
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import load_data
from src.data.normalize import normalize_to_json
from src.insights.heatmap import generate_insight
from src.insights.active_learning import ActiveLearningManager, create_ui_label_payload


def create_sample_xt_grid() -> np.ndarray:
    """Create sample xT grid for testing."""
    # Create a 105x68 grid with some interesting patterns
    xt_grid = np.random.rand(105, 68) * 0.5 + 0.3  # Base values 0.3-0.8

    # Add high-value zones
    xt_grid[60:75, 10:25] = np.random.rand(15, 15) * 0.3 + 0.6  # Left attack zone
    xt_grid[70:85, 40:55] = np.random.rand(15, 15) * 0.2 + 0.4  # Right zone

    return xt_grid


def test_insight_generation():
    """Test insight generation with sample data."""
    print("="*60)
    print("FieldSense AI v3.0 - Dashboard Test")
    print("="*60)

    # Setup paths
    base_dir = Path(__file__).parent.parent
    sample_dir = base_dir / "assets" / "sample_data"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load sample play
    print("\n[Step 1/5] Loading sample play...")
    sample_file = sample_dir / "custom_sample.csv"
    df = load_data(str(sample_file))
    normalized = normalize_to_json(df)

    # Get first frame
    play_data = normalized['frames'][0]
    print(f"Loaded play with {len(play_data['players'])} players")

    # Step 2: Generate synthetic xT grid
    print("\n[Step 2/5] Generating xT predictions...")
    xt_grid = create_sample_xt_grid()
    print(f"xT grid shape: {xt_grid.shape}")
    print(f"xT range: [{xt_grid.min():.3f}, {xt_grid.max():.3f}]")

    # Step 3: Generate insight
    print("\n[Step 3/5] Generating exploitation insight...")
    heatmap_path = output_dir / "opportunity_heatmap.png"
    insight = generate_insight(
        xt_grid=xt_grid,
        player_positions=play_data['players'],
        output_path=str(heatmap_path)
    )

    print(f"\n  Decision: {insight['top_decision']}")
    print(f"  Confidence: {insight['confidence']}")
    print(f"  Details:")
    print(f"    - Action: {insight['details']['action']}")
    print(f"    - Zone: {insight['details']['zone']}")
    print(f"    - Opportunity: {insight['details']['opportunity']:.2f}")
    print(f"    - Avg Entropy: {insight['details']['avg_entropy']:.3f}")

    # Step 4: Test active learning
    print("\n[Step 4/5] Testing active learning...")
    al_manager = ActiveLearningManager(
        labels_path=str(output_dir / "active_labels.json"),
        entropy_threshold=0.8
    )

    # Check if should flag
    should_flag = al_manager.should_flag_for_review(xt_grid, insight)
    print(f"  Should flag for review: {should_flag}")

    if should_flag:
        # Create UI payload
        play_data['play_id'] = 'test_play_001'
        ui_payload = create_ui_label_payload(play_data, insight, al_manager)

        if ui_payload:
            print(f"  Flagged reason: {ui_payload['flagged_reason']}")
            print(f"  UI buttons: {len(ui_payload['ui_config']['buttons'])}")

            # Save UI payload for Tauri
            ui_payload_path = output_dir / "ui_payload.json"
            with open(ui_payload_path, 'w') as f:
                # Convert zones to list for JSON serialization
                ui_payload_copy = ui_payload.copy()
                if 'zones' in insight:
                    # Don't include full zones in UI payload (too large)
                    pass

                json.dump(ui_payload, f, indent=2)
            print(f"  UI payload saved: {ui_payload_path}")

    # Simulate label submission
    print("\n  Simulating label submission...")
    label = al_manager.submit_label(
        play_id='test_play_001',
        outcome='Success',
        tags=['LB late', 'WR open'],
        notes='Good read by QB',
        play_data=play_data,
        insight=insight
    )
    print(f"  Label saved: {label['play_id']} -> {label['outcome']}")

    # Get statistics
    stats = al_manager.get_labeling_statistics()
    print(f"\n  Labeling stats:")
    print(f"    Total labels: {stats['total_labels']}")
    print(f"    Outcomes: {stats['outcomes']}")

    # Step 5: Export dashboard data
    print("\n[Step 5/5] Exporting dashboard data...")

    dashboard_data = {
        'play': play_data,
        'insight': {
            'top_decision': insight['top_decision'],
            'confidence': insight['confidence'],
            'details': insight['details']
        },
        'heatmap_path': str(heatmap_path),
        'label_prompt': ui_payload if should_flag else None
    }

    dashboard_path = output_dir / "dashboard_data.json"
    with open(dashboard_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"  Dashboard data saved: {dashboard_path}")

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"\n✓ Insight generated: {insight['top_decision']}")
    print(f"✓ Confidence level: {insight['confidence']}")
    print(f"✓ Heatmap saved: {heatmap_path}")
    print(f"✓ Labels saved: {output_dir / 'active_labels.json'}")
    print(f"✓ Dashboard data: {dashboard_path}")

    print("\n" + "="*60)
    print("To run Tauri dashboard:")
    print("="*60)
    print("1. Install dependencies:")
    print("   cd ui && npm install")
    print("2. Run development server:")
    print("   cd ui && npm run dev")
    print("3. (Optional) Build Tauri app:")
    print("   cargo tauri dev")
    print("\nNote: Tauri requires Rust toolchain to be installed")

    return 0


def test_decision_format():
    """Test decision string format validation."""
    print("\n" + "="*60)
    print("Testing Decision Format")
    print("="*60)

    # Create sample data
    xt_grid = create_sample_xt_grid()
    players = [
        {'id': '1', 'x': 50, 'y': 25, 'role': 'QB', 'possession': True,
         'vel': 5.0, 'dir': 45, 'acc': 1.0},
        {'id': '2', 'x': 65, 'y': 15, 'role': 'WR', 'possession': False,
         'vel': 7.0, 'dir': 90, 'acc': 1.5},
    ]

    insight = generate_insight(xt_grid, players)

    decision = insight['top_decision']
    print(f"\nDecision: {decision}")

    # Validate format: "Action zone: +X% xT"
    parts = decision.split(':')
    assert len(parts) == 2, "Decision should have format 'Action zone: improvement'"

    action_zone = parts[0].strip()
    improvement = parts[1].strip()

    print(f"  Action/Zone: {action_zone}")
    print(f"  Improvement: {improvement}")

    assert '+' in improvement, "Improvement should start with +"
    assert '%' in improvement, "Improvement should include %"
    assert 'xT' in improvement, "Improvement should mention xT"

    print("\n✓ Decision format validation passed")


if __name__ == '__main__':
    try:
        # Run insight generation test
        result = test_insight_generation()

        # Run format validation test
        test_decision_format()

        sys.exit(result)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
