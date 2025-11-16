#!/usr/bin/env python3
"""
FieldSense AI v3.0 - Ship Script

Full integration test and deployment runner:
- Calibration pipeline
- Live mode processing
- Privacy certification
- Coach feedback simulation
- System validation

Usage:
    python bin/ship.py
    ./dist/fieldsense (after PyInstaller build)
"""

import sys
import os
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_banner():
    """Print FieldSense AI banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                   FieldSense AI v3.0                         ║
    ║            Real-Time Sports Analytics Platform               ║
    ║                                                              ║
    ║   Features: Live Video • Counterfactuals • Privacy Cert     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print()


def run_calibration():
    """Run LoRA calibration test."""
    print("=" * 60)
    print("[1/5] Running LoRA Calibration")
    print("=" * 60)
    print()

    try:
        from src.calibration import LoRACalibrator

        print("Initializing calibrator...")
        calibrator = LoRACalibrator(rank=8)

        # Create dummy data
        import numpy as np
        X_dummy = np.random.randn(10, 128)
        y_dummy = np.random.rand(10)

        print("Running calibration on sample data...")
        calibrator.calibrate(X_dummy, y_dummy, epochs=2)

        print("✓ Calibration successful")
        print()
        return True

    except Exception as e:
        print(f"✗ Calibration failed: {e}")
        print()
        return False


def run_live_demo():
    """Run live mode demo."""
    print("=" * 60)
    print("[2/5] Running Live Mode Demo")
    print("=" * 60)
    print()

    try:
        from src.live import LiveEngine, LiveClipEngine

        print("Initializing live engine (dummy mode)...")
        engine = LiveEngine(
            video_source='dummy',
            target_fps=30,
            use_gpu=False,
            dummy_mode=True
        )

        clip_engine = LiveClipEngine(
            output_dir='./outputs/clips',
            xT_threshold=0.25,
            enable_voice=False
        )

        print("Starting live processing...")
        if not engine.start():
            print("✗ Failed to start engine")
            return False

        # Process 60 frames (2 seconds at 30 FPS)
        frame_count = 0
        max_frames = 60

        print(f"Processing {max_frames} frames...")

        while frame_count < max_frames:
            result = engine.ingestor.get_frame(timeout=0.1)
            if result is None:
                continue

            frame, metadata = result
            rendered, inference_result = engine.process_frame(frame, metadata)

            # Trigger clip at frame 30
            if frame_count == 30:
                inference_result.xT_value = 0.35

            clip_engine.process_frame(
                rendered,
                metadata.timestamp,
                inference_result.xT_value,
                {'frame_id': frame_count}
            )

            frame_count += 1

            if frame_count % 15 == 0:
                print(f"  Processed {frame_count}/{max_frames} frames...")

        engine.stop()

        stats = engine.get_stats()
        clip_stats = clip_engine.get_stats()

        print()
        print(f"Live Mode Statistics:")
        print(f"  Frames Processed: {stats['frame_count']}")
        print(f"  Average FPS: {stats['avg_fps']:.1f}")
        print(f"  Clips Generated: {clip_stats['total_clips']}")
        print()
        print("✓ Live mode successful")
        print()
        return True

    except Exception as e:
        print(f"✗ Live mode failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def run_privacy_cert():
    """Run privacy certification."""
    print("=" * 60)
    print("[3/5] Generating Privacy Certificate")
    print("=" * 60)
    print()

    try:
        from src.privacy import one_click_export

        print("Running privacy validation...")
        success = one_click_export(output_dir='./certifications')

        if success:
            print("✓ Privacy certification successful")
        else:
            print("✗ Privacy certification failed")

        print()
        return success

    except Exception as e:
        print(f"✗ Privacy certification error: {e}")
        print()
        return False


def run_coach_feedback():
    """Simulate coach feedback collection."""
    print("=" * 60)
    print("[4/5] Coach Feedback Simulation")
    print("=" * 60)
    print()

    try:
        # Simulate coach data
        coach_data = {
            'play_id': 'PLAY_001',
            'timestamp': time.time(),
            'decision': 'Attack left zone',
            'confidence': 0.87,
            'xT_value': 0.34,
        }

        print("Simulating coach decision review...")
        print(f"  Play ID: {coach_data['play_id']}")
        print(f"  AI Decision: {coach_data['decision']}")
        print(f"  Confidence: {coach_data['confidence']:.1%}")
        print(f"  xT Value: {coach_data['xT_value']:.2f}")
        print()

        # Simulate feedback prompt
        print("Coach Feedback Prompt:")
        print("  Was this decision correct?")
        print("    [1] Correct")
        print("    [2] Incorrect")
        print("    [3] Partially correct")
        print()

        # Auto-select for demo
        feedback = "Correct (simulated)"
        print(f"  Simulated Response: {feedback}")
        print()

        # Store feedback
        feedback_data = {
            **coach_data,
            'coach_feedback': feedback,
            'feedback_timestamp': time.time()
        }

        # Save to file
        import json
        output_dir = Path('./outputs/feedback')
        output_dir.mkdir(parents=True, exist_ok=True)

        feedback_file = output_dir / f"feedback_{coach_data['play_id']}.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)

        print(f"Feedback saved: {feedback_file}")
        print()
        print("✓ Coach feedback simulation successful")
        print()
        return True

    except Exception as e:
        print(f"✗ Coach feedback failed: {e}")
        print()
        return False


def run_system_validation():
    """Run system validation checks."""
    print("=" * 60)
    print("[5/5] System Validation")
    print("=" * 60)
    print()

    checks = []

    # Check imports
    print("Checking module imports...")
    try:
        import numpy
        print("  ✓ numpy")
        checks.append(True)
    except:
        print("  ✗ numpy")
        checks.append(False)

    try:
        import cv2
        print("  ✓ opencv-python")
        checks.append(True)
    except:
        print("  ✗ opencv-python")
        checks.append(False)

    try:
        import pandas
        print("  ✓ pandas")
        checks.append(True)
    except:
        print("  ✗ pandas")
        checks.append(False)

    try:
        import sklearn
        print("  ✓ scikit-learn")
        checks.append(True)
    except:
        print("  ✗ scikit-learn")
        checks.append(False)

    print()

    # Check directories
    print("Checking directory structure...")
    required_dirs = ['src', 'bin', 'ui', 'data', 'outputs']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
            checks.append(True)
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            checks.append(False)

    print()

    # Check modules
    print("Checking FieldSense modules...")
    modules = [
        'src.calibration',
        'src.insights',
        'src.live',
        'src.privacy'
    ]

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
            checks.append(True)
        except Exception as e:
            print(f"  ✗ {module_name} ({e})")
            checks.append(False)

    print()

    passed = sum(checks)
    total = len(checks)
    success_rate = passed / total if total > 0 else 0

    print(f"Validation Results: {passed}/{total} checks passed ({success_rate:.1%})")
    print()

    if success_rate >= 0.8:
        print("✓ System validation successful")
        print()
        return True
    else:
        print("✗ System validation failed")
        print()
        return False


def main():
    """Main entry point."""
    print_banner()

    # Track results
    results = []

    # Run all tests
    tests = [
        ("LoRA Calibration", run_calibration),
        ("Live Mode Demo", run_live_demo),
        ("Privacy Certification", run_privacy_cert),
        ("Coach Feedback", run_coach_feedback),
        ("System Validation", run_system_validation),
    ]

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print final summary
    print()
    print("=" * 60)
    print("Ship Test Summary")
    print("=" * 60)
    print()

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")

    print()

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    if passed_count == total_count:
        print(f"  ✓ ALL TESTS PASSED ({passed_count}/{total_count})")
        print()
        print("  FieldSense AI v3.0 is ready to ship! 🚀")
        print()
        return 0
    else:
        print(f"  ✗ SOME TESTS FAILED ({passed_count}/{total_count})")
        print()
        print("  Please review failures before shipping.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
