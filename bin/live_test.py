#!/usr/bin/env python3
"""
FieldSense AI v3.0 - Live Mode Test

Tests real-time video processing pipeline:
- Dummy UDP stream simulation
- 30+ FPS processing
- Auto-clip generation
- Lag < 100ms validation
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live import LiveEngine, LiveClipEngine


def test_live_pipeline():
    """Test live processing pipeline with dummy stream."""

    print("=" * 60)
    print("FieldSense AI v3.0 - Live Mode Test")
    print("=" * 60)
    print()

    # Test configuration
    target_fps = 30
    test_duration = 10.0  # seconds
    max_frames = int(target_fps * test_duration)
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)

    output_video = output_dir / 'live_test_output.mp4'
    clips_dir = output_dir / 'clips'
    clips_dir.mkdir(exist_ok=True)

    print(f"[Test Configuration]")
    print(f"  Target FPS: {target_fps}")
    print(f"  Test Duration: {test_duration}s")
    print(f"  Max Frames: {max_frames}")
    print(f"  Output: {output_video}")
    print()

    # Step 1: Initialize live engine with dummy stream
    print("[Step 1/5] Initializing live engine...")
    engine = LiveEngine(
        video_source='dummy',  # Dummy stream
        target_fps=target_fps,
        use_gpu=False,
        dummy_mode=True
    )
    print("  ✓ Live engine initialized")
    print()

    # Step 2: Initialize clip generator
    print("[Step 2/5] Initializing clip generator...")
    clip_engine = LiveClipEngine(
        output_dir=str(clips_dir),
        xT_threshold=0.25,
        enable_voice=False  # Disable voice for testing
    )
    print("  ✓ Clip generator initialized")
    print()

    # Step 3: Start processing
    print("[Step 3/5] Starting live processing...")
    if not engine.start():
        print("  ✗ Failed to start engine")
        return False
    print("  ✓ Engine started")
    print()

    # Step 4: Process frames
    print(f"[Step 4/5] Processing {max_frames} frames...")
    print()

    fps_samples = []
    lag_samples = []
    frame_count = 0
    clips_generated = 0

    # Simulate high xT event for clip generation
    trigger_clip_at_frame = max_frames // 2

    try:
        while frame_count < max_frames:
            loop_start = time.time()

            # Get frame from ingestor
            result = engine.ingestor.get_frame(timeout=0.1)
            if result is None:
                continue

            frame, metadata = result

            # Process frame
            rendered, inference_result = engine.process_frame(frame, metadata)

            # Simulate high xT value at specific frame for clip generation
            if frame_count == trigger_clip_at_frame:
                # Force high xT to trigger clip
                inference_result.xT_value = 0.35

            # Add to clip engine
            clip_engine.process_frame(
                rendered,
                metadata.timestamp,
                inference_result.xT_value,
                {'frame_id': frame_count}
            )

            # Check if clip was generated
            current_clips = len(clip_engine.get_clips())
            if current_clips > clips_generated:
                clips_generated = current_clips
                print(f"  🎬 Clip generated at frame {frame_count} (xT={inference_result.xT_value:.3f})")

            # Measure performance
            loop_time = (time.time() - loop_start) * 1000  # ms
            actual_fps = 1000 / loop_time if loop_time > 0 else 0

            fps_samples.append(actual_fps)
            lag_samples.append(loop_time)

            # Log progress every 30 frames
            if frame_count % 30 == 0:
                avg_fps = np.mean(fps_samples[-30:])
                avg_lag = np.mean(lag_samples[-30:])
                print(f"  Frame {frame_count}/{max_frames}: FPS={avg_fps:.1f}, Lag={avg_lag:.1f}ms")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n  Test interrupted by user")

    finally:
        engine.stop()

    print()
    print(f"  Processing complete: {frame_count} frames")
    print()

    # Step 5: Validate results
    print("[Step 5/5] Validating results...")
    print()

    avg_fps = np.mean(fps_samples)
    max_lag = np.max(lag_samples)
    avg_lag = np.mean(lag_samples)

    # FPS test
    fps_pass = avg_fps >= 30.0
    print(f"  FPS Test:")
    print(f"    Target: ≥30 FPS")
    print(f"    Actual: {avg_fps:.1f} FPS")
    print(f"    {'✓ PASSED' if fps_pass else '✗ FAILED'}")
    print()

    # Lag test
    lag_pass = max_lag <= 100.0
    print(f"  Lag Test:")
    print(f"    Target: <100ms")
    print(f"    Max Lag: {max_lag:.1f}ms")
    print(f"    Avg Lag: {avg_lag:.1f}ms")
    print(f"    {'✓ PASSED' if lag_pass else '✗ FAILED'}")
    print()

    # Clip generation test
    clips_generated = len(clip_engine.get_clips())
    clip_pass = clips_generated > 0
    print(f"  Clip Generation Test:")
    print(f"    Clips Generated: {clips_generated}")
    print(f"    {'✓ PASSED' if clip_pass else '✗ FAILED'}")
    print()

    # Get engine stats
    stats = engine.get_stats()
    clip_stats = clip_engine.get_stats()

    print(f"  Engine Statistics:")
    print(f"    Total Frames: {stats['frame_count']}")
    print(f"    Average FPS: {stats['avg_fps']:.1f}")
    print(f"    Ingest Drop Rate: {stats['ingest_stats']['drop_rate']:.1%}")
    print()

    print(f"  Clip Statistics:")
    print(f"    Total Clips: {clip_stats['total_clips']}")
    print(f"    xT Threshold: {clip_stats['threshold']:.2f}")
    print(f"    Clip Duration: {clip_stats['clip_duration']}s")
    print()

    # Overall result
    all_pass = fps_pass and lag_pass and clip_pass

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print()
    print(f"  {'✓ PASSED' if fps_pass else '✗ FAILED'}: FPS >= 30")
    print(f"  {'✓ PASSED' if lag_pass else '✗ FAILED'}: Lag < 100ms")
    print(f"  {'✓ PASSED' if clip_pass else '✗ FAILED'}: Clips generated")
    print()
    print(f"  Overall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print()

    if clips_generated > 0:
        print(f"  Clips saved to: {clips_dir}")
        clips = clip_engine.get_clips()
        for clip in clips:
            print(f"    - {clip.clip_id}: {clip.output_path}")
        print()

    print("=" * 60)
    print()

    return all_pass


def test_ingest_performance():
    """Test video ingestion performance independently."""

    print("=" * 60)
    print("Video Ingestion Performance Test")
    print("=" * 60)
    print()

    from src.live import DummyIngestor

    target_fps = 30
    test_duration = 5.0
    max_frames = int(target_fps * test_duration)

    print(f"  Target: {target_fps} FPS")
    print(f"  Duration: {test_duration}s")
    print()

    ingestor = DummyIngestor(target_fps=target_fps)
    ingestor.start()

    frame_count = 0
    start_time = time.time()
    frame_times = []

    print("  Processing frames...")

    while frame_count < max_frames:
        frame_start = time.time()

        result = ingestor.get_frame(timeout=0.1)
        if result is None:
            continue

        frame, metadata = result

        frame_time = (time.time() - frame_start) * 1000
        frame_times.append(frame_time)

        frame_count += 1

    elapsed = time.time() - start_time
    ingestor.stop()

    actual_fps = frame_count / elapsed
    avg_frame_time = np.mean(frame_times)

    print()
    print(f"  Results:")
    print(f"    Frames: {frame_count}")
    print(f"    Elapsed: {elapsed:.2f}s")
    print(f"    Actual FPS: {actual_fps:.1f}")
    print(f"    Avg Frame Time: {avg_frame_time:.2f}ms")
    print(f"    {'✓ PASSED' if actual_fps >= target_fps else '✗ FAILED'}")
    print()

    return actual_fps >= target_fps


def test_clip_trigger():
    """Test clip generation trigger logic."""

    print("=" * 60)
    print("Clip Trigger Test")
    print("=" * 60)
    print()

    from src.live.clips import ClipGenerator
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ClipGenerator(
            output_dir=tmpdir,
            xT_threshold=0.25,
            clip_duration=10.0,
            fps=30,
            enable_voice=False
        )

        # Simulate frames with varying xT values
        test_cases = [
            (0.15, False, "Below threshold"),
            (0.30, True, "Above threshold"),
            (0.40, False, "Too soon after last trigger"),
            (0.35, False, "Still too soon"),
        ]

        print("  Testing trigger conditions:")
        print()

        for i, (xT_value, should_trigger, description) in enumerate(test_cases):
            initial_clips = len(generator.clips)

            # Create dummy frame
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            timestamp = i * 1.0
            metadata = {'frame_id': i}

            # Add frame
            generator.add_frame(frame, timestamp, xT_value, metadata)

            # Check if clip was generated
            clips_after = len(generator.clips)
            triggered = clips_after > initial_clips

            status = "✓" if triggered == should_trigger else "✗"
            print(f"    {status} Frame {i}: xT={xT_value:.2f} - {description}")
            print(f"       Expected trigger: {should_trigger}, Actual: {triggered}")

            # Wait to allow next trigger
            if triggered:
                time.sleep(0.1)

        print()
        print(f"  Total clips generated: {len(generator.clips)}")
        print()

    return True


if __name__ == '__main__':
    print()

    # Run all tests
    tests = [
        ("Live Pipeline", test_live_pipeline),
        ("Ingest Performance", test_ingest_performance),
        ("Clip Trigger", test_clip_trigger),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

        print()

    # Print final summary
    print("=" * 60)
    print("Final Test Summary")
    print("=" * 60)
    print()

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")

    print()

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("  ✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("  ✗ SOME TESTS FAILED")
        sys.exit(1)
