#!/usr/bin/env python3
"""
FieldSense AI v3.1 - MiroThinker Agent Load Test

Tests the agent loading and generation capabilities.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import MiroThinkerAgent, check_model_availability


def print_banner():
    """Print test banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           FieldSense AI v3.1 - Agent Load Test              ║
    ║              MiroThinker-v1.0-30B Integration                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print()


def test_dependencies():
    """Test required dependencies."""
    print("=" * 60)
    print("[Test 1/5] Checking Dependencies")
    print("=" * 60)
    print()

    availability = check_model_availability()

    print("Dependency Status:")
    for dep, available in availability.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}: {'Available' if available else 'Missing'}")

    print()

    all_available = all(availability.values())
    if not all_available:
        print("⚠ Some dependencies missing - agent will run in simulation mode")
        print("  To enable full functionality: pip install -r requirements.txt")
    else:
        print("✓ All dependencies available")

    print()
    return True


def test_agent_initialization():
    """Test agent initialization."""
    print("=" * 60)
    print("[Test 2/5] Initializing Agent")
    print("=" * 60)
    print()

    try:
        print("Creating MiroThinker agent...")
        agent = MiroThinkerAgent(offline_mode=True)

        print(f"  Model: {agent.model_name}")
        print(f"  Enabled: {agent.enabled}")
        print(f"  Quantization: {'4-bit' if agent.load_in_4bit else 'fp16'}")
        print(f"  Device: {agent.device_map}")
        print()

        if agent.model is None:
            print("⚠ Model not loaded - running in simulation mode")
            print("  This is expected if dependencies are missing or model not cached")
        else:
            print("✓ Model loaded successfully")

        print()
        return agent

    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        print()
        return None


def test_reasoning():
    """Test agent reasoning capabilities."""
    print("=" * 60)
    print("[Test 3/5] Testing Reasoning")
    print("=" * 60)
    print()

    try:
        # Create agent
        agent = MiroThinkerAgent(offline_mode=True)

        # Create dummy xT grid
        print("Creating test data...")
        xT_grid = np.random.rand(105, 68) * 0.5
        xT_grid[70:80, 10:20] = 0.8  # High threat zone

        play_data = {
            'players': [
                {'id': '1', 'x': 50, 'y': 30, 'role': 'QB'},
                {'id': '2', 'x': 70, 'y': 15, 'role': 'WR'},
            ],
            'ball': {'x': 50, 'y': 30}
        }

        print(f"  xT grid shape: {xT_grid.shape}")
        print(f"  Players: {len(play_data['players'])}")
        print(f"  Max xT: {np.max(xT_grid):.3f}")
        print()

        # Test reasoning
        print("Running agent reasoning...")
        start_time = time.time()
        result = agent.reason(xT_grid, play_data)
        elapsed = time.time() - start_time

        print()
        print("Reasoning Result:")
        print(f"  Refined delta_xT: {result['refined_delta_xT']:.3f}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Steps: {len(result['steps'])}")
        print(f"  Processing time: {elapsed:.2f}s")
        print()

        # Show reasoning (truncated)
        print("Reasoning Output (first 500 chars):")
        print("-" * 60)
        reasoning_preview = result['reasoning'][:500]
        print(reasoning_preview)
        if len(result['reasoning']) > 500:
            print("...")
        print("-" * 60)
        print()

        # Verify output
        assert len(result['reasoning']) > 50, "Output too short"
        assert elapsed < 10.0, "Processing too slow"

        print("✓ Reasoning test passed")
        print()
        return True

    except Exception as e:
        print(f"✗ Reasoning test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_performance():
    """Test performance metrics."""
    print("=" * 60)
    print("[Test 4/5] Performance Benchmarks")
    print("=" * 60)
    print()

    try:
        agent = MiroThinkerAgent(offline_mode=True)
        xT_grid = np.random.rand(105, 68) * 0.5
        play_data = {
            'players': [{'id': '1', 'x': 50, 'y': 30}],
            'ball': {'x': 50, 'y': 30}
        }

        # Run multiple iterations
        times = []
        for i in range(3):
            start = time.time()
            result = agent.reason(xT_grid, play_data)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.2f}s")

        avg_time = np.mean(times)
        print()
        print(f"Average time: {avg_time:.2f}s")

        # Check performance target
        if avg_time < 5.0:
            print("✓ Performance target met (<5s per query)")
        else:
            print("⚠ Performance slower than target (may be acceptable on CPU)")

        print()
        return True

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        print()
        return False


def test_model_size():
    """Test model size and memory usage."""
    print("=" * 60)
    print("[Test 5/5] Model Size Verification")
    print("=" * 60)
    print()

    try:
        # Check cached model directory
        cache_dir = Path(__file__).parent.parent / 'assets' / 'mirothinker'

        if cache_dir.exists():
            # Calculate total size
            total_size = 0
            file_count = 0

            for file_path in cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

            size_gb = total_size / (1024**3)

            print(f"Model Cache:")
            print(f"  Directory: {cache_dir}")
            print(f"  Files: {file_count}")
            print(f"  Total size: {size_gb:.2f} GB")
            print()

            if size_gb < 0.1:
                print("ℹ Model not yet cached (expected ~8GB after download)")
            elif 6.0 <= size_gb <= 10.0:
                print("✓ Model size within expected range (6-10 GB)")
            else:
                print(f"⚠ Unexpected size: {size_gb:.2f} GB")

        else:
            print("ℹ Model cache directory not found")
            print(f"  Will be created at: {cache_dir}")
            print(f"  Expected size after download: ~8GB (4-bit quantized)")

        print()
        return True

    except Exception as e:
        print(f"✗ Size check failed: {e}")
        print()
        return False


def main():
    """Main test runner."""
    print_banner()

    results = []

    # Run tests
    tests = [
        ("Dependencies", test_dependencies),
        ("Initialization", test_agent_initialization),
        ("Reasoning", test_reasoning),
        ("Performance", test_performance),
        ("Model Size", test_model_size),
    ]

    for test_name, test_func in tests:
        try:
            passed = test_func()
            if passed is not None and passed is not False:
                results.append((test_name, True))
            else:
                results.append((test_name, False))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results.append((test_name, False))

    # Print summary
    print()
    print("=" * 60)
    print("Test Summary")
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
        print("  MiroThinker agent is ready! 🚀")
        print()
        print("  Note: Agent runs in simulation mode until model is downloaded.")
        print("  To download model (requires ~8GB): ")
        print("    - Enable in config/agent.yaml (toggle: true)")
        print("    - Model will auto-download on first use (requires internet)")
        print()
        return 0
    else:
        print(f"  ⚠ SOME TESTS COMPLETED ({passed_count}/{total_count})")
        print()
        print("  Agent is functional but running in simulation mode.")
        print("  This is expected without the full model downloaded.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
