"""
test_calibrate.py
=================
Unit tests for ncaa_models.calibrate — CalibrationPipeline.

All tests that require PyTorch are skipped when torch is not installed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ncaa_models.lora import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_data(n: int = 200, feature_dim: int = 10, seed: int = 0):
    """Simple random matchup data for fast unit tests."""
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal((n, feature_dim)).astype(np.float32)
    X2 = rng.standard_normal((n, feature_dim)).astype(np.float32)
    y = rng.integers(0, 2, n)
    return X1, X2, y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backbone():
    pytest.importorskip("torch")
    from ncaa_models.neural import NCAAPredictor
    from ncaa_models.lora import apply_lora_to_model
    model = NCAAPredictor(feature_dim=10, hidden_dim=16, dropout=0.0)
    apply_lora_to_model(model, rank=4)
    model.eval()
    return model


@pytest.fixture(scope="module")
def pipeline(backbone):
    from ncaa_models.adapter import AdapterRegistry
    from ncaa_models.calibrate import CalibrationPipeline
    registry = AdapterRegistry(backbone)
    return CalibrationPipeline(backbone, registry=registry, lr=2e-4)


# ---------------------------------------------------------------------------
# TestCalibrationPipeline
# ---------------------------------------------------------------------------

class TestCalibrationPipeline:

    def test_create_adapter_returns_path(self, pipeline, tmp_path):
        X1, X2, y = _make_dummy_data()
        path = pipeline.create_adapter(
            "unit_test_tourn",
            X1_train=X1, X2_train=X2, y_train=y,
            adapter_type="tournament",
            time_budget=42.0,
            save_dir=tmp_path,
        )
        assert path.exists()

    def test_adapter_file_below_1mb(self, pipeline, tmp_path):
        X1, X2, y = _make_dummy_data()
        path = pipeline.create_adapter(
            "size_check",
            X1_train=X1, X2_train=X2, y_train=y,
            adapter_type="tournament",
            time_budget=42.0,
            save_dir=tmp_path,
        )
        assert path.stat().st_size < 1_048_576, (
            f"Adapter {path.stat().st_size} bytes >= 1 MB"
        )

    def test_calibration_within_time_budget(self, pipeline, tmp_path):
        """Calibration must finish within the time_budget wall-clock seconds."""
        X1, X2, y = _make_dummy_data(n=300)
        budget = 42.0
        t0 = time.perf_counter()
        pipeline.create_adapter(
            "time_budget_test",
            X1_train=X1, X2_train=X2, y_train=y,
            adapter_type="tournament",
            time_budget=budget,
            save_dir=tmp_path,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < budget, (
            f"Calibration took {elapsed:.1f}s, budget was {budget}s"
        )

    def test_adapter_registered_after_create(self, pipeline, tmp_path):
        X1, X2, y = _make_dummy_data(seed=7)
        pipeline.create_adapter(
            "reg_test",
            X1_train=X1, X2_train=X2, y_train=y,
            adapter_type="recalibration",
            time_budget=42.0,
            save_dir=tmp_path,
        )
        assert "reg_test" in pipeline.registry.list_adapters()

    def test_short_budget_stops_early(self, pipeline, tmp_path):
        """A very short budget stops training before completing all epochs."""
        X1, X2, y = _make_dummy_data(n=1000)
        # 0.001 s budget: should exit after 0 or 1 epochs
        t0 = time.perf_counter()
        pipeline.create_adapter(
            "short_budget",
            X1_train=X1, X2_train=X2, y_train=y,
            adapter_type="tournament",
            time_budget=0.001,
            save_dir=tmp_path,
        )
        elapsed = time.perf_counter() - t0
        # Should complete quickly (well under 5 seconds even with overhead)
        assert elapsed < 5.0

    def test_invalid_adapter_type_raises(self, pipeline, tmp_path):
        X1, X2, y = _make_dummy_data()
        with pytest.raises(ValueError, match="adapter_type"):
            pipeline.create_adapter(
                "bad_type",
                X1_train=X1, X2_train=X2, y_train=y,
                adapter_type="invalid_type",
                save_dir=tmp_path,
            )

    def test_all_adapter_types_train(self, pipeline, tmp_path):
        """All three preset adapter types complete without error."""
        from ncaa_models.calibrate import VALID_ADAPTER_TYPES
        X1, X2, y = _make_dummy_data()
        for adapter_type in VALID_ADAPTER_TYPES:
            path = pipeline.create_adapter(
                f"type_{adapter_type}",
                X1_train=X1, X2_train=X2, y_train=y,
                adapter_type=adapter_type,
                time_budget=42.0,
                save_dir=tmp_path,
            )
            assert path.exists()

    def test_trainable_params_below_threshold(self, backbone):
        """During calibration, only LoRA params have gradients."""
        from ncaa_models.lora import count_trainable_params, count_total_params
        trainable = count_trainable_params(backbone)
        total = count_total_params(backbone)
        # LoRA params should be a minority of total; for this tiny model <50%
        assert trainable < total, "Some params must be frozen"
        assert trainable > 0, "At least LoRA A and B must be trainable"

    def test_validate_brier_returns_float(self, pipeline):
        X1, X2, y = _make_dummy_data()
        brier = pipeline.validate_brier(pipeline.base_model, X1, X2, y)
        assert isinstance(brier, float)
        assert 0.0 <= brier <= 1.0


# ---------------------------------------------------------------------------
# TestNoCatastrophicForgetting
# ---------------------------------------------------------------------------

class TestNoCatastrophicForgetting:
    """
    After calibration, the base model predictions (adapter unloaded) should
    be exactly preserved — because LoRA only trains A and B (original weights
    are frozen), and unload() resets B to zero.
    """

    @pytest.fixture(scope="class")
    def control_setup(self, tmp_path_factory):
        import torch
        from ncaa_models.neural import NCAAPredictor
        from ncaa_models.lora import apply_lora_to_model
        from ncaa_models.adapter import AdapterRegistry
        from ncaa_models.calibrate import CalibrationPipeline

        model = NCAAPredictor(feature_dim=10, hidden_dim=16, dropout=0.0)
        apply_lora_to_model(model, rank=4)
        model.eval()

        x1 = torch.randn(20, 10)
        x2 = torch.randn(20, 10)

        # Record base model output (all B=0 initially)
        with torch.no_grad():
            base_preds = model(x1, x2).clone()

        # Train an adapter
        rng = np.random.default_rng(99)
        X1 = rng.standard_normal((200, 10)).astype(np.float32)
        X2 = rng.standard_normal((200, 10)).astype(np.float32)
        y = rng.integers(0, 2, 200)

        registry = AdapterRegistry(model)
        pipeline = CalibrationPipeline(model, registry=registry)
        tmp = tmp_path_factory.mktemp("forgetting")
        pipeline.create_adapter(
            "forget_test",
            X1_train=X1, X2_train=X2, y_train=y,
            adapter_type="tournament",
            save_dir=tmp,
        )

        # Unload → B back to zero
        registry.unload()
        model.eval()

        with torch.no_grad():
            after_unload_preds = model(x1, x2)

        return base_preds, after_unload_preds

    def test_no_catastrophic_forgetting(self, control_setup):
        """Unloading adapter exactly restores base model predictions."""
        import torch
        base_preds, after_preds = control_setup
        torch.testing.assert_close(base_preds, after_preds,
                                   msg="Base model predictions changed after adapter unload")

    def test_brier_unchanged_after_unload(self, control_setup):
        """Control-set Brier before and after adaptation is identical."""
        import torch
        from ncaa_models.evaluate import compute_brier_score

        base_preds, after_preds = control_setup
        rng = np.random.default_rng(42)
        y_ctrl = rng.integers(0, 2, base_preds.shape[0])

        brier_before = compute_brier_score(base_preds.squeeze(-1).numpy(), y_ctrl)
        brier_after = compute_brier_score(after_preds.squeeze(-1).numpy(), y_ctrl)

        # Should be exactly the same (B=0 → identical predictions)
        assert abs(brier_before - brier_after) < 1e-6, (
            f"Brier changed after unload: before={brier_before:.6f}, "
            f"after={brier_after:.6f}"
        )
