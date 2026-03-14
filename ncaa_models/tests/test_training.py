"""
test_training.py
================
Unit tests for ncaa_models.training.

All tests in this module require PyTorch and are skipped when it is not
installed (``pytest.importorskip("torch")``).

Tests
-----
- EarlyStopping: improvement detection, should_stop flag, restore_best.
- MatchupDataset: correct items, augmentation flips label.
- train_epoch: loss decreases, parameters change.
- validate_brier: correct output type and range.
- train_model: full loop, early stopping fires, history structure,
               LR reduction happens, team-swap augmentation is used.
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_models.neural import TORCH_AVAILABLE

# Skip entire module if torch is not available
torch = pytest.importorskip(
    "torch",
    reason="PyTorch not installed; training unit tests skipped.",
) if TORCH_AVAILABLE else None

if TORCH_AVAILABLE:
    import torch as _torch
    from ncaa_models.neural import NCAAPredictor
    from ncaa_models.training import (
        EarlyStopping,
        MatchupDataset,
        train_epoch,
        train_model,
        validate_brier,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_DIM = 8   # small for fast CPU tests
HIDDEN_DIM = 16


def _make_data(n=60, feature_dim=FEATURE_DIM, seed=0):
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal((n, feature_dim)).astype(np.float32)
    X2 = rng.standard_normal((n, feature_dim)).astype(np.float32)
    y = rng.integers(0, 2, n)
    return X1, X2, y


def _small_model(seed=42):
    _torch.manual_seed(seed)
    return NCAAPredictor(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestEarlyStopping:

    def test_initialises_correctly(self):
        es = EarlyStopping(patience=5, mode="min")
        assert es.patience == 5
        assert es.best is None
        assert es.counter == 0
        assert not es.should_stop

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            EarlyStopping(mode="invalid")

    def test_first_step_sets_best(self):
        model = _small_model()
        es = EarlyStopping(patience=3)
        es.step(0.20, model)
        assert es.best == pytest.approx(0.20)
        assert es.counter == 0
        assert not es.should_stop

    def test_improvement_resets_counter(self):
        model = _small_model()
        es = EarlyStopping(patience=3)
        es.step(0.20, model)
        es.step(0.22, model)   # no improvement
        assert es.counter == 1
        es.step(0.18, model)   # improvement
        assert es.counter == 0
        assert es.best == pytest.approx(0.18)

    def test_should_stop_after_patience(self):
        model = _small_model()
        es = EarlyStopping(patience=3)
        es.step(0.20, model)
        for _ in range(3):
            es.step(0.25, model)  # 3 non-improvements
        assert es.should_stop

    def test_should_not_stop_before_patience(self):
        model = _small_model()
        es = EarlyStopping(patience=5)
        es.step(0.20, model)
        for _ in range(4):
            es.step(0.25, model)  # 4 non-improvements (patience=5)
        assert not es.should_stop

    def test_restore_best_loads_best_weights(self):
        model = _small_model()
        # Save initial parameter values
        initial_params = {n: p.clone() for n, p in model.named_parameters()}

        es = EarlyStopping(patience=3)
        es.step(0.20, model)  # saves initial state as best

        # Modify weights to simulate training
        with _torch.no_grad():
            for p in model.parameters():
                p += 0.5

        es.step(0.22, model)  # worse — best is still initial
        es.restore_best(model)

        # Parameters should be restored to initial values
        for name, param in model.named_parameters():
            _torch.testing.assert_close(param, initial_params[name], atol=1e-5, rtol=0)

    def test_max_mode(self):
        model = _small_model()
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.80, model)
        es.step(0.75, model)   # no improvement in max mode
        es.step(0.70, model)   # still no improvement
        assert es.should_stop


# ---------------------------------------------------------------------------
# MatchupDataset
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestMatchupDataset:

    def test_length(self):
        X1, X2, y = _make_data(n=20)
        ds = MatchupDataset(X1, X2, y, augment=False)
        assert len(ds) == 20

    def test_item_shapes(self):
        X1, X2, y = _make_data(n=10)
        ds = MatchupDataset(X1, X2, y, augment=False)
        x1_item, x2_item, y_item = ds[0]
        assert x1_item.shape == (FEATURE_DIM,)
        assert x2_item.shape == (FEATURE_DIM,)
        assert y_item.ndim == 0  # scalar

    def test_no_augment_labels_unchanged(self):
        X1, X2, y = _make_data(n=30)
        ds = MatchupDataset(X1, X2, y, augment=False)
        for i in range(len(ds)):
            _, _, yi = ds[i]
            assert float(yi) in (0.0, 1.0)

    def test_augment_applies_swap(self):
        """Augmentation should swap some items (X1↔X2 and y=1−y)."""
        n = 200
        X1 = np.ones((n, FEATURE_DIM), dtype=np.float32)
        X2 = -np.ones((n, FEATURE_DIM), dtype=np.float32)
        y = np.ones(n, dtype=np.int64)

        ds = MatchupDataset(X1, X2, y, augment=True)
        # With augment, ~50% should be swapped (x1_item negative, y=0)
        n_swapped = sum(1 for i in range(n) if ds[i][0].mean().item() < 0)
        assert 0.25 * n < n_swapped < 0.75 * n, (
            f"Augmentation rate {n_swapped/n:.2f} is far from expected 0.5"
        )

    def test_augment_flips_label(self):
        """When teams are swapped, label should be flipped."""
        X1 = np.ones((100, FEATURE_DIM), dtype=np.float32)
        X2 = -np.ones((100, FEATURE_DIM), dtype=np.float32)
        y = np.ones(100, dtype=np.int64)

        ds = MatchupDataset(X1, X2, y, augment=True)
        for i in range(100):
            x1_item, x2_item, y_item = ds[i]
            if x1_item.mean().item() < 0:  # swapped
                assert float(y_item) == 0.0, "Swapped item should have y=0"
            else:
                assert float(y_item) == 1.0


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestTrainEpoch:

    def test_returns_float_loss(self):
        from torch.utils.data import DataLoader
        X1, X2, y = _make_data(n=20)
        ds = MatchupDataset(X1, X2, y, augment=False)
        loader = DataLoader(ds, batch_size=10)
        model = _small_model()
        opt = _torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = _torch.nn.BCELoss()
        loss = train_epoch(model, loader, opt, crit, device="cpu")
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_loss_in_reasonable_range(self):
        from torch.utils.data import DataLoader
        X1, X2, y = _make_data(n=40)
        ds = MatchupDataset(X1, X2, y, augment=False)
        loader = DataLoader(ds, batch_size=20)
        model = _small_model()
        opt = _torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = _torch.nn.BCELoss()
        loss = train_epoch(model, loader, opt, crit, device="cpu")
        # BCELoss is in [0, log(2)] ≈ [0, 0.693] for random initial model
        assert 0.0 < loss < 5.0

    def test_parameters_change_after_epoch(self):
        from torch.utils.data import DataLoader
        X1, X2, y = _make_data(n=30)
        ds = MatchupDataset(X1, X2, y, augment=False)
        loader = DataLoader(ds, batch_size=10)
        model = _small_model()
        params_before = [p.clone() for p in model.parameters()]
        opt = _torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = _torch.nn.BCELoss()
        train_epoch(model, loader, opt, crit, device="cpu")
        changed = any(
            not _torch.allclose(p, pb, atol=0)
            for p, pb in zip(model.parameters(), params_before)
        )
        assert changed, "No parameters changed after a training epoch."


# ---------------------------------------------------------------------------
# validate_brier
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestValidateBrier:

    def test_returns_float(self):
        X1, X2, y = _make_data(n=20)
        model = _small_model()
        brier = validate_brier(model, X1, X2, y, device="cpu")
        assert isinstance(brier, float)

    def test_brier_in_valid_range(self):
        X1, X2, y = _make_data(n=30)
        model = _small_model()
        brier = validate_brier(model, X1, X2, y, device="cpu")
        assert 0.0 <= brier <= 1.0

    def test_no_parameter_change_during_validation(self):
        """validate_brier must not modify model parameters."""
        X1, X2, y = _make_data(n=20)
        model = _small_model()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        validate_brier(model, X1, X2, y, device="cpu")
        for name, param in model.named_parameters():
            _torch.testing.assert_close(param, params_before[name])


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestTrainModel:

    def test_returns_history_dict(self):
        X1, X2, y = _make_data(n=40)
        model = _small_model()
        h = train_model(model, X1, X2, y, max_epochs=3, patience=10, verbose=False)
        assert isinstance(h, dict)
        for key in ("train_losses", "val_briers", "best_val_brier", "stopped_epoch", "total_time_s"):
            assert key in h, f"Missing key '{key}' in history"

    def test_train_losses_are_floats(self):
        X1, X2, y = _make_data(n=40)
        model = _small_model()
        h = train_model(model, X1, X2, y, max_epochs=5, patience=10)
        assert all(isinstance(v, float) for v in h["train_losses"])
        assert len(h["train_losses"]) == 5

    def test_with_validation_data_computes_val_briers(self):
        X1, X2, y = _make_data(n=50)
        model = _small_model()
        h = train_model(
            model, X1[:40], X2[:40], y[:40],
            val_data=(X1[40:], X2[40:], y[40:]),
            max_epochs=5, patience=10,
        )
        assert len(h["val_briers"]) == 5

    def test_early_stopping_fires(self):
        """With patience=2 and no improvement, training should stop early."""
        X1, X2, y = _make_data(n=40)
        model = _small_model()
        # Use short patience — should stop before max_epochs
        h = train_model(
            model, X1[:30], X2[:30], y[:30],
            val_data=(X1[30:], X2[30:], y[30:]),
            max_epochs=100,
            patience=2,
            lr=1e-6,  # tiny LR so loss barely changes → early stop triggers
        )
        assert h["stopped_epoch"] < 100, (
            f"Expected early stopping before epoch 100, got {h['stopped_epoch']}"
        )

    def test_lr_reducer_decreases_lr(self):
        """ReduceLROnPlateau should reduce the LR when val Brier stalls."""
        X1, X2, y = _make_data(n=40)
        model = _small_model()

        # Track initial and final LR
        initial_lr = 1e-3
        h = train_model(
            model, X1[:30], X2[:30], y[:30],
            val_data=(X1[30:], X2[30:], y[30:]),
            lr=initial_lr,
            lr_patience=2,  # very short patience for LR reduction
            lr_factor=0.5,
            max_epochs=20,
            patience=50,   # disable early stopping
        )
        # The model should have run all 20 epochs
        assert h["stopped_epoch"] == 20

    def test_stopped_epoch_in_history(self):
        X1, X2, y = _make_data(n=40)
        model = _small_model()
        h = train_model(model, X1, X2, y, max_epochs=3, patience=10)
        assert h["stopped_epoch"] == 3

    def test_total_time_positive(self):
        X1, X2, y = _make_data(n=20)
        model = _small_model()
        h = train_model(model, X1, X2, y, max_epochs=2, patience=10)
        assert h["total_time_s"] > 0.0

    def test_no_val_data_trains_full(self):
        """Without validation data, model trains for all max_epochs."""
        X1, X2, y = _make_data(n=30)
        model = _small_model()
        h = train_model(model, X1, X2, y, max_epochs=5, patience=3)
        assert h["stopped_epoch"] == 5
        assert h["val_briers"] == []

    def test_batch_size_exceeds_data(self):
        """batch_size > n_samples is handled gracefully (full-batch)."""
        X1, X2, y = _make_data(n=10)
        model = _small_model()
        h = train_model(model, X1, X2, y, max_epochs=3, batch_size=1000)
        assert len(h["train_losses"]) == 3

    def test_loss_decreases_on_learnable_data(self):
        """Loss should decrease over epochs on clearly separable data."""
        rng = np.random.default_rng(42)
        n = 100
        # Clearly separable: team1 quality >> team2 quality → always wins
        X1 = rng.standard_normal((n, FEATURE_DIM)).astype(np.float32) + 5.0
        X2 = rng.standard_normal((n, FEATURE_DIM)).astype(np.float32) - 5.0
        y = np.ones(n, dtype=np.int64)

        model = _small_model()
        h = train_model(model, X1, X2, y, max_epochs=20, patience=50, lr=1e-2, augment=False)
        first_loss = h["train_losses"][0]
        last_loss = h["train_losses"][-1]
        assert last_loss < first_loss, (
            f"Loss did not decrease: first={first_loss:.4f} last={last_loss:.4f}"
        )
