"""
test_optimize.py
================
Tests for ncaa_deployment.optimize.

Covers:
- BayesianEnsembleOptimizer converges: optimised Brier < baseline Brier
- OptimizationResult structure and properties
- WeightedEnsemble prediction shape and weight normalisation
- apply_temperature correctness
- Improvement is recorded correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_deployment.optimize import (
    BayesianEnsembleOptimizer,
    OptimizationResult,
    WeightedEnsemble,
    apply_temperature,
)

from .conftest import ConstantModel, TEAM_IDS, SEASON


# ---------------------------------------------------------------------------
# apply_temperature
# ---------------------------------------------------------------------------

class TestApplyTemperature:
    def test_temperature_1_no_change(self):
        probs = np.array([0.3, 0.5, 0.7])
        out = apply_temperature(probs, 1.0)
        assert np.allclose(out, probs, atol=1e-3)

    def test_temperature_gt1_pushes_toward_half(self):
        probs = np.array([0.9, 0.1])
        out = apply_temperature(probs, 5.0)
        # High temperature should pull 0.9 down toward 0.5
        assert out[0] < 0.9
        assert out[1] > 0.1

    def test_temperature_lt1_pushes_away_from_half(self):
        probs = np.array([0.6, 0.4])
        out = apply_temperature(probs, 0.5)
        assert out[0] > 0.6
        assert out[1] < 0.4

    def test_output_clipped_to_valid_range(self):
        probs = np.array([0.0001, 0.9999])
        out = apply_temperature(probs, 1.0)
        assert (out >= 0.01).all()
        assert (out <= 0.99).all()

    def test_symmetry(self):
        """apply_temperature(p, T) + apply_temperature(1-p, T) ≈ 1."""
        p = np.array([0.7])
        t = 2.0
        a = apply_temperature(p, t)
        b = apply_temperature(1.0 - p, t)
        assert abs(float(a[0]) + float(b[0]) - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# WeightedEnsemble
# ---------------------------------------------------------------------------

class TestWeightedEnsemble:
    def test_predict_proba_shape(self, strong_model, val_arrays):
        X1, X2, _ = val_arrays
        ens = WeightedEnsemble(models=[strong_model], weights=np.array([1.0]))
        out = ens.predict_proba(X1, X2)
        assert out.shape == (len(X1),)

    def test_weights_normalised(self, strong_model, weak_model):
        ens = WeightedEnsemble(
            models=[strong_model, weak_model],
            weights=np.array([3.0, 1.0]),
        )
        assert abs(ens.weights.sum() - 1.0) < 1e-9

    def test_equal_weights_matches_average(self, strong_model, weak_model, val_arrays):
        X1, X2, _ = val_arrays
        ens = WeightedEnsemble(
            models=[strong_model, weak_model],
            weights=np.array([1.0, 1.0]),
        )
        out = ens.predict_proba(X1, X2)
        expected = (
            strong_model.predict_proba(X1, X2)
            + weak_model.predict_proba(X1, X2)
        ) / 2.0
        assert np.allclose(out, expected, atol=0.01)

    def test_fit_is_noop(self, strong_model, val_arrays):
        X1, X2, y = val_arrays
        ens = WeightedEnsemble(models=[strong_model], weights=np.array([1.0]))
        result = ens.fit(X1, X2, y)
        assert result is ens  # returns self

    def test_temperature_applied(self, strong_model, val_arrays):
        X1, X2, _ = val_arrays
        ens_t1 = WeightedEnsemble(models=[strong_model], weights=np.array([1.0]), temperature=1.0)
        ens_t5 = WeightedEnsemble(models=[strong_model], weights=np.array([1.0]), temperature=5.0)
        p1 = ens_t1.predict_proba(X1, X2)
        p5 = ens_t5.predict_proba(X1, X2)
        # High temperature should produce predictions closer to 0.5
        spread_t1 = float(np.std(p1))
        spread_t5 = float(np.std(p5))
        assert spread_t5 <= spread_t1 + 0.01  # allow tiny floating point slack


# ---------------------------------------------------------------------------
# BayesianEnsembleOptimizer
# ---------------------------------------------------------------------------

class TestBayesianEnsembleOptimizer:
    def test_optimizer_converges(self, component_models, train_arrays, val_arrays):
        """Optimised weights must produce lower Brier than equal-weight baseline."""
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(
            models=component_models,
            n_trials=50,
            seed=42,
        )
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert result.best_brier < result.baseline_brier, (
            f"Optimiser did not improve: best={result.best_brier:.4f} "
            f">= baseline={result.baseline_brier:.4f}"
        )

    def test_optimizer_returns_result(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=30)
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert isinstance(result, OptimizationResult)

    def test_result_improved_property(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=30)
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert result.improved is True

    def test_weights_sum_to_one(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=30)
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert abs(result.best_weights.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=30)
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert (result.best_weights >= 0).all()

    def test_n_trials_recorded(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=30)
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert result.n_trials > 0

    def test_history_non_empty(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=30)
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert len(result.history) > 0

    def test_improvement_equals_delta(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=30)
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        expected = result.baseline_brier - result.best_brier
        assert abs(result.improvement - expected) < 1e-9

    def test_result_accessor(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=20)
        optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert optimizer.result() is not None

    def test_to_dict_has_required_keys(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=20)
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        d = result.to_dict()
        for key in ("best_weights", "best_brier", "baseline_brier",
                    "n_trials", "improvement", "improved"):
            assert key in d

    def test_build_optimized_ensemble(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=20)
        optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        ens = optimizer.build_optimized_ensemble()
        assert isinstance(ens, WeightedEnsemble)
        preds = ens.predict_proba(X1_val, X2_val)
        assert preds.shape == (len(X1_val),)

    def test_build_ensemble_raises_before_optimize(self, component_models):
        optimizer = BayesianEnsembleOptimizer(models=component_models)
        with pytest.raises(RuntimeError):
            optimizer.build_optimized_ensemble()

    def test_two_model_optimizer(self, strong_model, train_arrays, val_arrays):
        """Optimizer with just 2 models should still converge."""
        from .conftest import ConstantModel
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        import copy
        optimizer = BayesianEnsembleOptimizer(
            models=[copy.deepcopy(strong_model), ConstantModel()],
            n_trials=30,
        )
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert result.improved is True

    def test_metadata_has_n_models(self, component_models, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=20)
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert result.metadata.get("n_models") == len(component_models)
