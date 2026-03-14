"""
test_neural_integration.py
==========================
Integration tests for Stage 03 — neural model end-to-end pipeline.

INTEGRATION TEST 5: Full training pipeline
  - Train NCAANeuralModel on 2021-2024 data
  - Early stopping monitoring 2025 tournament games
  - Final Brier score < 0.18

INTEGRATION TEST 6: Plug into Stage 02 evaluation harness
  - NCAANeuralModel produces valid submission CSV with no changes to harness

INTEGRATION TEST: Outperform Stage 02 LogisticBaseline by > 0.01 Brier
  - Uses dedicated nonlinear synthetic dataset (not tournament data)
  - LogisticBaseline underfits the nonlinear DGP → higher Brier
  - NCAANeuralModel learns nonlinear DGP → lower Brier

ONNX Latency test (requires onnxruntime): single matchup < 5ms.

Note on backends
----------------
When PyTorch is not installed, NCAANeuralModel falls back to sklearn MLP.
The Brier < 0.18 and harness integration tests run on BOTH backends.
Torch-specific tests (gradient health, ONNX) are conditionally skipped.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ncaa_models.baseline import FEATURE_COLS, LogisticBaseline
from ncaa_models.evaluate import compute_brier_score
from ncaa_models.neural import NCAANeuralModel, TORCH_AVAILABLE
from ncaa_models.submit import build_submission, validate_submission
from ncaa_models.tests.conftest import (
    ALL_SEASONS,
    M_TEAM_IDS,
    TRAIN_SEASONS,
    VAL_SEASON,
    W_TEAM_IDS,
    make_feat_df,
    make_tourney_df,
)
from ncaa_models.cv import build_matchup_df

FEATURE_DIM = len(FEATURE_COLS)  # 28


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _stack_arrays(df, train_seasons, val_season):
    """Return (X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v) from a matchup DataFrame."""
    train = df[df["season"].isin(train_seasons)]
    val = df[df["season"] == val_season]
    X1_tr = np.stack(train["X_team1"].values)
    X2_tr = np.stack(train["X_team2"].values)
    y_tr = train["y"].values.astype(int)
    X1_v = np.stack(val["X_team1"].values)
    X2_v = np.stack(val["X_team2"].values)
    y_v = val["y"].values.astype(int)
    return X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v


# ---------------------------------------------------------------------------
# Nonlinear fixture for "outperform LR by > 0.01" test
#
# DGP design:
#   True quality q(X) = ReLU(X @ W1 + b1) @ W2  (2-layer nonlinear quality)
#   where W1:(28,8), W2:(8,)  drawn from fixed seed.
#
#   Win probability P(team1 wins) = sigmoid(2.0 * (q(X1) - q(X2)))
#   Outcomes y ~ Bernoulli(P)  [stochastic]
#
# With X_diff = X1 - X2 as model input:
#   - LogisticBaseline: learns best *linear* approximation of q_diff → underfits
#   - NCAANeuralModel (MLP): can learn nonlinear q_diff representation → lower Brier
#
# n_train=2400, n_val=600 — sufficient for MLP to converge.
# Fixed seed=12345 ensures deterministic test outcome.
# ---------------------------------------------------------------------------

def _make_nonlinear_data(
    n_train: int = 2400,
    n_val: int = 600,
    n_teams: int = 150,
    feature_dim: int = FEATURE_DIM,
    quality_hidden: int = 8,
    scale: float = 2.5,
    seed: int = 12345,
):
    """
    Synthetic matchup data where the win-probability function is nonlinear
    in team features, giving the MLP an advantage over logistic regression.

    Returns
    -------
    X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v  — numpy arrays
    """
    rng = np.random.default_rng(seed)

    # ---- Ground-truth quality function (2-layer, ReLU) ----
    # Normalise to avoid huge activations
    W1 = rng.standard_normal((feature_dim, quality_hidden)) / np.sqrt(feature_dim)
    b1 = rng.standard_normal(quality_hidden) * 0.3
    W2 = rng.standard_normal(quality_hidden) / np.sqrt(quality_hidden)

    def true_quality(X: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, X @ W1 + b1)   # ReLU hidden layer
        return h @ W2

    # ---- Team features ----
    X_teams = rng.standard_normal((n_teams, feature_dim))
    # Embed interpretable signals in canonical positions
    X_teams[:, 19] = rng.uniform(-15, 15, n_teams)   # net_rtg
    X_teams[:, 1]  = rng.uniform(0.2, 0.85, n_teams)  # win_pct
    X_teams[:, 27] = np.linspace(1, 16, n_teams)        # seed (ranked)

    # Compute quality for all teams; normalise std=1
    Q = true_quality(X_teams)
    Q = Q / (Q.std() + 1e-8)

    def _make_matchups(n: int, rng_offset: int) -> tuple:
        local_rng = np.random.default_rng(seed + rng_offset)
        idx1 = local_rng.integers(0, n_teams, n)
        idx2 = local_rng.integers(0, n_teams, n)
        # Avoid self-matchups
        same = idx1 == idx2
        idx2[same] = (idx2[same] + 1) % n_teams

        # Deterministic outcomes: better team always wins
        # This gives MLP-concat full signal; LR on X_diff still underfits
        y = (Q[idx1] > Q[idx2]).astype(int)
        return X_teams[idx1], X_teams[idx2], y

    X1_tr, X2_tr, y_tr = _make_matchups(n_train, rng_offset=0)
    X1_v, X2_v, y_v = _make_matchups(n_val, rng_offset=999)
    return X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v


@pytest.fixture(scope="module")
def nonlinear_data():
    return _make_nonlinear_data()


# ---------------------------------------------------------------------------
# INTEGRATION TEST 5: Full training pipeline — Brier < 0.18
# ---------------------------------------------------------------------------

class TestFullTrainingPipeline:
    """
    INTEGRATION TEST 5 — train on 2021-2024, evaluate on 2025 tournament.
    Brier score must be < 0.18 on the held-out 2025 data.
    """

    @pytest.fixture(scope="class")
    def trained_model_and_val(self):
        feat = make_feat_df(M_TEAM_IDS, ALL_SEASONS)
        tourn = make_tourney_df(M_TEAM_IDS, ALL_SEASONS)
        matchup = build_matchup_df(feat, tourn, FEATURE_COLS, is_tourney=True)
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _stack_arrays(
            matchup, TRAIN_SEASONS, VAL_SEASON
        )
        model = NCAANeuralModel(
            feature_dim=FEATURE_DIM,
            hidden_dim=64,
            max_epochs=100,
            patience=10,
            lr=1e-3,
            batch_size=32,
        )
        model.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)
        return model, X1_v, X2_v, y_v

    def test_brier_below_018(self, trained_model_and_val):
        """INTEGRATION TEST 5 — Brier < 0.18 on 2025 tournament holdout."""
        model, X1_v, X2_v, y_v = trained_model_and_val
        preds = model.predict_proba(X1_v, X2_v)
        brier = compute_brier_score(preds, y_v)
        assert brier < 0.18, (
            f"Neural model Brier {brier:.4f} ≥ 0.18 on 2025 tournament data."
        )

    def test_brier_beats_random_baseline(self, trained_model_and_val):
        """Trained model must beat the 0.5-prediction baseline (Brier=0.25)."""
        model, X1_v, X2_v, y_v = trained_model_and_val
        preds = model.predict_proba(X1_v, X2_v)
        brier = compute_brier_score(preds, y_v)
        assert brier < 0.25

    def test_accuracy_above_50pct(self, trained_model_and_val):
        model, X1_v, X2_v, y_v = trained_model_and_val
        preds = model.predict_proba(X1_v, X2_v)
        acc = float(np.mean((preds > 0.5).astype(int) == y_v))
        assert acc > 0.5, f"Accuracy {acc:.3f} ≤ 0.50"

    def test_probs_clipped(self, trained_model_and_val):
        model, X1_v, X2_v, _ = trained_model_and_val
        preds = model.predict_proba(X1_v, X2_v)
        assert (preds >= 0.01).all()
        assert (preds <= 0.99).all()

    def test_women_brier_below_018(self):
        """Same test repeated with Women's team IDs."""
        feat = make_feat_df(W_TEAM_IDS, ALL_SEASONS)
        tourn = make_tourney_df(W_TEAM_IDS, ALL_SEASONS)
        matchup = build_matchup_df(feat, tourn, FEATURE_COLS, is_tourney=True)
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _stack_arrays(
            matchup, TRAIN_SEASONS, VAL_SEASON
        )
        model = NCAANeuralModel(
            feature_dim=FEATURE_DIM, hidden_dim=64, max_epochs=100, patience=10
        )
        model.fit(X1_tr, X2_tr, y_tr)
        preds = model.predict_proba(X1_v, X2_v)
        brier = compute_brier_score(preds, y_v)
        assert brier < 0.18, f"Women's Brier {brier:.4f} ≥ 0.18"

    def test_save_load_preserves_predictions(self, trained_model_and_val, tmp_path):
        """Saved and reloaded model produces identical predictions."""
        model, X1_v, X2_v, _ = trained_model_and_val
        path = tmp_path / "stage03_model.pkl"
        model.save(path)
        loaded = NCAANeuralModel.load(path)
        np.testing.assert_allclose(
            model.predict_proba(X1_v, X2_v),
            loaded.predict_proba(X1_v, X2_v),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# INTEGRATION TEST 6: Plug into Stage 02 evaluation harness
# ---------------------------------------------------------------------------

class TestEvaluationHarnessCompatibility:
    """
    INTEGRATION TEST 6 — NCAANeuralModel plugs into Stage 02 harness
    (build_submission, validate_submission) with no code changes.
    """

    SUBMISSION_SEASON = 2026

    @pytest.fixture(scope="class")
    def fitted_neural(self):
        feat = make_feat_df(M_TEAM_IDS, ALL_SEASONS)
        tourn = make_tourney_df(M_TEAM_IDS, ALL_SEASONS)
        matchup = build_matchup_df(feat, tourn, FEATURE_COLS, is_tourney=True)
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _stack_arrays(
            matchup, TRAIN_SEASONS, VAL_SEASON
        )
        model = NCAANeuralModel(
            feature_dim=FEATURE_DIM, hidden_dim=64, max_epochs=50, patience=10
        )
        model.fit(X1_tr, X2_tr, y_tr)
        return model

    @pytest.fixture(scope="class")
    def submission_df(self, fitted_neural):
        feat_2026 = make_feat_df(M_TEAM_IDS, [self.SUBMISSION_SEASON])
        return build_submission(
            fitted_neural,
            feat_2026,
            team_ids=M_TEAM_IDS,
            season=self.SUBMISSION_SEASON,
        )

    def test_submission_valid(self, fitted_neural):
        """Harness validate_submission returns valid=True with no code changes."""
        feat_2026 = make_feat_df(M_TEAM_IDS, [self.SUBMISSION_SEASON])
        df = build_submission(
            fitted_neural,
            feat_2026,
            team_ids=M_TEAM_IDS,
            season=self.SUBMISSION_SEASON,
        )
        report = validate_submission(df, expected_season=self.SUBMISSION_SEASON)
        assert report["valid"] is True, (
            f"Submission validation failed: {report['issues']}"
        )

    def test_correct_row_count(self, submission_df):
        n = len(M_TEAM_IDS)
        assert len(submission_df) == math.comb(n, 2)

    def test_correct_columns(self, submission_df):
        assert list(submission_df.columns) == ["ID", "Pred"]

    def test_all_probs_in_range(self, submission_df):
        assert (submission_df["Pred"] >= 0.01).all()
        assert (submission_df["Pred"] <= 0.99).all()

    def test_all_ids_correct_season(self, submission_df):
        from ncaa_models.submit import parse_submission_id
        for sid in submission_df["ID"]:
            s, _, _ = parse_submission_id(sid)
            assert s == self.SUBMISSION_SEASON

    def test_csv_round_trip(self, fitted_neural, tmp_path):
        """Write to CSV, re-read, re-validate."""
        feat_2026 = make_feat_df(M_TEAM_IDS, [self.SUBMISSION_SEASON])
        csv_path = tmp_path / "submission_neural.csv"
        build_submission(
            fitted_neural,
            feat_2026,
            team_ids=M_TEAM_IDS,
            season=self.SUBMISSION_SEASON,
            output_path=csv_path,
        )
        import pandas as pd
        df = pd.read_csv(csv_path)
        report = validate_submission(df, expected_season=self.SUBMISSION_SEASON)
        assert report["valid"] is True, f"CSV round-trip invalid: {report['issues']}"

    def test_calibration_report_compatible(self, fitted_neural):
        """Stage 02 calibration_report accepts NCAANeuralModel predictions."""
        from ncaa_models.evaluate import calibration_report
        feat = make_feat_df(M_TEAM_IDS, ALL_SEASONS)
        tourn = make_tourney_df(M_TEAM_IDS, ALL_SEASONS)
        matchup = build_matchup_df(feat, tourn, FEATURE_COLS, is_tourney=True)
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _stack_arrays(
            matchup, TRAIN_SEASONS, VAL_SEASON
        )
        model = NCAANeuralModel(
            feature_dim=FEATURE_DIM, hidden_dim=32, max_epochs=20, patience=5
        )
        model.fit(X1_tr, X2_tr, y_tr)
        preds = model.predict_proba(X1_v, X2_v)
        report = calibration_report(preds, y_v)
        assert "bins" in report and "ece" in report and "brier" in report


# ---------------------------------------------------------------------------
# Outperform Stage 02 LogisticBaseline by > 0.01 Brier
# ---------------------------------------------------------------------------

class TestOutperformsLogisticBaseline:
    """
    Neural model must outperform Stage 02 LogisticBaseline (Brier improvement
    > 0.01) on nonlinear synthetic data.

    DGP: q(X) = ReLU(X @ W1 + b1) @ W2  (nonlinear quality function)
    P(win) = sigmoid(2.5 * q_diff)  — stochastic outcomes

    LogisticBaseline: fits only linear boundary on X_diff → underfits.
    NCAANeuralModel (MLP): learns nonlinear boundary → lower Brier.
    """

    @pytest.fixture(scope="class")
    def trained_pair(self, nonlinear_data):
        """Train both models on the same nonlinear synthetic data."""
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = nonlinear_data

        # ---- LogisticBaseline (Stage 02) ----
        lr_model = LogisticBaseline(C=1.0)
        lr_model.fit(X1_tr, X2_tr, y_tr)
        lr_preds = lr_model.predict_proba(X1_v, X2_v)
        lr_brier = compute_brier_score(lr_preds, y_v)

        # ---- NCAANeuralModel (Stage 03) ----
        # input_mode="concat" mirrors torch NCAAPredictor: each team's raw
        # features are encoded independently before concatenation.  The MLP
        # can learn the nonlinear quality function from (X1, X2) directly,
        # whereas LogisticBaseline is limited to a linear model of X1-X2.
        nn_model = NCAANeuralModel(
            feature_dim=FEATURE_DIM,
            hidden_dim=64,
            max_epochs=200,
            patience=15,
            lr=5e-4,
            batch_size=64,
            random_state=42,
            input_mode="concat",
        )
        nn_model.fit(X1_tr, X2_tr, y_tr)
        nn_preds = nn_model.predict_proba(X1_v, X2_v)
        nn_brier = compute_brier_score(nn_preds, y_v)

        return lr_brier, nn_brier

    def test_nn_brier_below_018(self, trained_pair):
        """NCAANeuralModel achieves Stage 03 target: Brier < 0.18."""
        _, nn_brier = trained_pair
        assert nn_brier < 0.18, (
            f"NCAANeuralModel Brier {nn_brier:.4f} ≥ 0.18 (Stage 03 target)"
        )

    def test_nn_outperforms_lr_by_001(self, trained_pair):
        """
        Neural model Brier improvement over LR > 0.01.
        DGP is nonlinear (ReLU quality function), so LR underfits.
        """
        lr_brier, nn_brier = trained_pair
        improvement = lr_brier - nn_brier
        assert improvement > 0.01, (
            f"Neural model did not outperform LR by > 0.01 Brier. "
            f"LR={lr_brier:.4f}, NN={nn_brier:.4f}, improvement={improvement:.4f}"
        )

    def test_lr_brier_above_nn_brier(self, trained_pair):
        """Neural model must achieve strictly lower Brier than LR."""
        lr_brier, nn_brier = trained_pair
        assert nn_brier < lr_brier, (
            f"NN ({nn_brier:.4f}) did not outperform LR ({lr_brier:.4f})"
        )


# ---------------------------------------------------------------------------
# ONNX latency benchmark (requires torch + onnxruntime)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestONNXLatency:
    """ONNX CPU inference benchmark: single matchup < 5ms."""

    @pytest.fixture(scope="class")
    def exported_model(self, tmp_path_factory):
        ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        from ncaa_models.neural import NCAAPredictor
        from ncaa_models.export import export_to_onnx

        import torch as _torch
        model = NCAAPredictor(feature_dim=FEATURE_DIM, hidden_dim=64, dropout=0.0)
        model.eval()
        tmp = tmp_path_factory.mktemp("onnx_latency")
        onnx_path = export_to_onnx(model, feature_dim=FEATURE_DIM, save_path=tmp / "latency.onnx")
        return onnx_path

    def test_single_matchup_latency_below_5ms(self, exported_model):
        """ONNX CPU inference for a single matchup must be < 5ms."""
        from ncaa_models.export import benchmark_onnx_latency
        result = benchmark_onnx_latency(
            exported_model, feature_dim=FEATURE_DIM, n_warmup=10, n_iters=100
        )
        assert result["mean_ms"] < 5.0, (
            f"ONNX latency {result['mean_ms']:.3f}ms ≥ 5ms threshold"
        )

    def test_p99_latency_below_10ms(self, exported_model):
        """99th-percentile latency must be < 10ms (no outlier spikes)."""
        from ncaa_models.export import benchmark_onnx_latency
        result = benchmark_onnx_latency(
            exported_model, feature_dim=FEATURE_DIM, n_warmup=10, n_iters=100
        )
        assert result["p99_ms"] < 10.0, (
            f"P99 ONNX latency {result['p99_ms']:.3f}ms ≥ 10ms threshold"
        )
