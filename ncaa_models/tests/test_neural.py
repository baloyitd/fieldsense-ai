"""
test_neural.py
==============
Unit tests for ncaa_models.neural (NCAAPredictor + NCAANeuralModel).

Torch-specific tests (NCAAPredictor architecture, ONNX equivalence,
gradient health) are skipped when PyTorch is not installed via
``pytest.importorskip("torch")``.

Backend-agnostic tests (MatchupPredictor interface, probability clipping,
symmetry, save/load) run under the sklearn backend without torch.

Unit Tests Required
-------------------
1. Model output shape (batch_size, 1) with values in [0, 1] for random inputs.
2. Shared encoder symmetry: identical inputs → identical embeddings.
3. ONNX export equivalence: max diff < 1e-5 across 100 random samples.
4. Gradient health: no NaN or zero gradients after forward-backward pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_models.baseline import FEATURE_COLS
from ncaa_models.neural import NCAANeuralModel, TORCH_AVAILABLE

# ---------------------------------------------------------------------------
# Torch imports — SKIP entire torch-specific class if torch is absent
# ---------------------------------------------------------------------------

torch = pytest.importorskip(
    "torch",
    reason="PyTorch not installed; NCAAPredictor unit tests skipped.",
) if TORCH_AVAILABLE else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_DIM = len(FEATURE_COLS)   # 28
HIDDEN_DIM = 32                   # small for fast tests
BATCH_SIZE = 8


def _random_pair(n=BATCH_SIZE, feature_dim=FEATURE_DIM, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n, feature_dim)).astype(np.float32),
        rng.standard_normal((n, feature_dim)).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# NCAAPredictor architecture tests (require torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNCAAPredictor:
    """Architecture tests for the torch nn.Module."""

    @pytest.fixture
    def model(self):
        from ncaa_models.neural import NCAAPredictor
        m = NCAAPredictor(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0)
        m.eval()
        return m

    # ----------------------------------------------------------------
    # UNIT TEST 1: output shape + range
    # ----------------------------------------------------------------

    def test_output_shape_batched(self, model):
        """UNIT TEST 1a — output shape (batch_size, 1)."""
        X1 = torch.randn(BATCH_SIZE, FEATURE_DIM)
        X2 = torch.randn(BATCH_SIZE, FEATURE_DIM)
        out = model(X1, X2)
        assert out.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {out.shape}"

    def test_output_values_in_0_1(self, model):
        """UNIT TEST 1b — all outputs in [0, 1]."""
        X1 = torch.randn(100, FEATURE_DIM)
        X2 = torch.randn(100, FEATURE_DIM)
        out = model(X1, X2).squeeze(-1)
        assert (out >= 0.0).all().item(), f"min={out.min().item():.4f}"
        assert (out <= 1.0).all().item(), f"max={out.max().item():.4f}"

    def test_output_shape_single(self, model):
        """Single-sample forward pass produces (1, 1) output."""
        X1 = torch.randn(1, FEATURE_DIM)
        X2 = torch.randn(1, FEATURE_DIM)
        out = model(X1, X2)
        assert out.shape == (1, 1)

    def test_extreme_inputs_finite(self, model):
        """Extreme inputs produce finite (not NaN/Inf) outputs."""
        X1 = torch.full((2, FEATURE_DIM), 1000.0)
        X2 = torch.full((2, FEATURE_DIM), -1000.0)
        out = model(X1, X2)
        assert torch.isfinite(out).all().item()

    # ----------------------------------------------------------------
    # UNIT TEST 2: shared encoder symmetry
    # ----------------------------------------------------------------

    def test_shared_encoder_identical_inputs_same_embedding(self, model):
        """UNIT TEST 2 — identical inputs produce identical embeddings."""
        X = torch.randn(BATCH_SIZE, FEATURE_DIM)
        enc_a = model.encode(X)
        enc_b = model.encode(X)
        torch.testing.assert_close(enc_a, enc_b, atol=1e-6, rtol=0.0)

    def test_shared_encoder_different_inputs_different_embeddings(self, model):
        """Different inputs produce different embeddings (non-trivial encoder)."""
        X1 = torch.randn(4, FEATURE_DIM)
        X2 = X1 + torch.randn(4, FEATURE_DIM) * 2.0
        enc1 = model.encode(X1)
        enc2 = model.encode(X2)
        diff = (enc1 - enc2).abs().max().item()
        assert diff > 0.0, "Encoder produced identical embeddings for different inputs."

    def test_encode_output_shape(self, model):
        """encode() produces (batch_size, hidden_dim) tensor."""
        X = torch.randn(BATCH_SIZE, FEATURE_DIM)
        enc = model.encode(X)
        assert enc.shape == (BATCH_SIZE, HIDDEN_DIM)

    # ----------------------------------------------------------------
    # UNIT TEST 4: gradient health
    # ----------------------------------------------------------------

    def test_no_nan_gradients(self):
        """UNIT TEST 4a — no NaN gradients after backward pass."""
        from ncaa_models.neural import NCAAPredictor
        model = NCAAPredictor(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, dropout=0.1)
        model.train()

        X1 = torch.randn(8, FEATURE_DIM, requires_grad=False)
        X2 = torch.randn(8, FEATURE_DIM, requires_grad=False)
        y = torch.randint(0, 2, (8,)).float()

        preds = model(X1, X2).squeeze(-1)
        loss = torch.nn.BCELoss()(preds, y)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any().item(), f"NaN gradient in {name}"

    def test_no_zero_gradients(self):
        """UNIT TEST 4b — no all-zero gradients (dead neurons check)."""
        from ncaa_models.neural import NCAAPredictor
        model = NCAAPredictor(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0)
        model.train()

        # Use a varied batch to ensure gradients flow
        rng = np.random.default_rng(0)
        X1 = torch.tensor(rng.standard_normal((16, FEATURE_DIM)), dtype=torch.float32)
        X2 = torch.tensor(rng.standard_normal((16, FEATURE_DIM)), dtype=torch.float32)
        y = torch.randint(0, 2, (16,)).float()

        preds = model(X1, X2).squeeze(-1)
        loss = torch.nn.BCELoss()(preds, y)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None
            assert not (param.grad == 0).all().item(), f"All-zero gradient in {name}"

    # ----------------------------------------------------------------
    # Architecture sanity
    # ----------------------------------------------------------------

    def test_weight_initialisation_no_nan(self):
        """Initial weights are finite."""
        from ncaa_models.neural import NCAAPredictor
        model = NCAAPredictor(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0)
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all().item(), f"Non-finite weight in {name}"

    def test_dropout_train_vs_eval_differ(self):
        """Dropout causes train mode outputs to differ from eval mode outputs."""
        from ncaa_models.neural import NCAAPredictor
        model = NCAAPredictor(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, dropout=0.5)
        torch.manual_seed(0)
        X1 = torch.randn(32, FEATURE_DIM)
        X2 = torch.randn(32, FEATURE_DIM)

        model.train()
        out_train = model(X1, X2).detach()
        model.eval()
        out_eval = model(X1, X2).detach()

        # In eval mode, results are deterministic; train vs eval should differ
        assert not torch.allclose(out_train, out_eval, atol=1e-4), (
            "Dropout had no effect — train and eval outputs are identical."
        )


# ---------------------------------------------------------------------------
# UNIT TEST 3: ONNX equivalence (requires torch + onnxruntime)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestONNXEquivalence:
    """UNIT TEST 3: PyTorch vs ONNX outputs must agree within 1e-5."""

    @pytest.fixture(scope="class")
    def onnx_model_and_path(self, tmp_path_factory):
        """Export a small NCAAPredictor and return (pt_model, onnx_path)."""
        onnxruntime = pytest.importorskip(
            "onnxruntime", reason="onnxruntime not installed"
        )
        from ncaa_models.neural import NCAAPredictor
        from ncaa_models.export import export_to_onnx

        model = NCAAPredictor(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0)
        model.eval()

        tmp = tmp_path_factory.mktemp("onnx")
        onnx_path = export_to_onnx(model, feature_dim=FEATURE_DIM, save_path=tmp / "test.onnx")
        return model, onnx_path

    def test_onnx_file_exists(self, onnx_model_and_path):
        _, onnx_path = onnx_model_and_path
        assert onnx_path.exists()

    def test_onnx_max_diff_below_tolerance(self, onnx_model_and_path):
        """UNIT TEST 3 — max absolute diff between PyTorch and ONNX < 1e-5."""
        from ncaa_models.export import verify_onnx_equivalence
        pt_model, onnx_path = onnx_model_and_path
        result = verify_onnx_equivalence(pt_model, onnx_path, FEATURE_DIM, n_samples=100)
        assert result["passed"], (
            f"ONNX equivalence failed: max_diff={result['max_abs_diff']:.2e} >= tol={result['tol']:.2e}"
        )

    def test_onnx_mean_diff_very_small(self, onnx_model_and_path):
        from ncaa_models.export import verify_onnx_equivalence
        pt_model, onnx_path = onnx_model_and_path
        result = verify_onnx_equivalence(pt_model, onnx_path, FEATURE_DIM, n_samples=100)
        assert result["mean_abs_diff"] < 1e-6

    def test_onnx_output_shape_matches(self, onnx_model_and_path):
        """ONNX model output shape matches PyTorch."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort
        import torch as _torch

        pt_model, onnx_path = onnx_model_and_path
        rng = np.random.default_rng(5)
        X1 = rng.standard_normal((10, FEATURE_DIM)).astype(np.float32)
        X2 = rng.standard_normal((10, FEATURE_DIM)).astype(np.float32)

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(["prob"], {"x1": X1, "x2": X2})[0]

        pt_model.eval()
        with _torch.no_grad():
            pt_out = pt_model(
                _torch.tensor(X1), _torch.tensor(X2)
            ).numpy()

        assert ort_out.shape == pt_out.shape


# ---------------------------------------------------------------------------
# NCAANeuralModel — backend-agnostic tests (run without torch)
# ---------------------------------------------------------------------------

class TestNCAANeuralModelInterface:
    """Interface and clipping tests — run on whatever backend is available."""

    @pytest.fixture(scope="class")
    def small_model(self):
        return NCAANeuralModel(
            feature_dim=FEATURE_DIM, hidden_dim=32, max_epochs=20, patience=5
        )

    @pytest.fixture(scope="class")
    def fitted_model(self):
        rng = np.random.default_rng(0)
        n = 50
        X1 = rng.standard_normal((n, FEATURE_DIM)).astype(np.float64)
        X2 = rng.standard_normal((n, FEATURE_DIM)).astype(np.float64)
        y = rng.integers(0, 2, n)
        m = NCAANeuralModel(feature_dim=FEATURE_DIM, hidden_dim=32, max_epochs=20, patience=5)
        m.fit(X1, X2, y)
        return m

    def test_isinstance_matchup_predictor(self, small_model):
        from ncaa_models.base import MatchupPredictor
        assert isinstance(small_model, MatchupPredictor)

    def test_predict_raises_before_fit(self, small_model):
        X1, X2 = _random_pair()
        with pytest.raises(RuntimeError, match="not fitted"):
            small_model.predict_proba(X1, X2)

    def test_fit_returns_self(self):
        rng = np.random.default_rng(1)
        X1 = rng.standard_normal((20, FEATURE_DIM))
        X2 = rng.standard_normal((20, FEATURE_DIM))
        y = rng.integers(0, 2, 20)
        m = NCAANeuralModel(feature_dim=FEATURE_DIM, hidden_dim=16, max_epochs=5, patience=3)
        result = m.fit(X1, X2, y)
        assert result is m

    def test_predict_returns_ndarray(self, fitted_model):
        X1, X2 = _random_pair()
        probs = fitted_model.predict_proba(X1, X2)
        assert isinstance(probs, np.ndarray)

    def test_predict_shape(self, fitted_model):
        X1, X2 = _random_pair(n=12)
        probs = fitted_model.predict_proba(X1, X2)
        assert probs.shape == (12,)

    def test_probs_clipped_to_001_099(self, fitted_model):
        """Probabilities must be in [0.01, 0.99]."""
        X1, X2 = _random_pair(n=50)
        probs = fitted_model.predict_proba(X1, X2)
        assert (probs >= 0.01).all(), f"min={probs.min():.4f}"
        assert (probs <= 0.99).all(), f"max={probs.max():.4f}"

    def test_extreme_inputs_clipped(self, fitted_model):
        """Even extreme feature values produce clipped probabilities."""
        X1 = np.full((2, FEATURE_DIM), 1e6)
        X2 = np.zeros((2, FEATURE_DIM))
        probs = fitted_model.predict_proba(X1, X2)
        assert (probs >= 0.01).all()
        assert (probs <= 0.99).all()

    def test_feature_cols_attribute(self, fitted_model):
        assert fitted_model.feature_cols == FEATURE_COLS

    def test_repr_contains_backend(self, fitted_model):
        r = repr(fitted_model)
        assert "fitted" in r

    def test_save_load_roundtrip(self, fitted_model, tmp_path):
        path = tmp_path / "nn.pkl"
        fitted_model.save(path)
        loaded = NCAANeuralModel.load(path)
        X1, X2 = _random_pair(n=10)
        orig_probs = fitted_model.predict_proba(X1, X2)
        loaded_probs = loaded.predict_proba(X1, X2)
        np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-6)

    def test_save_unfitted_raises(self):
        m = NCAANeuralModel()
        with pytest.raises(RuntimeError):
            m.save("/tmp/unfitted_test.pkl")
