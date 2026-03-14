"""
ncaa_models.neural
==================
Stage 03 — Feed-forward neural network for NCAA tournament prediction.

Architecture
------------
NCAAPredictor (torch.nn.Module):
  team_encoder  : Linear(feature_dim, hidden_dim) → ReLU → Dropout
                  → Linear(hidden_dim, hidden_dim) → ReLU
  matchup_net   : Linear(hidden_dim*2, hidden_dim) → ReLU
                  → Linear(hidden_dim, 1) → Sigmoid

  Both teams pass through the SAME shared encoder weights.
  Encodings are concatenated [enc1, enc2] before matchup_net.
  Output: P(team1 wins) as scalar in [0, 1].

  Kaggle convention: team1 = lower TeamId.
  Team-order augmentation during training ensures approximate symmetry.

NCAANeuralModel (MatchupPredictor wrapper):
  Training backend: PyTorch (preferred) or sklearn MLPClassifier (fallback).
  Probabilities clipped to [0.01, 0.99].

ONNX export: via ncaa_models.export.export_to_onnx().
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .base import MatchupPredictor
from .baseline import FEATURE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch availability guard
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]

PROB_CLIP_LOW: float = 0.01
PROB_CLIP_HIGH: float = 0.99

# Default feature dimension = number of Stage 01 features
_DEFAULT_FEATURE_DIM: int = len(FEATURE_COLS)


# ---------------------------------------------------------------------------
# NCAAPredictor — torch nn.Module
# ---------------------------------------------------------------------------

# Conditional base class so the module imports cleanly without torch.
_TorchBase = (nn.Module if TORCH_AVAILABLE else object)  # type: ignore[misc]


class NCAAPredictor(_TorchBase):  # type: ignore[misc]
    """
    Shared-encoder + matchup-comparison feed-forward network.

    Parameters
    ----------
    feature_dim : int
        Input feature dimension per team. Default 50 (override with
        len(FEATURE_COLS)=28 for the Stage 01 pipeline).
    hidden_dim  : int
        Hidden dimension for encoder and matchup_net. Default 64.
    dropout     : float
        Dropout probability in the encoder. Default 0.3.

    Architecture
    ------------
    team_encoder : Linear(feature_dim, hidden_dim) → ReLU
                   → Dropout(dropout)
                   → Linear(hidden_dim, hidden_dim) → ReLU

    matchup_net  : Linear(hidden_dim*2, hidden_dim) → ReLU
                   → Linear(hidden_dim, 1) → Sigmoid

    Forward pass
    ------------
    enc1     = team_encoder(x1)              # (B, hidden_dim)
    enc2     = team_encoder(x2)              # (B, hidden_dim) — SAME weights
    combined = cat([enc1, enc2], dim=-1)     # (B, hidden_dim*2)
    output   = matchup_net(combined)         # (B, 1)  in (0, 1)
    """

    def __init__(
        self,
        feature_dim: int = 50,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for NCAAPredictor. "
                "Install with: pip install torch"
            )
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        # Shared team encoder (both teams use identical weights)
        self.team_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Matchup comparison head
        self.matchup_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He (Kaiming) initialisation for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return team encoding (hidden_dim vector) for a batch of feature vectors."""
        return self.team_encoder(x)

    def forward(
        self,
        x1: "torch.Tensor",
        x2: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute P(team1 wins) for each matchup.

        Parameters
        ----------
        x1, x2 : Tensor, shape (batch_size, feature_dim)

        Returns
        -------
        Tensor, shape (batch_size, 1), values in (0, 1).
        """
        enc1 = self.team_encoder(x1)               # (B, hidden_dim)
        enc2 = self.team_encoder(x2)               # (B, hidden_dim)
        combined = torch.cat([enc1, enc2], dim=-1)  # (B, hidden_dim*2)
        return self.matchup_net(combined)           # (B, 1)


# ---------------------------------------------------------------------------
# NCAANeuralModel — MatchupPredictor wrapper
# ---------------------------------------------------------------------------

class NCAANeuralModel(MatchupPredictor):
    """
    MatchupPredictor wrapper for the NCAAPredictor architecture.

    Training backend is automatically selected:
    - **PyTorch** (preferred): trains NCAAPredictor with Adam + early stopping.
    - **sklearn MLP** (fallback when torch not installed): uses
      MLPClassifier with feature differencing (X1 - X2) as input.

    Both backends clip output probabilities to [0.01, 0.99].

    Parameters
    ----------
    feature_dim  : int   Feature dimension. Defaults to len(FEATURE_COLS)=28.
    hidden_dim   : int   Hidden dimension. Default 64.
    dropout      : float Dropout probability (torch backend only). Default 0.3.
    lr           : float Learning rate. Default 1e-3.
    weight_decay : float Adam weight decay. Default 1e-4.
    batch_size   : int   Mini-batch size. Default 64.
    max_epochs   : int   Maximum training epochs. Default 200.
    patience     : int   Early-stopping patience on validation Brier. Default 10.
    device       : str   Torch device string. Default 'cpu'.
    random_state : int   Seed for reproducibility. Default 42.
    feature_cols : list  Feature column names (for metadata). Defaults to FEATURE_COLS.
    """

    def __init__(
        self,
        feature_dim: int = _DEFAULT_FEATURE_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 10,
        device: str = "cpu",
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        input_mode: str = "diff",
    ) -> None:
        """
        Parameters
        ----------
        input_mode : str
            'diff'   — model input is X1 - X2 (symmetric, default).
                       Mirrors LogisticBaseline and the shared-encoder
                       linear projection.
            'concat' — model input is [X1, X2] concatenated (2*feature_dim).
                       Matches the torch NCAAPredictor's approach: each team's
                       raw features are passed through the shared encoder
                       independently before concatenation.
        """
        if input_mode not in ("diff", "concat"):
            raise ValueError(f"input_mode must be 'diff' or 'concat', got '{input_mode}'")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.input_mode = input_mode
        self.feature_cols: List[str] = (
            list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
        )
        self._is_fitted: bool = False
        self._backend: str = "torch" if TORCH_AVAILABLE else "sklearn"
        self._history: dict = {}

        # Backend state
        self._torch_model: Optional["NCAAPredictor"] = None
        self._scaler: Optional[StandardScaler] = None
        self._mlp: Optional[MLPClassifier] = None

    # ------------------------------------------------------------------
    # MatchupPredictor interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_team1: np.ndarray,
        X_team2: np.ndarray,
        y: np.ndarray,
        X_val1: Optional[np.ndarray] = None,
        X_val2: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "NCAANeuralModel":
        """
        Train the model on matchup feature arrays.

        Parameters
        ----------
        X_team1, X_team2 : np.ndarray, shape (n_matchups, feature_dim)
            Raw per-team feature vectors. team1 = lower TeamId.
        y : np.ndarray, shape (n_matchups,)
            y=1 if team1 wins, y=0 if team2 wins.
        X_val1, X_val2, y_val : optional validation arrays for early stopping.
        """
        if self._backend == "torch":
            self._fit_torch(X_team1, X_team2, y, X_val1, X_val2, y_val)
        else:
            self._fit_sklearn(X_team1, X_team2, y)
        self._is_fitted = True
        return self

    def predict_proba(
        self,
        X_team1: np.ndarray,
        X_team2: np.ndarray,
    ) -> np.ndarray:
        """
        Return P(team1 wins) clipped to [0.01, 0.99].

        Parameters
        ----------
        X_team1, X_team2 : np.ndarray, shape (n_matchups, feature_dim)

        Returns
        -------
        np.ndarray, shape (n_matchups,), values in [0.01, 0.99].
        """
        self._check_fitted()
        if self._backend == "torch":
            raw = self._predict_torch(X_team1, X_team2)
        else:
            raw = self._predict_sklearn(X_team1, X_team2)
        return np.clip(raw, PROB_CLIP_LOW, PROB_CLIP_HIGH)

    # ------------------------------------------------------------------
    # Torch backend
    # ------------------------------------------------------------------

    def _fit_torch(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
        X_val1: Optional[np.ndarray],
        X_val2: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> None:
        from .training import train_model

        self._torch_model = NCAAPredictor(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        val_data = (
            (X_val1, X_val2, y_val)
            if (X_val1 is not None and y_val is not None)
            else None
        )
        self._history = train_model(
            model=self._torch_model,
            X1_train=X1,
            X2_train=X2,
            y_train=y,
            val_data=val_data,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            device=self.device,
            random_state=self.random_state,
        )
        self._torch_model.eval()
        logger.info(
            "NCAANeuralModel (torch) fitted. Best val Brier: %.4f",
            self._history.get("best_val_brier", float("nan")),
        )

    def _predict_torch(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        self._torch_model.eval()
        with torch.no_grad():
            t1 = torch.tensor(X1, dtype=torch.float32, device=self.device)
            t2 = torch.tensor(X2, dtype=torch.float32, device=self.device)
            probs = self._torch_model(t1, t2).squeeze(-1).cpu().numpy()
        return probs.astype(float)

    # ------------------------------------------------------------------
    # Sklearn fallback backend
    # ------------------------------------------------------------------

    def _fit_sklearn(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Sklearn MLPClassifier fallback.

        input_mode='diff'   : X_diff = X1 - X2 (symmetric linear encoding).
        input_mode='concat' : X = [X1, X2] concatenated (2*feature_dim).
                              Mirrors the torch NCAAPredictor's per-team encoding.
        """
        # Apply team-swap augmentation (mirrors torch training augmentation)
        X1_aug, X2_aug, y_aug = _augment_matchups(
            X1, X2, y, seed=self.random_state
        )

        if self.input_mode == "concat":
            X_input = np.concatenate([X1_aug, X2_aug], axis=1)
        else:
            X_input = X1_aug - X2_aug

        self._scaler = StandardScaler(with_mean=False)
        X_scaled = self._scaler.fit_transform(X_input)

        effective_batch = min(self.batch_size, len(y_aug))
        # sklearn early stopping uses an internal 10% split
        n_iter_no_change = max(5, self.patience)

        self._mlp = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim),
            activation="relu",
            solver="adam",
            learning_rate_init=self.lr,
            alpha=self.weight_decay,
            max_iter=self.max_epochs,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=n_iter_no_change,
            batch_size=effective_batch,
            random_state=self.random_state,
            verbose=False,
        )
        self._mlp.fit(X_scaled, y_aug.astype(int))
        self._history = {
            "n_iter": self._mlp.n_iter_,
            "best_val_score": float(self._mlp.best_validation_score_),
        }
        logger.info(
            "NCAANeuralModel (sklearn) fitted in %d iterations. "
            "Best validation score: %.4f",
            self._mlp.n_iter_,
            self._mlp.best_validation_score_,
        )

    def _predict_sklearn(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.input_mode == "concat":
            X_input = np.concatenate([X1, X2], axis=1)
        else:
            X_input = X1 - X2
        X_scaled = self._scaler.transform(X_input)
        return self._mlp.predict_proba(X_scaled)[:, 1].astype(float)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist model (torch state_dict or sklearn pickle)."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "backend": self._backend,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "device": self.device,
            "random_state": self.random_state,
            "feature_cols": self.feature_cols,
            "input_mode": self.input_mode,
            "history": self._history,
        }
        if self._backend == "torch":
            payload["state_dict"] = self._torch_model.state_dict()
        else:
            payload["scaler"] = self._scaler
            payload["mlp"] = self._mlp

        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("NCAANeuralModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "NCAANeuralModel":
        """Load a model saved with save()."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            feature_dim=payload["feature_dim"],
            hidden_dim=payload["hidden_dim"],
            dropout=payload["dropout"],
            lr=payload["lr"],
            weight_decay=payload["weight_decay"],
            batch_size=payload["batch_size"],
            max_epochs=payload["max_epochs"],
            patience=payload["patience"],
            device=payload["device"],
            random_state=payload["random_state"],
            feature_cols=payload["feature_cols"],
            input_mode=payload.get("input_mode", "diff"),
        )
        obj._backend = payload["backend"]
        obj._history = payload.get("history", {})

        if payload["backend"] == "torch":
            obj._torch_model = NCAAPredictor(
                feature_dim=obj.feature_dim,
                hidden_dim=obj.hidden_dim,
                dropout=obj.dropout,
            )
            obj._torch_model.load_state_dict(payload["state_dict"])
            obj._torch_model.eval()
        else:
            obj._scaler = payload["scaler"]
            obj._mlp = payload["mlp"]

        obj._is_fitted = True
        logger.info("NCAANeuralModel loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Torch model accessor (for ONNX export etc.)
    # ------------------------------------------------------------------

    @property
    def torch_model(self) -> "NCAAPredictor":
        """Return the underlying NCAAPredictor (torch backend only)."""
        if self._backend != "torch" or self._torch_model is None:
            raise RuntimeError(
                "torch_model is only available when the torch backend is used."
            )
        return self._torch_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() before predict_proba()."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"NCAANeuralModel(backend={self._backend}, "
            f"feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim}, "
            f"{status})"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _augment_matchups(
    X1: np.ndarray,
    X2: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Team-swap augmentation: for each matchup, with 50% probability swap
    (X1, X2, y=1) → (X2, X1, y=0) and vice versa.

    This prevents the model from learning order-of-input biases and
    enforces approximate P(A,B) + P(B,A) = 1 symmetry.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(len(y)) < 0.5
    X1_out = np.where(mask[:, None], X2, X1)
    X2_out = np.where(mask[:, None], X1, X2)
    y_out = np.where(mask, 1 - y, y)
    return X1_out, X2_out, y_out.astype(y.dtype)
