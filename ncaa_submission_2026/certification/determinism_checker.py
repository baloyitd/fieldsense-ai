"""
ncaa_submission_2026.certification.determinism_checker
=======================================================
Stage 09 — Deterministic inference certification.

Verifies that running the prediction pipeline twice with identical inputs
and seeds produces bitwise-identical output arrays.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (if available).

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed; skip


@dataclass
class DeterminismReport:
    """Result of a determinism check."""

    is_deterministic: bool
    n_runs: int
    n_comparisons: int
    n_mismatches: int
    max_abs_diff: float
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_deterministic": self.is_deterministic,
            "n_runs": self.n_runs,
            "n_comparisons": self.n_comparisons,
            "n_mismatches": self.n_mismatches,
            "max_abs_diff": self.max_abs_diff,
            "details": self.details,
        }


class DeterminismChecker:
    """
    Verifies that a model's predictions are bitwise-identical across runs.

    Parameters
    ----------
    seed : int
        Random seed set before each run.
    n_runs : int
        Number of prediction runs.  All must match run 0.
    """

    def __init__(self, seed: int = 42, n_runs: int = 2) -> None:
        self.seed = seed
        self.n_runs = n_runs

    def check_model(
        self,
        model,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> DeterminismReport:
        """
        Run ``model.predict_proba(X1, X2)`` *n_runs* times and compare.

        Parameters
        ----------
        model : MatchupPredictor
        X1, X2 : np.ndarray

        Returns
        -------
        DeterminismReport
        """
        outputs: List[np.ndarray] = []
        for _ in range(self.n_runs):
            set_all_seeds(self.seed)
            probs = np.asarray(model.predict_proba(X1, X2), dtype=float)
            outputs.append(probs)

        return self._compare_outputs(outputs)

    def check_function(
        self,
        func: Callable[..., np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> DeterminismReport:
        """
        Run *func* *n_runs* times with identical args and compare outputs.

        Parameters
        ----------
        func : callable
            Must return a numpy-convertible array.
        *args, **kwargs : passed to func unchanged.

        Returns
        -------
        DeterminismReport
        """
        outputs: List[np.ndarray] = []
        for _ in range(self.n_runs):
            set_all_seeds(self.seed)
            result = np.asarray(func(*args, **kwargs), dtype=float)
            outputs.append(result)

        return self._compare_outputs(outputs)

    def check_csv(
        self,
        csv_func: Callable[[], str],
    ) -> DeterminismReport:
        """
        Run *csv_func* *n_runs* times and compare CSV content as arrays.

        Parameters
        ----------
        csv_func : callable
            Returns CSV text string with 'ID,Pred' format.

        Returns
        -------
        DeterminismReport
        """
        import io
        import pandas as pd

        outputs: List[np.ndarray] = []
        for _ in range(self.n_runs):
            set_all_seeds(self.seed)
            csv_text = csv_func()
            df = pd.read_csv(io.StringIO(csv_text)).sort_values("ID")
            outputs.append(df["Pred"].values.astype(float))

        return self._compare_outputs(outputs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compare_outputs(self, outputs: List[np.ndarray]) -> DeterminismReport:
        """Compare a list of output arrays for bitwise equality."""
        n_mismatches = 0
        max_diff = 0.0
        ref = outputs[0]
        n_comparisons = len(outputs) - 1

        for i, other in enumerate(outputs[1:], start=1):
            if ref.shape != other.shape:
                return DeterminismReport(
                    is_deterministic=False,
                    n_runs=self.n_runs,
                    n_comparisons=n_comparisons,
                    n_mismatches=1,
                    max_abs_diff=float("inf"),
                    details=f"Run {i} output shape {other.shape} differs from ref {ref.shape}.",
                )
            mismatch_mask = ref != other
            n_mm = int(mismatch_mask.sum())
            n_mismatches += n_mm
            if n_mm > 0:
                max_diff = max(max_diff, float(np.abs(ref[mismatch_mask] - other[mismatch_mask]).max()))

        is_det = n_mismatches == 0
        details = (
            "All runs bitwise-identical." if is_det
            else f"{n_mismatches} element mismatches; max |diff|={max_diff:.2e}."
        )
        report = DeterminismReport(
            is_deterministic=is_det,
            n_runs=self.n_runs,
            n_comparisons=n_comparisons,
            n_mismatches=n_mismatches,
            max_abs_diff=max_diff,
            details=details,
        )
        if is_det:
            logger.info("Determinism check PASSED: %s", details)
        else:
            logger.warning("Determinism check FAILED: %s", details)
        return report
