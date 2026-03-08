"""
ncaa_model.evaluate
===================
Model-agnostic evaluation harness — used by ALL stages 02-10.

Protocol
--------
Any model that implements ``NCAAPredictor`` can be plugged in::

    class NCAAPredictor(Protocol):
        feature_cols: list[str]
        def predict_proba_matchup(feat_a, feat_b) -> float: ...

Evaluation functions
--------------------
``evaluate_season``     Evaluate a fitted model on one tournament season.
``cross_validate``      Leave-one-season-out CV given a model factory.
``calibration_bins``    Calibration curve data (reliability diagram).

Metrics returned
----------------
``EvaluationResult``
  * brier_score     : mean squared error between probs and outcomes  ∈ [0, 1]
  * log_loss        : binary cross-entropy
  * accuracy        : fraction of games correctly predicted (p > 0.5 → winner)
  * n_games         : number of tournament games evaluated
  * predictions_df  : per-game DataFrame with columns
                        season, gender, team_id_low, team_id_high,
                        y_true, y_pred, correct
  * calibration     : list of dicts {bin_centre, mean_pred, fraction_pos, n}

Brier score formula (Kaggle standard)
--------------------------------------
  BS = (1/N) * Σ (p_i - y_i)²
  where p_i ∈ [0,1] and y_i ∈ {0,1}.
  Perfect = 0.0, always-0.5 baseline = 0.25.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Callable, Dict, Iterable, List, Optional, Protocol, runtime_checkable,
)

import numpy as np
import pandas as pd

from .matchup import (
    DEFAULT_FEATURE_COLS,
    build_training_data,
    impute_features,
    matchup_vector,
)

logger = logging.getLogger(__name__)

EPSILON = 1e-15   # clip probabilities away from 0/1 for log-loss stability


# ---------------------------------------------------------------------------
# Protocol — every stage-N model must implement this
# ---------------------------------------------------------------------------

@runtime_checkable
class NCAAPredictor(Protocol):
    """
    Duck-type interface for NCAA tournament prediction models.

    All stages (02-10) produce models satisfying this protocol so the
    evaluation harness can be used without modification.
    """

    @property
    def feature_cols(self) -> List[str]:
        """Ordered list of feature names the model was trained on."""
        ...

    def predict_proba_matchup(
        self,
        feat_a: pd.Series,
        feat_b: pd.Series,
    ) -> float:
        """
        Return P(team_a wins) ∈ [0, 1].

        team_a is the **lower raw TeamID** team.
        The symmetry constraint must hold:
            predict_proba_matchup(a, b) == 1 - predict_proba_matchup(b, a)
        """
        ...


# Type alias for a callable that creates and trains a model from scratch.
# Signature: (feat_df, tourney_df, train_seasons) -> NCAAPredictor
ModelFactory = Callable[
    [pd.DataFrame, pd.DataFrame, List[int]],
    NCAAPredictor,
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Metrics produced by evaluating one NCAA tournament prediction model
    on a single season's tournament games.

    Attributes
    ----------
    season : int
        Tournament season evaluated.
    gender : str
        ``'M'``, ``'W'``, or ``'both'``.
    brier_score : float
        Primary Kaggle metric.  Lower is better; target < 0.20.
    log_loss : float
        Binary cross-entropy.
    accuracy : float
        Fraction of games where p > 0.5 correctly predicts the winner.
    n_games : int
        Number of tournament games evaluated.
    predictions_df : pd.DataFrame
        Per-game predictions; columns: season, gender, team_id_low,
        team_id_high, y_true, y_pred, correct.
    calibration : list[dict]
        Reliability diagram data.  Each dict has keys:
        bin_centre, mean_pred, fraction_pos, n.
    """

    season: int
    gender: str
    brier_score: float
    log_loss: float
    accuracy: float
    n_games: int
    predictions_df: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    calibration: List[dict] = field(repr=False, default_factory=list)

    def __str__(self) -> str:
        return (
            f"EvaluationResult({self.gender} {self.season}): "
            f"Brier={self.brier_score:.4f}  LogLoss={self.log_loss:.4f}  "
            f"Acc={self.accuracy:.3f}  n={self.n_games}"
        )


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Brier score.

    Parameters
    ----------
    y_true : array-like of int  (0 or 1)
    y_pred : array-like of float  ∈ [0, 1]

    Returns
    -------
    float  ∈ [0, 1]  (lower is better)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute binary cross-entropy log-loss.

    Parameters
    ----------
    y_true : array-like of int  (0 or 1)
    y_pred : array-like of float  ∈ [0, 1]

    Returns
    -------
    float  (lower is better)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), EPSILON, 1 - EPSILON)
    return float(-np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    ))


def calibration_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> List[dict]:
    """
    Compute calibration curve data (reliability diagram).

    Parameters
    ----------
    y_true  : array-like of int   (0 or 1)
    y_pred  : array-like of float  ∈ [0, 1]
    n_bins  : number of equal-width probability bins

    Returns
    -------
    list of dict with keys: bin_centre, mean_pred, fraction_pos, n
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if lo == 0:
            mask |= y_pred == 0.0
        if hi == 1.0:
            mask |= y_pred == 1.0
        n = int(mask.sum())
        if n == 0:
            continue
        result.append({
            "bin_centre": float((lo + hi) / 2),
            "mean_pred": float(y_pred[mask].mean()),
            "fraction_pos": float(y_true[mask].mean()),
            "n": n,
        })
    return result


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_season(
    model: NCAAPredictor,
    feat_df: pd.DataFrame,
    tourney_df: pd.DataFrame,
    val_season: int,
    gender: str = "both",
    n_calibration_bins: int = 10,
) -> EvaluationResult:
    """
    Evaluate a fitted model on one tournament season.

    Parameters
    ----------
    model : NCAAPredictor
        Fitted model implementing ``predict_proba_matchup``.
    feat_df : pd.DataFrame
        Team-season features (output of ``compute_features``).
    tourney_df : pd.DataFrame
        All normalised tournament rows (``is_tourney=True`` subset of
        ``normalize_all`` output).
    val_season : int
        Tournament season to evaluate (e.g. 2025).
    gender : str
        ``'M'``, ``'W'``, or ``'both'`` to filter tournament games.
    n_calibration_bins : int
        Number of bins for reliability diagram.

    Returns
    -------
    EvaluationResult
    """
    feat_imp = impute_features(feat_df, model.feature_cols)
    feat_index = feat_imp.set_index(["gender", "season", "canonical_id"])[model.feature_cols]

    # Filter tournament games for this season
    mask = tourney_df["season"] == val_season
    if gender != "both":
        mask &= tourney_df["gender"] == gender
    tourn = tourney_df[mask & tourney_df["won"]].copy()   # de-duplicate

    if tourn.empty:
        raise ValueError(
            f"No tournament games found for season={val_season}, gender={gender}."
        )

    y_true_list, y_pred_list, pred_rows = [], [], []

    for _, game in tourn.iterrows():
        g      = str(game["gender"])
        cid_w  = str(game["canonical_id"])
        cid_l  = str(game["opp_canonical_id"])
        tid_w  = int(game["team_id"])
        tid_l  = int(game["opp_team_id"])

        key_w = (g, val_season, cid_w)
        key_l = (g, val_season, cid_l)

        if key_w not in feat_index.index or key_l not in feat_index.index:
            logger.debug("Missing features for %s vs %s; skipping.", cid_w, cid_l)
            continue

        feat_w = feat_index.loc[key_w]
        feat_l = feat_index.loc[key_l]

        # Lower-ID convention
        if tid_w <= tid_l:
            tid_low, tid_high = tid_w, tid_l
            feat_a, feat_b = feat_w, feat_l
            y_true = 1   # lower-ID team (winner) won
        else:
            tid_low, tid_high = tid_l, tid_w
            feat_a, feat_b = feat_l, feat_w
            y_true = 0   # lower-ID team (loser) lost

        y_pred = model.predict_proba_matchup(feat_a, feat_b)
        correct = int((y_pred > 0.5) == bool(y_true))

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        pred_rows.append({
            "season": val_season, "gender": g,
            "team_id_low": tid_low, "team_id_high": tid_high,
            "y_true": y_true, "y_pred": y_pred, "correct": correct,
        })

    if not y_true_list:
        raise ValueError(
            f"All games skipped due to missing features for season {val_season}."
        )

    y_true_arr = np.array(y_true_list, dtype=float)
    y_pred_arr = np.array(y_pred_list, dtype=float)

    result = EvaluationResult(
        season=val_season,
        gender=gender,
        brier_score=brier_score(y_true_arr, y_pred_arr),
        log_loss=log_loss(y_true_arr, y_pred_arr),
        accuracy=float(np.mean([r["correct"] for r in pred_rows])),
        n_games=len(y_true_list),
        predictions_df=pd.DataFrame(pred_rows),
        calibration=calibration_bins(y_true_arr, y_pred_arr, n_calibration_bins),
    )

    logger.info(str(result))
    return result


# ---------------------------------------------------------------------------
# Cross-validation harness
# ---------------------------------------------------------------------------

def cross_validate(
    model_factory: ModelFactory,
    feat_df: pd.DataFrame,
    tourney_df: pd.DataFrame,
    val_seasons: Iterable[int],
    train_lookback: int = 4,
    gender: str = "both",
) -> Dict[int, EvaluationResult]:
    """
    Leave-one-season-out cross-validation.

    For each validation season, trains on the preceding *train_lookback*
    seasons (or all available earlier seasons, whichever is smaller) and
    evaluates on the validation season.

    Parameters
    ----------
    model_factory : callable
        ``(feat_df, tourney_df, train_seasons) -> NCAAPredictor``
        Creates and trains a fresh model.
    feat_df : pd.DataFrame
        Full team-season features.
    tourney_df : pd.DataFrame
        Full normalised tournament rows.
    val_seasons : iterable of int
        Seasons to use as validation holdout (e.g. ``[2023, 2024, 2025]``).
    train_lookback : int
        Maximum number of seasons to look back for training data.
    gender : str
        ``'M'``, ``'W'``, or ``'both'``.

    Returns
    -------
    dict[int, EvaluationResult]
        One result per validation season.
    """
    all_seasons = sorted(tourney_df["season"].unique().tolist())
    results: Dict[int, EvaluationResult] = {}

    for val_season in sorted(val_seasons):
        earlier = [s for s in all_seasons if s < val_season]
        train_seasons = earlier[-train_lookback:] if earlier else []

        if not train_seasons:
            logger.warning("No training seasons available before %d; skipping.", val_season)
            continue

        logger.info(
            "CV fold: val=%d  train=%s", val_season, train_seasons
        )
        model = model_factory(feat_df, tourney_df, train_seasons)
        try:
            result = evaluate_season(model, feat_df, tourney_df, val_season, gender)
            results[val_season] = result
        except ValueError as exc:
            logger.warning("Could not evaluate season %d: %s", val_season, exc)

    return results


def summarise_cv(results: Dict[int, EvaluationResult]) -> pd.DataFrame:
    """
    Summarise cross-validation results into a tidy DataFrame.

    Parameters
    ----------
    results : dict[int, EvaluationResult]
        Output of :func:`cross_validate`.

    Returns
    -------
    pd.DataFrame  columns: season, brier_score, log_loss, accuracy, n_games
    """
    rows = []
    for season, r in sorted(results.items()):
        rows.append({
            "season": season,
            "brier_score": r.brier_score,
            "log_loss": r.log_loss,
            "accuracy": r.accuracy,
            "n_games": r.n_games,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        agg = pd.DataFrame([{
            "season": "mean",
            "brier_score": df["brier_score"].mean(),
            "log_loss": df["log_loss"].mean(),
            "accuracy": df["accuracy"].mean(),
            "n_games": df["n_games"].sum(),
        }])
        df = pd.concat([df, agg], ignore_index=True)
    return df
