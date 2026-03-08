"""
ncaa_models.evaluate
====================
Model-agnostic evaluation harness — reusable by any MatchupPredictor.

Functions
---------
compute_brier_score   : Mean squared error between predictions and actuals.
calibration_report    : Per-bin calibration stats + ECE + reliability diagram.
per_seed_analysis     : Accuracy and Brier breakdown by seed matchup type.
per_round_analysis    : Brier score and accuracy breakdown by tournament round.

Brier score formula (Kaggle standard)
--------------------------------------
  BS = (1/N) * Σ (p_i - y_i)²
  Perfect = 0.0, always-0.5 baseline = 0.25, target < 0.20.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

EPSILON = 1e-15  # clip floor for log-loss stability


# ---------------------------------------------------------------------------
# Primary metric
# ---------------------------------------------------------------------------

def compute_brier_score(
    predictions: Sequence[float],
    actuals: Sequence[int],
) -> float:
    """
    Compute the Brier score: mean((p - y)^2).

    Parameters
    ----------
    predictions : array-like of float in [0, 1]
    actuals     : array-like of int in {0, 1}

    Returns
    -------
    float  (lower is better; perfect=0.0, always-0.5 baseline=0.25)

    Examples
    --------
    >>> compute_brier_score([0.8, 0.3], [1, 0])
    0.065
    """
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(actuals, dtype=float)
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_report(
    predictions: Sequence[float],
    actuals: Sequence[int],
    n_bins: int = 10,
    save_json: Optional[str | Path] = None,
    save_png: Optional[str | Path] = None,
) -> Dict:
    """
    Compute calibration curve (reliability diagram) data.

    Parameters
    ----------
    predictions : array-like of float in [0, 1]
    actuals     : array-like of int in {0, 1}
    n_bins      : number of equal-width probability bins
    save_json   : optional path to write JSON report
    save_png    : optional path to write reliability diagram PNG

    Returns
    -------
    dict with keys:
        bins  : list of dicts {bin_centre, mean_pred, fraction_pos, n}
        ece   : Expected Calibration Error (weighted mean absolute calibration error)
        brier : overall Brier score
    """
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(actuals, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict] = []
    weighted_abs_errors: List[float] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p >= lo) & (p < hi)
        if lo == 0.0:
            mask |= p == 0.0
        if hi == 1.0:
            mask |= p == 1.0
        n = int(mask.sum())
        if n == 0:
            continue

        mean_pred = float(p[mask].mean())
        frac_pos = float(y[mask].mean())
        bins.append({
            "bin_centre": float((lo + hi) / 2),
            "mean_pred": mean_pred,
            "fraction_pos": frac_pos,
            "n": n,
        })
        weighted_abs_errors.append(abs(mean_pred - frac_pos) * n)

    ece = float(sum(weighted_abs_errors) / len(p)) if len(p) > 0 else 0.0
    report = {
        "bins": bins,
        "ece": ece,
        "brier": compute_brier_score(p, y),
    }

    if save_json is not None:
        _save_json(report, Path(save_json), "calibration report")

    if save_png is not None:
        _save_reliability_diagram(bins, Path(save_png))

    return report


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

def per_seed_analysis(
    predictions: Sequence[float],
    actuals: Sequence[int],
    seeds_team1: Sequence[int],
    seeds_team2: Sequence[int],
    save_json: Optional[str | Path] = None,
) -> Dict:
    """
    Accuracy and Brier score breakdown by seed matchup category.

    Categories
    ----------
    favourite : lower-seeded team (expected winner) is team1 (seed1 < seed2)
    upset     : higher-seeded team (underdog) is team1 (seed1 > seed2)
    equal_seed: same seed number for both teams

    Also reports breakdown by absolute seed differential quintile.

    Parameters
    ----------
    predictions   : array-like of float in [0, 1]  — P(team1 wins)
    actuals       : array-like of int in {0, 1}
    seeds_team1   : seed numbers for team1 (1=best)
    seeds_team2   : seed numbers for team2

    Returns
    -------
    dict with category keys each containing {n, accuracy, brier, mean_pred, fraction_pos}
    and a 'by_seed_diff_quintile' list.
    """
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(actuals, dtype=float)
    s1 = np.asarray(seeds_team1, dtype=int)
    s2 = np.asarray(seeds_team2, dtype=int)
    seed_diff = s1 - s2  # negative = team1 is better-seeded (favourite)

    result: Dict = {}

    for label, mask in [
        ("favourite", seed_diff < 0),   # team1 is better seed
        ("upset", seed_diff > 0),        # team1 is worse seed
        ("equal_seed", seed_diff == 0),
    ]:
        n = int(mask.sum())
        if n == 0:
            continue
        result[label] = {
            "n": n,
            "accuracy": float(np.mean((p[mask] > 0.5) == y[mask])),
            "brier": compute_brier_score(p[mask], y[mask]),
            "mean_pred": float(p[mask].mean()),
            "fraction_pos": float(y[mask].mean()),
        }

    # Breakdown by |seed diff| quintile
    abs_diff = np.abs(seed_diff)
    if len(abs_diff) > 0:
        quintile_edges = np.percentile(abs_diff, [0, 20, 40, 60, 80, 100])
        # deduplicate edges
        edges = sorted(set(quintile_edges))
        by_quintile = []
        for q_lo, q_hi in zip(edges[:-1], edges[1:]):
            mask = (abs_diff >= q_lo) & (abs_diff <= q_hi)
            n = int(mask.sum())
            if n == 0:
                continue
            by_quintile.append({
                "seed_diff_range": [float(q_lo), float(q_hi)],
                "n": n,
                "accuracy": float(np.mean((p[mask] > 0.5) == y[mask])),
                "brier": compute_brier_score(p[mask], y[mask]),
            })
        result["by_seed_diff_quintile"] = by_quintile

    if save_json is not None:
        _save_json(result, Path(save_json), "per-seed analysis")

    return result


# ---------------------------------------------------------------------------
# Round analysis
# ---------------------------------------------------------------------------

def per_round_analysis(
    predictions: Sequence[float],
    actuals: Sequence[int],
    rounds: Sequence[int],
    save_json: Optional[str | Path] = None,
) -> Dict:
    """
    Brier score and accuracy breakdown by tournament round.

    Parameters
    ----------
    predictions : array-like of float
    actuals     : array-like of int
    rounds      : array-like of int  (e.g. 1=Round of 64, 2=Round of 32, ...)
    save_json   : optional path to write JSON

    Returns
    -------
    dict mapping round (int) → {n, brier, accuracy}
    """
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(actuals, dtype=float)
    r = np.asarray(rounds, dtype=int)

    result: Dict = {}
    for rnd in sorted(np.unique(r)):
        mask = r == rnd
        result[int(rnd)] = {
            "n": int(mask.sum()),
            "brier": compute_brier_score(p[mask], y[mask]),
            "accuracy": float(np.mean((p[mask] > 0.5) == y[mask])),
        }

    if save_json is not None:
        # JSON keys must be strings
        _save_json({str(k): v for k, v in result.items()}, Path(save_json), "per-round analysis")

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_json(data: Dict, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("%s saved to %s", label, path)


def _save_reliability_diagram(bins: List[Dict], path: Path) -> None:
    """Save reliability diagram PNG using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping reliability diagram PNG.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    mean_preds = [b["mean_pred"] for b in bins]
    frac_pos = [b["fraction_pos"] for b in bins]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(mean_preds, frac_pos, zorder=5)
    ax.plot(mean_preds, frac_pos, alpha=0.6, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    logger.info("Reliability diagram saved to %s", path)
