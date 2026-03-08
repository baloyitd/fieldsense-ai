"""
ncaa_models — NCAA Tournament Prediction Models
================================================
Stage 02/10: Logistic regression baseline with reusable evaluation harness.

Exports
-------
MatchupPredictor    : Abstract base class all models must implement.
LogisticBaseline    : Sklearn logistic regression baseline.
compute_brier_score : Standalone Brier score metric.
calibration_report  : Reliability diagram data + ECE.
per_seed_analysis   : Accuracy breakdown by seed matchup type.
per_round_analysis  : Brier score breakdown by tournament round.
build_submission    : Generate Kaggle submission CSV.
temporal_cross_validate : Leave-one-season-out CV with expanding window.
"""

from .base import MatchupPredictor
from .baseline import LogisticBaseline, FEATURE_COLS
from .evaluate import (
    compute_brier_score,
    calibration_report,
    per_seed_analysis,
    per_round_analysis,
)
from .submit import build_submission, validate_submission, make_submission_id
from .cv import temporal_cross_validate, build_matchup_df, CVResult, CVFold

__version__ = "1.0.0"
__stage__ = "02/10"

__all__ = [
    "MatchupPredictor",
    "LogisticBaseline",
    "FEATURE_COLS",
    "compute_brier_score",
    "calibration_report",
    "per_seed_analysis",
    "per_round_analysis",
    "build_submission",
    "validate_submission",
    "make_submission_id",
    "temporal_cross_validate",
    "build_matchup_df",
    "CVResult",
    "CVFold",
]
