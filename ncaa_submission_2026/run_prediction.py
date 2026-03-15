#!/usr/bin/env python3
"""
run_prediction.py — FieldSense NCAA 2026 Prediction Entry Point
===============================================================

Usage
-----
::

    python run_prediction.py \\
        --data-dir ./kaggle_data \\
        --output submission.csv \\
        --season 2026 \\
        --seed 42

    # Without real data (uses synthetic 8-team dataset):
    python run_prediction.py --synthetic --output test_submission.csv

Arguments
---------
--data-dir   Directory containing Kaggle CSV data files.
--output     Output CSV path (default: submission.csv).
--season     Prediction season (default: 2026).
--seed       Random seed for determinism (default: 42).
--synthetic  Use synthetic 8-team data (for testing without Kaggle data).
--certify    Run certification checks before prediction.

Exit Codes
----------
0  Success
1  Prediction failed
2  Certification failed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_prediction")

# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = 42) -> None:
    """Set Python, NumPy, and (if available) PyTorch seeds."""
    import random
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
        pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kaggle_data(data_dir: Path, season: int) -> tuple:
    """
    Load team features and team IDs from Kaggle data directory.

    Expects files: MTeams.csv, MRegularSeasonDetailedResults.csv
    (or WTeams.csv / WRegularSeasonDetailedResults.csv).

    Returns
    -------
    (feat_df, team_ids)
    """
    from ncaa_models.baseline import FEATURE_COLS
    # This is a stub implementation; a full production system would
    # parse the Kaggle files.  For reproducibility testing we raise
    # a descriptive error if the files are not found.
    teams_path = data_dir / "MTeams.csv"
    if not teams_path.exists():
        raise FileNotFoundError(
            f"Kaggle data not found at {data_dir}. "
            "Use --synthetic for testing without Kaggle data."
        )
    raise NotImplementedError(
        "Full Kaggle data loading is not yet implemented. "
        "Use --synthetic for testing."
    )


def load_synthetic_data(season: int = 2026) -> tuple:
    """
    Generate synthetic 8-team tournament data for testing.

    Returns
    -------
    (feat_df, team_ids, train_X1, train_X2, train_y)
    """
    import sys
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ncaa_models.baseline import FEATURE_COLS, LogisticBaseline
    # Import helpers from ncaa_models test suite
    ncaa_test_dir = Path(__file__).parent.parent / "ncaa_models" / "tests"
    sys.path.insert(0, str(ncaa_test_dir))
    from conftest import (
        make_feat_df, make_tourney_df,
        M_TEAM_IDS, TRAIN_SEASONS, ALL_SEASONS,
    )
    from ncaa_models.cv import build_matchup_df

    all_seasons = TRAIN_SEASONS + [season - 1]  # use last available year
    feat_df = make_feat_df(M_TEAM_IDS, ALL_SEASONS)
    # For season 2026, create an extra copy of the 2025 season features
    season_df = feat_df[feat_df["season"] == 2025].copy()
    season_df["season"] = season
    feat_df = pd.concat([feat_df, season_df], ignore_index=True)

    # Build training arrays from 2021-2024
    tourney_df = make_tourney_df(M_TEAM_IDS, TRAIN_SEASONS)
    matchup_df = build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)
    train = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]
    X1_tr = np.stack(train["X_team1"].values)
    X2_tr = np.stack(train["X_team2"].values)
    y_tr = train["y"].values.astype(int)

    return feat_df, M_TEAM_IDS, X1_tr, X2_tr, y_tr


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(X1: np.ndarray, X2: np.ndarray, y: np.ndarray):
    """Train and return a fitted LogisticBaseline."""
    from ncaa_models.baseline import LogisticBaseline
    model = LogisticBaseline()
    model.fit(X1, X2, y)
    logger.info("Model trained on %d matchups.", len(y))
    return model


# ---------------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------------

def generate_submission(
    model,
    feat_df: pd.DataFrame,
    team_ids: list,
    season: int,
    output_path: Path,
) -> pd.DataFrame:
    """Generate and save the Kaggle submission CSV."""
    from ncaa_models.baseline import FEATURE_COLS
    from ncaa_models.submit import build_submission, validate_submission

    sub_df = build_submission(
        model=model,
        team_features=feat_df,
        team_ids=team_ids,
        season=season,
        feature_cols=FEATURE_COLS,
        clip_probs=True,
        output_path=str(output_path),
    )

    result = validate_submission(sub_df, expected_season=season)
    if not result["valid"]:
        logger.error("Submission validation failed: %s", result.get("issues"))
        raise ValueError(f"Invalid submission: {result}")

    logger.info(
        "Submission generated: %d rows → %s", len(sub_df), output_path
    )
    return sub_df


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------

def run_certification(
    train_df: pd.DataFrame,
    model,
    X1_val: np.ndarray,
    X2_val: np.ndarray,
    cert_dir: Path,
) -> bool:
    """Run all certification checks and save reports."""
    from ncaa_submission_2026.certification.certifier import PrivacyCertifier

    certifier = PrivacyCertifier()
    report = certifier.certify(
        train_df=train_df,
        model=model,
        X1_val=X1_val,
        X2_val=X2_val,
    )
    cert_dir.mkdir(parents=True, exist_ok=True)
    certifier.save_json(str(cert_dir / "privacy_cert.json"))
    certifier.save_md(str(cert_dir / "privacy_cert.md"))

    if report.passed:
        logger.info("Certification PASSED.")
    else:
        logger.error("Certification FAILED: %s", report.summary)
    return report.passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="FieldSense NCAA 2026 prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Kaggle data directory")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"),
                        help="Output CSV path")
    parser.add_argument("--season", type=int, default=2026,
                        help="Prediction season")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (for testing)")
    parser.add_argument("--certify", action="store_true",
                        help="Run certification checks before prediction")
    parser.add_argument("--cert-dir", type=Path,
                        default=Path("certification"),
                        help="Directory for certification reports")
    args = parser.parse_args(argv)

    # --- Set seeds for deterministic output ---
    set_all_seeds(args.seed)

    try:
        if args.synthetic or args.data_dir is None:
            logger.info("Using synthetic data.")
            feat_df, team_ids, X1_tr, X2_tr, y_tr = load_synthetic_data(args.season)
        else:
            feat_df, team_ids, X1_tr, X2_tr, y_tr = load_kaggle_data(
                args.data_dir, args.season
            )

        # Train model
        set_all_seeds(args.seed)  # re-seed before training
        model = train_model(X1_tr, X2_tr, y_tr)

        # Certification (optional)
        if args.certify:
            # Use first 7 samples as mini validation set
            ok = run_certification(
                train_df=feat_df[feat_df["season"] < args.season],
                model=model,
                X1_val=X1_tr[:7],
                X2_val=X2_tr[:7],
                cert_dir=args.cert_dir,
            )
            if not ok:
                logger.error("Certification failed; aborting prediction.")
                return 2

        # Generate submission
        set_all_seeds(args.seed)  # re-seed before inference for determinism
        sub_df = generate_submission(
            model=model,
            feat_df=feat_df,
            team_ids=team_ids,
            season=args.season,
            output_path=args.output,
        )
        print(f"Submission saved: {args.output} ({len(sub_df)} rows)")
        return 0

    except Exception as e:
        logger.exception("Prediction pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
