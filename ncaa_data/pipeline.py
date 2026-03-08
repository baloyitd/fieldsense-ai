"""
ncaa_data.pipeline
==================
End-to-end orchestration: ingest → normalize → features → output.

The pipeline operates in three stages:

  1. **Ingest** – :func:`~ncaa_data.ingest.load_raw_data` reads all available
     Kaggle CSV files from the raw data directory.

  2. **Normalize** – :func:`~ncaa_data.normalize.normalize_all` flattens
     game-level win/loss rows into team-perspective rows with canonical IDs
     and optional conference / seed metadata.

  3. **Features** – :func:`~ncaa_data.features.compute_features` aggregates
     to one feature row per (gender, season, team) with 28 computed columns.

Output files (written to *out_dir*):
  * ``ncaa_features_{season}.parquet`` – one Parquet per season
  * ``ncaa_features_all.csv``          – concatenated CSV for all seasons
  * ``pipeline_manifest.json``         – metadata / provenance record

Usage::

    from ncaa_data.pipeline import run_pipeline
    feat_df = run_pipeline(
        data_dir="data/ncaa/raw",
        out_dir="data/ncaa/features",
        seasons=range(2021, 2026),
    )
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .ingest import load_raw_data
from .normalize import ConferenceRegistry, normalize_all
from .features import compute_features, FEATURE_COLS, KEY_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    data_dir: str | Path = "data/ncaa/raw",
    out_dir: str | Path = "data/ncaa/features",
    seasons: Optional[Iterable[int]] = None,
    save_parquet: bool = True,
    save_csv: bool = True,
    save_pickle: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run the full NCAA data pipeline.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing raw Kaggle CSV files.
    out_dir : str | Path
        Directory where output feature files will be written.
    seasons : iterable of int, optional
        Seasons to include (e.g. ``range(2021, 2026)``).
        Defaults to seasons found in the data (no filter).
    save_parquet : bool
        Write one ``.parquet`` file per season (requires ``pyarrow`` or
        ``fastparquet``).
    save_csv : bool
        Write a single combined CSV for all seasons.
    save_pickle : bool
        Write ``ncaa_features_all.pkl`` for fast downstream loading.
    dry_run : bool
        If True, skip all file I/O (useful for testing / CI).

    Returns
    -------
    pd.DataFrame
        Feature matrix; shape = (n_team_seasons, n_features + 4 key cols).
    """
    t0 = time.perf_counter()
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    logger.info("=== NCAA Data Pipeline  Stage 01/10 ===")
    logger.info("data_dir : %s", data_dir)
    logger.info("out_dir  : %s", out_dir)

    # ------------------------------------------------------------------
    # Stage 1 – Ingest
    # ------------------------------------------------------------------
    logger.info("[1/3] Ingesting raw data …")
    raw_data = load_raw_data(data_dir)

    # ------------------------------------------------------------------
    # Stage 2 – Normalize
    # ------------------------------------------------------------------
    logger.info("[2/3] Normalizing …")
    registry = ConferenceRegistry()
    games_df = normalize_all(raw_data, registry=registry)

    # ------------------------------------------------------------------
    # Stage 3 – Features
    # ------------------------------------------------------------------
    logger.info("[3/3] Computing features …")
    feat_df = compute_features(games_df, seasons=seasons)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    n_seasons = feat_df["season"].nunique()
    n_teams = len(feat_df)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Pipeline complete: %d team-seasons across %d seasons  (%.1fs)",
        n_teams, n_seasons, elapsed,
    )

    validation_report = build_validation_report(feat_df)
    _print_validation_report(validation_report)

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_outputs(
            feat_df, out_dir,
            save_parquet=save_parquet,
            save_csv=save_csv,
            save_pickle=save_pickle,
        )
        _save_validation_report(validation_report, out_dir)
        _write_manifest(
            out_dir=out_dir,
            data_dir=data_dir,
            n_teams=n_teams,
            n_seasons=n_seasons,
            seasons=sorted(feat_df["season"].unique().tolist()),
            conf_registry=registry.mapping,
            elapsed=elapsed,
            feature_cols=FEATURE_COLS,
        )

    return feat_df


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_outputs(
    feat_df: pd.DataFrame,
    out_dir: Path,
    save_parquet: bool,
    save_csv: bool,
    save_pickle: bool = True,
) -> None:
    """Write per-season Parquet files, combined CSV, and/or a pickle."""

    if save_parquet:
        for season, grp in feat_df.groupby("season"):
            path = out_dir / f"ncaa_features_{season}.parquet"
            grp.reset_index(drop=True).to_parquet(path, index=False)
            logger.info("  Written %s  (%d rows)", path.name, len(grp))

    if save_csv:
        csv_path = out_dir / "ncaa_features_all.csv"
        feat_df.to_csv(csv_path, index=False)
        logger.info("  Written %s  (%d rows)", csv_path.name, len(feat_df))

    if save_pickle:
        pkl_path = out_dir / "ncaa_features_all.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(feat_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("  Written %s  (%d rows)", pkl_path.name, len(feat_df))


def _write_manifest(
    out_dir: Path,
    data_dir: Path,
    n_teams: int,
    n_seasons: int,
    seasons: list,
    conf_registry: dict,
    elapsed: float,
    feature_cols: list,
) -> None:
    """Write a JSON provenance manifest alongside the feature files."""
    manifest = {
        "pipeline_version": "1.0.0",
        "stage": "01/10",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "n_team_seasons": n_teams,
        "n_seasons": n_seasons,
        "seasons": seasons,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "key_cols": KEY_COLS,
        "conference_registry": conf_registry,
        "elapsed_seconds": round(elapsed, 2),
        "target_metric": "Brier Score (lower is better; target < 0.20 baseline)",
        "competition": "2026 NCAA March Machine Learning Mania",
    }
    path = out_dir / "pipeline_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    logger.info("  Written %s", path.name)


# ---------------------------------------------------------------------------
# Convenience: load already-computed features
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def build_validation_report(feat_df: pd.DataFrame) -> dict:
    """
    Build a completeness and range validation report for the feature matrix.

    Returns a dict that is also serialised to ``validation_report.json`` in
    the output directory.

    Checks per-season:
      * team count
      * NaN fraction per feature column
      * fraction of features within expected domain ranges

    Returns
    -------
    dict  (JSON-serialisable)
    """
    from .features import FEATURE_COLS

    # Domain-valid ranges for key features
    _RANGES = {
        "ortg":         (60,  140),
        "drtg":         (60,  140),
        "tempo":        (55,   85),
        "fg_pct":       (0.25, 0.70),
        "fg3_pct":      (0.15, 0.55),
        "ft_pct":       (0.40, 0.95),
        "orb_rate":     (0.10, 0.55),
        "drb_rate":     (0.40, 0.90),
        "win_pct":      (0.00, 1.00),
        "sos":          (0.20, 0.80),
        "net_rtg":      (-60,   60),
        "scoring_margin": (-40, 40),
    }

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_team_seasons": int(len(feat_df)),
        "seasons": {},
        "range_violations": {},
    }

    for season, grp in feat_df.groupby("season"):
        season_info: dict = {
            "team_count": int(len(grp)),
            "gender_counts": grp["gender"].value_counts().to_dict(),
            "completeness": {},
        }
        for col in FEATURE_COLS:
            if col in grp.columns:
                nan_frac = float(grp[col].isnull().mean())
                season_info["completeness"][col] = round(1.0 - nan_frac, 4)
        report["seasons"][int(season)] = season_info

    # Range violations (across all seasons)
    for col, (lo, hi) in _RANGES.items():
        if col not in feat_df.columns:
            continue
        valid = feat_df[col].dropna()
        n_violations = int(((valid < lo) | (valid > hi)).sum())
        if n_violations:
            report["range_violations"][col] = {
                "expected": [lo, hi],
                "n_violations": n_violations,
                "pct_violations": round(100 * n_violations / len(valid), 2),
            }

    total_violations = sum(
        v["n_violations"] for v in report["range_violations"].values()
    )
    report["total_range_violations"] = total_violations
    report["status"] = "PASS" if total_violations == 0 else "WARN"

    return report


def _print_validation_report(report: dict) -> None:
    """Print a compact validation summary to stdout."""
    print("\n" + "=" * 60)
    print(f"Data Validation Report  [{report['status']}]")
    print("=" * 60)
    print(f"  Total team-seasons : {report['total_team_seasons']:,}")
    for season, info in sorted(report["seasons"].items()):
        gc = info["gender_counts"]
        print(f"  Season {season} : {info['team_count']} teams  {gc}")
    if report["range_violations"]:
        print("\n  Range violations detected:")
        for col, v in report["range_violations"].items():
            print(
                f"    {col:<20}  {v['n_violations']} rows  "
                f"({v['pct_violations']:.1f}%)  expected {v['expected']}"
            )
    else:
        print("\n  All features within expected domain ranges.")
    print("=" * 60 + "\n")


def _save_validation_report(report: dict, out_dir: Path) -> None:
    path = out_dir / "validation_report.json"
    path.write_text(json.dumps(report, indent=2))
    logger.info("  Written %s", path.name)


def load_features(
    out_dir: str | Path = "data/ncaa/features",
    seasons: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Load previously-computed feature files from *out_dir*.

    Tries Parquet first; falls back to the combined CSV.

    Parameters
    ----------
    out_dir  : directory written by :func:`run_pipeline`
    seasons  : filter to specific seasons after loading

    Returns
    -------
    pd.DataFrame  (same schema as :func:`run_pipeline` output)
    """
    out_dir = Path(out_dir)
    csv_path = out_dir / "ncaa_features_all.csv"
    parquet_files = sorted(out_dir.glob("ncaa_features_*.parquet"))

    pkl_path = out_dir / "ncaa_features_all.pkl"

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)
    elif parquet_files:
        frames = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(frames, ignore_index=True)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No feature files found in {out_dir}. Run run_pipeline() first."
        )

    if seasons is not None:
        df = df[df["season"].isin(list(seasons))].reset_index(drop=True)

    return df
